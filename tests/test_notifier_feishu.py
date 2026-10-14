# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""R5.2 通知通道测试：飞书 Webhook 通知器 + build_notifier 装配规则。

覆盖：
* 成功推送：POST 一次，请求体是飞书 text 信封（msg_type/content.text），不含金额；
* 网络失败 / HTTP 错误 / 非 JSON 响应 / 业务错误码 → **静默吞掉**（notify 不抛）；
* 空 webhook_url 构造报错（空串应走 LogNotifier 兜底）；
* ``build_notifier`` 只认配置：webhook 空 → LogNotifier；非空 → MultiNotifier(Log+Feishu)；
* MultiNotifier 扇出到全部通道。
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from miaosuan.runtime.notifiers import (
    FeishuNotifier,
    LogNotifier,
    MultiNotifier,
    build_notifier,
)

# ── 假 HTTP 响应 ──────────────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, payload: dict[str, Any] | None = None, status: int = 200) -> None:
        self._payload = payload if payload is not None else {"code": 0, "msg": "success"}
        self.status = status
        self._body = json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _install_urlopen(monkeypatch: pytest.MonkeyPatch, capture: dict[str, Any]) -> None:
    """把 urllib.request.urlopen 换成记录器；返回捕获的 (url, data, headers)。"""

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResp:
        capture["url"] = request.full_url
        capture["data"] = json.loads(request.data.decode("utf-8"))
        capture["headers"] = dict(request.headers)
        capture["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


# ── FeishuNotifier ──────────────────────────────────────────────────────────


def test_feishu_posts_text_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict[str, Any] = {}
    _install_urlopen(monkeypatch, capture)
    n = FeishuNotifier("https://example.com/hook/abc")

    n.notify("信号转折 XAUUSD.H1", "多→空 pos=-0.25", level="info")

    assert capture["url"] == "https://example.com/hook/abc"
    assert capture["data"]["msg_type"] == "text"
    text = capture["data"]["content"]["text"]
    assert "信号转折 XAUUSD.H1" in text
    assert "多→空 pos=-0.25" in text
    assert "timeout" in capture and capture["timeout"] == 5.0
    # urllib 头键大小写不保证；按小写比较。
    headers = {k.lower(): v for k, v in capture["headers"].items()}
    assert headers["content-type"].startswith("application/json")


def test_feishu_rejects_empty_webhook() -> None:
    with pytest.raises(ValueError):
        FeishuNotifier("")
    with pytest.raises(ValueError):
        FeishuNotifier("   ")


def test_feishu_swallows_network_error(monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    def boom(request: Any, timeout: float | None = None) -> _FakeResp:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    n = FeishuNotifier("https://example.com/hook/abc")
    n.notify("title", "body")  # 不抛
    assert "飞书通知失败" in caplog.text


def test_feishu_swallows_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: Any, timeout: float | None = None) -> _FakeResp:
        raise urllib.error.HTTPError("url", 429, "rate limited", None, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    FeishuNotifier("https://example.com/hook/abc").notify("t", "b")  # 不抛


def test_feishu_swallows_business_error(monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    def biz_err(request: Any, timeout: float | None = None) -> _FakeResp:
        return _FakeResp({"code": 190001, "msg": "invalid token"})

    monkeypatch.setattr("urllib.request.urlopen", biz_err)
    FeishuNotifier("https://example.com/hook/abc").notify("t", "b")  # 不抛
    assert "业务错误" in caplog.text


def test_feishu_swallows_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadResp(_FakeResp):
        def read(self) -> bytes:
            return b"<html>gateway error</html>"

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _BadResp({}, status=502)
    )
    FeishuNotifier("https://example.com/hook/abc").notify("t", "b")  # 不抛


# ── build_notifier（只认配置）───────────────────────────────────────────────


def test_build_notifier_empty_webhook_returns_log_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "miaosuan.settings.get_config",
        _config_with_webhook(""),
    )
    n = build_notifier()
    assert isinstance(n, LogNotifier)


def test_build_notifier_configured_returns_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "miaosuan.settings.get_config",
        _config_with_webhook("https://open.feishu.cn/hook/x"),
    )
    n = build_notifier()
    assert isinstance(n, MultiNotifier)
    kinds = [type(ch).__name__ for ch in n.channels]
    assert kinds == ["LogNotifier", "FeishuNotifier"]


# ── MultiNotifier ────────────────────────────────────────────────────────────


def test_multi_notifier_fans_out(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _Rec(LogNotifier):
        def __init__(self, tag: str) -> None:
            self._tag = tag

        def notify(self, title: str, body: str, *, level: str = "info") -> None:
            seen.append(self._tag)

    mn = MultiNotifier(_Rec("a"), _Rec("b"))
    mn.notify("t", "b", level="warning")
    assert seen == ["a", "b"]


# ── 测试装置 ───────────────────────────────────────────────────────────────


def _config_with_webhook(webhook: str) -> Any:
    """构造一个只改 notifiers.feishu.webhook_url 的 Settings（其余用默认）。"""
    from miaosuan.settings import Settings

    cfg = Settings()
    cfg.notifiers.feishu.webhook_url = webhook
    return lambda: cfg