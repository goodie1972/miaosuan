"""WebUI 仪表盘接口测试（``src/miaosuan/webui/server.py``）。

为什么不用 ``fastapi.testclient.TestClient``
--------------------------------------------
``TestClient`` 依赖 ``httpx``，而本仓库并未声明 / 安装它（web 栈只有
``fastapi`` + ``uvicorn``）。为避免引入新依赖，这里用 ``asyncio`` + 裸 ASGI
``(scope, receive, send)`` 三元组直接驱动 app：走的仍是**真实**的路由分发与
异常转换（``HTTPException`` → 状态码 + ``{"detail": ...}``），只把传输层换成内存。

覆盖（回归防线）

* 未知标的 + 无 market → **400**（缺陷根因：以前会把 CLI 子进程的 Python
  堆栈直接甩给前端用户看）
* 未知标的 + 显式 market → 放行（不挡非标品种）
* 已知标的 ``XAUUSD`` → 放行，且**不**注入 ``--market``（保持自动推断）
* ``/api/specs`` 能列出**不含** ``spec_id`` 键的 spec JSON（回归 511581e）
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from typing import Any

import pytest

from miaosuan.webui import server

# ── 零依赖 ASGI 客户端 ──────────────────────────────────────────────────────

def _call(
    app: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """用内存 ASGI 调用 app，返回 ``(状态码, 解析后的 JSON)``。"""
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    # ASGI 把 path 与 query 分开传：路由匹配只看 path，query 单独放 query_string。
    # 不分家的话 "/api/inspect?path=x" 会整串当 path，匹配不到路由直接 404。
    path_only, _, query = path.partition("?")
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path_only,
        "raw_path": path.encode("utf-8"),
        "query_string": query.encode("utf-8"),
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def drive() -> None:
        await app(scope, receive, send)

    asyncio.run(drive())

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return int(status), (json.loads(raw.decode("utf-8")) if raw else None)


class _FakeProc:
    """``subprocess.Popen`` 替身：不真跑挖掘（很慢），但让 pump 线程正常收尾。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.stdout: list[str] = []
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        """立即"结束"，返回 0。"""
        return self.returncode


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """隔离产物目录 + 桩掉子进程，返回全新的 app。"""
    monkeypatch.setattr(server, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(server.subprocess, "Popen", _FakeProc)
    server._jobs._job = None
    created = server.create_app()
    yield created
    server._jobs._job = None


# ── 前置校验：未知标的必须拦成 400 ──────────────────────────────────────────

def test_unknown_symbol_without_market_returns_400(app: Any) -> None:
    """未知标的 + 无 market → 400，且文案给出可操作指引（而不是 Python 堆栈）。"""
    status, data = _call(app, "POST", "/api/mine", {
        "data": "data/TESTXAU_H1.parquet",
        "symbol": "TESTXAU",
        "timeframe": "H1",
    })
    assert status == 400
    detail = data["detail"]
    assert "市场画像" in detail
    assert "TESTXAU" in detail
    assert "XAUUSD" in detail


def test_empty_symbol_without_market_returns_400(app: Any) -> None:
    """标的与 market 都为空 → 同样拦成 400（pipeline 一样会报错）。"""
    status, data = _call(app, "POST", "/api/mine", {"data": "data/TESTXAU_H1.parquet"})
    assert status == 400
    assert "市场画像" in data["detail"]


def test_unknown_symbol_with_explicit_market_is_allowed(app: Any) -> None:
    """显式指定 market 时放行，并且必须透传给 CLI 子进程。"""
    status, data = _call(app, "POST", "/api/mine", {
        "data": "data/TESTXAU_H1.parquet",
        "symbol": "TESTXAU",
        "market": "FOREX_XAUUSD",
    })
    assert status == 200, data
    assert data["job_id"]
    assert "--market" in data["cmd"]
    assert "FOREX_XAUUSD" in data["cmd"]


def test_known_symbol_without_market_is_allowed(app: Any) -> None:
    """已知标的 XAUUSD 走自动推断，不应注入 --market。"""
    status, data = _call(app, "POST", "/api/mine", {
        "data": "data/TESTXAU_H1.parquet",
        "symbol": "XAUUSD",
    })
    assert status == 200, data
    assert "--market" not in data["cmd"]
    assert "--symbol" in data["cmd"]


# ── 单一数据源：不重复硬编码字符串 ──────────────────────────────────────────

def test_known_symbols_and_profiles_come_from_single_source() -> None:
    """已知标的 / 可选画像都从各自唯一来源派生，避免两处字符串漂移。"""
    from miaosuan.pipeline import _SYMBOL_PROFILE_FALLBACK

    assert server._SYMBOL_PROFILE_FALLBACK is _SYMBOL_PROFILE_FALLBACK
    assert "XAUUSD" in server._SYMBOL_HINT
    assert "BTCUSDT" in server._SYMBOL_HINT
    for name in server.EXPECTED_PROFILE_NAMES:
        assert name in server._PROFILE_HINT


# ── 回归：/api/specs 不能因为缺 spec_id 键就吞掉历史产物 ───────────────────

def test_specs_lists_spec_without_spec_id_key(app: Any, tmp_path: Any) -> None:
    """只含 ``payload`` + ``spec_version``（**无** ``spec_id``）的 spec 也要能列出。

    回归 511581e：spec 列表曾用错的 schema 过滤，导致历史产物全部不可见。
    """
    (tmp_path / "spec_no_id.json").write_text(
        json.dumps({
            "payload": {"kind": "factor", "tokens": [51, 121, 53]},
            "spec_version": "1.0",
        }),
        encoding="utf-8",
    )
    status, data = _call(app, "GET", "/api/specs")
    assert status == 200
    files = [row["file"] for row in data]
    assert "spec_no_id.json" in files
    row = next(r for r in data if r["file"] == "spec_no_id.json")
    assert row["spec_id"]          # 列表接口自己派生短 id，不依赖文件里的键
    assert row["n_tokens"] == 3


# ── T07：AlgoForge 双主题改造后的新契约 ──────────────────────────────────────

def test_meta_exposes_real_frozen_vocab(app: Any) -> None:
    """/api/meta 必须回报**真实**冻结词表，页头不能写死数字。

    回归防线：页头"词表 v9217a2c0d91a · 127 tokens"若被硬编码，词表一变就会
    和内核不一致；这里强制它从 :data:`VOCAB_VERSION` / ``FORMULA_VOCAB`` 派生。
    """
    from miaosuan.core.vocab import FORMULA_VOCAB, VOCAB_VERSION

    status, data = _call(app, "GET", "/api/meta")
    assert status == 200
    assert data["vocab_version"] == VOCAB_VERSION == "v9217a2c0d91a"
    assert data["n_tokens"] == FORMULA_VOCAB.size
    assert data["n_features"] == FORMULA_VOCAB.feature_count
    assert data["n_operators"] == len(FORMULA_VOCAB.operator_names)
    assert "XAUUSD" in data["known_symbols"]
    assert "FOREX_XAUUSD" in data["profiles"]


def test_inspect_rejects_path_outside_data_dirs(app: Any) -> None:
    """/api/inspect 必须拒绝数据目录之外的路径（防目录穿越读到任意文件）。"""
    status, data = _call(app, "GET", "/api/inspect?path=" + "C:/Windows/System32/config/SAM")
    assert status == 400
    assert "数据目录" in data["detail"]


def test_inspect_reports_404_for_missing_in_dir_file(app: Any) -> None:
    """目录内但文件不存在 → 404（区别于"路径非法"的 400）。"""
    missing = str(server._REPO_ROOT / "data" / "__definitely_missing__.parquet")
    status, data = _call(app, "GET", "/api/inspect?path=" + urllib.parse.quote(missing))
    assert status == 404
    assert "不存在" in data["detail"]


def test_strategies_lists_only_real_exports(app: Any, tmp_path: Any) -> None:
    """只列带 ``STRATEGY_MAGIC`` 的文件 —— artifacts 下的临时脚本不算已导出策略。

    回归防线：该常量由导出模板强制写入，是"这是导出产物"的准确判据。若不过滤，
    artifacts 里的调试脚本会被当成策略列出，而且每个都带 4 条 lint error（因为
    lint 规则要求声明该常量），把真实导出淹掉。
    """
    (tmp_path / "real_export.py").write_text(
        'STRATEGY_MAGIC = "661801"\nSTRATEGY_NAME = "x"\n', encoding="utf-8"
    )
    (tmp_path / "scratch_debug.py").write_text("print(1)\n", encoding="utf-8")

    status, data = _call(app, "GET", "/api/strategies")
    assert status == 200
    files = [row["file"] for row in data]
    assert "real_export.py" in files
    assert "scratch_debug.py" not in files
    row = next(r for r in data if r["file"] == "real_export.py")
    assert row["magic"] == "661801"
    assert "lint_errors" in row and "lint_warnings" in row


def test_specs_surfaces_symbol_and_timeframe(app: Any, tmp_path: Any) -> None:
    """/api/specs 要带 symbol / timeframe（⑧ 历史 Spec 表格依赖这两列）。"""
    (tmp_path / "spec_x.json").write_text(
        json.dumps({
            "payload": {"tokens": [51, 121, 53], "vocab_version": "v9217a2c0d91a"},
            "spec_version": "1.0",
            "semantics": {"timeframe": "H1"},
            "provenance": {"config_snapshot": {"symbol": "XAUUSD"}},
        }),
        encoding="utf-8",
    )
    status, data = _call(app, "GET", "/api/specs")
    assert status == 200
    row = next(r for r in data if r["file"] == "spec_x.json")
    assert row["symbol"] == "XAUUSD"
    assert row["timeframe"] == "H1"


def test_history_endpoint_returns_real_points(app: Any, tmp_path: Any) -> None:
    """/api/history 回传 sidecar 原文，供训练曲线画真实逐代曲线。"""
    (tmp_path / "spec_1.history.json").write_text(
        json.dumps({
            "version": 1,
            "spec": "artifacts/spec_1.json",
            "budget": "quick",
            "stop_reason": "MAX_GENERATIONS",
            "generations": 2,
            "n_evaluations": 20,
            "points": [
                {"generation": 0, "best": 1.0, "mean": 0.5, "diversity": 0.4, "n_evaluations": 10},
                {"generation": 1, "best": 2.0, "mean": 1.0, "diversity": 0.3, "n_evaluations": 20},
            ],
        }),
        encoding="utf-8",
    )
    status, data = _call(app, "GET", "/api/history?name=spec_1.history.json")
    assert status == 200
    assert len(data["points"]) == 2
    assert data["points"][-1]["best"] == 2.0


def test_history_endpoint_404_for_missing_file(app: Any) -> None:
    """没有对应 sidecar → 404（前端按"无历史"降级，不报错）。"""
    status, data = _call(app, "GET", "/api/history?name=__nope__.history.json")
    assert status == 404
    assert "不存在" in data["detail"]


def test_history_endpoint_rejects_traversal(app: Any) -> None:
    """路径穿越必须被挡（与 /api/inspect 同一套约束）。"""
    status, _ = _call(app, "GET", "/api/history?name=..%2F..%2Fpyproject.toml")
    assert status in (400, 404)


# ── 静态页面：单文件 + 零 CDN + 双主题（防止后来人引入外链或砍掉浅色）────────

def test_index_html_is_single_file_zero_cdn_dual_theme() -> None:
    """页头/主题/图表都在一个文件里，且不依赖任何外部资源。

    回归防线：一旦有人引 CDN（Chart.js / 字体），离线环境会整页白屏；这里从
    静态层面把"零外链"钉死。双主题则要求两套 CSS 变量块都还在。
    """
    html = server._INDEX_HTML.read_text(encoding="utf-8")
    assert "<script" in html and html.count("<script") == 1
    assert 'src="http' not in html and "src='http" not in html
    assert '<link' not in html  # 零外链样式表
    assert 'html[data-theme="dark"]' in html
    assert 'html[data-theme="light"]' in html
    assert "localStorage" in html
    # 涨绿跌红（Binance 口径），不要把方向写反
    assert "#0ecb81" in html and "#f6465d" in html
    # 浅色底是用户拍板的**纯白**（曾为 #f5f7fa 灰底），层次靠边框 + 三级表面撑
    assert "--bg: #ffffff" in html
    assert "--bg: #f5f7fa" not in html
    # 卡片不能刷成纯白，否则白底白卡片糊成一片（--input-bg 仍用 #f5f7fa 属正常）
    assert "--card-grad: linear-gradient(135deg, #ffffff" not in html
    # 三步导航：挖掘 / 回测 / 实时
    for step in ("01", "02", "03"):
        assert step in html
    # 训练曲线：真实逐代历史 + 无历史时的优雅降级（不能画空图 / 不能报错）
    assert "renderCurve" in html
    assert "暂无逐代历史" in html
    assert "/api/history" in html
