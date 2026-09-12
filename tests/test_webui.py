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
import urllib.error
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


def test_backtest_endpoint_rejects_unknown_spec(app: Any) -> None:
    """/api/backtest 必须先校验 spec 文件存在（防目录穿越 + 防空跑子进程）。"""
    status, data = _call(app, "POST", "/api/backtest", {"spec": "__nope__.json", "data": "x.csv"})
    assert status == 404
    assert "不存在" in data["detail"]


def test_backtests_lists_previous_runs(app: Any, tmp_path: Any) -> None:
    """/api/backtests 列出历史回测（供回测页复用 / 追溯）。"""
    (tmp_path / "backtest_20260912_010101.json").write_text(
        json.dumps({
            "meta": {"symbol": "XAUUSD", "timeframe": "H1", "n_bars": 8000},
            "summary": {"sharpe": 0.9762, "total_return": 0.1893},
        }),
        encoding="utf-8",
    )
    status, data = _call(app, "GET", "/api/backtests")
    assert status == 200
    row = next(r for r in data if r["file"] == "backtest_20260912_010101.json")
    assert row["symbol"] == "XAUUSD"
    assert row["n_bars"] == 8000
    assert row["sharpe"] == 0.9762


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
    # 回测页已实装（不再是占位）：有运行入口、走 /api/backtest，且带口径警告
    assert '/api/backtest' in html
    assert "运行回测" in html
    # 03 实时页已实装（不再是占位）：只读接入 AlgoForge 后端，占位说明已移除
    assert "待接入 · 计划中" not in html
    assert "只读接入" in html
    assert "绝不下单" in html            # 红线声明必须留在页面上
    assert "后端不可达" in html          # 离线降级文案（不显示 0 / 空表）
    for ep in ("/api/realtime/status", "/api/realtime/price",
               "/api/realtime/candles", "/api/realtime/signals"):
        assert ep in html
    # 红线：实时页绝不含任何写操作入口（下单 / 平仓 / 启停引擎）
    for forbidden in ("/api/order", "/api/orders", "/api/trade",
                      "/api/close", "/api/engine/start", "/api/engine/stop"):
        assert forbidden not in html
    # 口径铁律：val_score 是搜索适应度，不得当绩效展示
    assert "不是绩效指标" in html or "不是绩效" in html


def test_index_html_has_no_inline_onclick_filename_injection() -> None:
    """回归 B4：文件名不得拼进 ``onclick`` 的 JS 字符串上下文（注入风险）。

    ``onclick="loadSpecByName('<file>')"`` 里，HTML 转义（``&#39;``）会被浏览器在
    解析 attribute 时**先解码再交给 JS 引擎**，文件名里的引号因此能逃出字符串 →
    注入。改为 ``data-file`` + 事件委托（``getAttribute``）后不存在该上下文。
    """
    html = server._INDEX_HTML.read_text(encoding="utf-8")
    assert 'onclick="loadSpecByName' not in html
    assert "data-file=" in html
    assert 'closest("button[data-file]")' in html


# ── 03 实时页：只读代理端点 ────────────────────────────────────────────────
#
# 红线（用户明确要求，且其 MT4 正在跑实盘）：**只读，绝不下单**。
# 这里在**端点层**把红线钉死：写方法一律 405、离线降级醒目（ok=false 且 data
# 为 None，绝不伪造 0），只读白名单越界在客户端层还会直接抛异常（见
# ``tests/test_algoforge_realtime.py``）。


class _FakeResp:
    """``urlopen`` 返回体替身：支持 ``with`` + ``.status`` + ``.read()``。"""

    def __init__(self, payload: Any, status: int = 200) -> None:
        self.status = status
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def _install_realtime(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[str]:
    """把 ``server._realtime_client`` 换成 ``_open`` 被替换的只读客户端。

    这样端点走的是**真实**路由 / ``safe_call`` / ``_outcome_payload`` 逻辑，只把
    最底层的一次 socket 换成内存桩（避免真连后端、避免 3s 超时拖慢测试）。

    Args:
        monkeypatch: pytest 的 monkeypatch fixture。
        handler: 接收 ``urllib.request.Request``，返回响应或抛异常。

    Returns:
        记录每次请求完整 URL 的列表（用于断言查询参数是否透传）。
    """
    from miaosuan.adapters.algoforge.realtime import AlgoforgeReadOnlyClient

    seen: list[str] = []
    client = AlgoforgeReadOnlyClient(base_url="http://127.0.0.1:1783", timeout=2.0)

    def _open(request: Any) -> Any:
        seen.append(request.full_url)
        return handler(request)

    monkeypatch.setattr(client, "_open", _open)
    monkeypatch.setattr(server, "_realtime_client", lambda: client)
    return seen


def test_realtime_status_degrades_loudly_when_backend_down(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后端不可达 → 200 + ok=false + 原因码，``data`` 为 ``None``（**绝不伪造 0**）。"""
    def _boom(request: Any) -> Any:
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    _install_realtime(monkeypatch, _boom)
    status, data = _call(app, "GET", "/api/realtime/status")
    assert status == 200
    assert data["ok"] is False
    assert data["reason"] == "unreachable"
    assert data["data"] is None          # 关键：不是 0、不是空字典
    assert data["error"]                 # 有可读原因
    assert data["backend"] == "http://127.0.0.1:1783"


def test_realtime_reflects_timeout_reason(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """超时与不可达要能被前端区分（reason 码不同），都走 ok=false 降级。"""
    def _timeout(request: Any) -> Any:
        raise urllib.error.URLError(TimeoutError("timed out"))

    _install_realtime(monkeypatch, _timeout)
    status, data = _call(app, "GET", "/api/realtime/status")
    assert status == 200
    assert data["ok"] is False
    assert data["reason"] == "timeout"


def test_realtime_http_error_is_reported_not_swallowed(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后端返回非 2xx → ok=false + http_error + 状态码（不静默成"正常"）。"""
    def _err(request: Any) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", {}, None
        )

    _install_realtime(monkeypatch, _err)
    status, data = _call(app, "GET", "/api/realtime/status")
    assert status == 200
    assert data["ok"] is False
    assert data["reason"] == "http_error"
    assert data["status"] == 503


def test_realtime_price_returns_real_quote(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后端在线 → 原样回传报价（不加工、不伪造）。"""
    quote = {"symbol": "XAUUSD", "bid": 2031.45, "ask": 2031.75, "spread": 30}
    _install_realtime(monkeypatch, lambda request: _FakeResp(quote))
    status, data = _call(app, "GET", "/api/realtime/price")
    assert status == 200
    assert data["ok"] is True
    assert data["reason"] == "ok"
    assert data["data"] == quote
    assert data["data"]["bid"] == 2031.45  # 真实数字原样透传


def test_realtime_candles_forwards_symbol_timeframe_count(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K 线端点必须把 symbol/timeframe/count 透传给只读客户端。"""
    bars = [{"close": 2031.0}, {"close": 2032.0}]
    seen = _install_realtime(monkeypatch, lambda request: _FakeResp(bars))
    status, data = _call(
        app, "GET", "/api/realtime/candles?symbol=EURUSD&timeframe=M15&count=50"
    )
    assert status == 200
    assert data["ok"] is True
    assert len(seen) == 1
    q = urllib.parse.parse_qs(urllib.parse.urlparse(seen[0]).query)
    assert q["symbol"] == ["EURUSD"]
    assert q["timeframe"] == ["M15"]
    assert q["count"] == ["50"]


def test_realtime_latest_requires_strategy_and_sends_no_request(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺 ``strategy`` → 400，且**一个请求都不发**（校验先于网络）。"""
    seen = _install_realtime(monkeypatch, lambda request: _FakeResp({}))
    status, data = _call(app, "GET", "/api/realtime/latest")
    assert status == 400
    assert "strategy" in data["detail"]
    assert seen == []


def test_realtime_latest_forwards_strategy(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """带 ``strategy`` → 透传并回传。"""
    seen = _install_realtime(monkeypatch, lambda request: _FakeResp({"signal": "BUY"}))
    status, data = _call(app, "GET", "/api/realtime/latest?strategy=ExpForge_x")
    assert status == 200
    assert data["ok"] is True
    q = urllib.parse.parse_qs(urllib.parse.urlparse(seen[0]).query)
    assert q["strategy"] == ["ExpForge_x"]


def test_realtime_endpoints_reject_non_get(app: Any) -> None:
    """写方法一律 405：实时页**不存在**任何可写的后端入口（红线可证明）。"""
    for verb in ("POST", "PUT", "DELETE", "PATCH"):
        status, _ = _call(app, verb, "/api/realtime/status", {})
        assert status == 405, f"{verb} 竟然被放行"


def test_realtime_client_base_url_is_configurable(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """后端地址可配置（环境变量优先），默认 ``http://127.0.0.1:1783``。"""
    monkeypatch.delenv("MIAOSUAN_ALGOFORGE_URL", raising=False)
    assert server._realtime_client().base_url == "http://127.0.0.1:1783"
    monkeypatch.setenv("MIAOSUAN_ALGOFORGE_URL", "http://10.0.0.9:8080/")
    assert server._realtime_client().base_url == "http://10.0.0.9:8080"
