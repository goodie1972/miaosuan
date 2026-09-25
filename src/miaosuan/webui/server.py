# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""妙算本地 Web 仪表盘（模式 A 可视化）。

职责边界（延续架构依赖铁律）：

* 本模块是**唯一**允许被 CLI 之外的入口启动的 IO 层，与 :mod:`miaosuan.cli` 同级；
* 所有挖掘/导出/校验都通过**子进程调用 CLI**（``python -m miaosuan.cli ...``），
  不直接 import pipeline —— 保证 UI 展示的输出与命令行完全一致，也避免
  长任务把 Web 事件循环卡死；
* 同一时刻只允许一个 ``mine`` 任务（简单互斥），日志按行增量推送。

安全约定：仅绑定 ``127.0.0.1``；文件参数一律做白名单校验（只允许
``artifacts/`` 下的产物名与数据目录下的行情文件，拒绝任何路径分隔符）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..adapters.shenji.generator import readable_formula
from ..adapters.shenji.lint import lint_source
from ..adapters.shenji.realtime import (
    ShenjiReadOnlyClient,
    RealtimeOutcome,
    env_config,
    safe_call,
)
from ..config import is_bundled, kline_data_dir
from ..core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from ..data.loader import load
from ..ir.provenance import MIAOSUAN_VERSION
from ..market.profiles import EXPECTED_PROFILE_NAMES, all_profiles
from ..pipeline import _SYMBOL_PROFILE_FALLBACK

__all__ = ["create_app"]

#: 仓库根（``src/miaosuan/webui/server.py`` 往上 3 层）。
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
#: 产物目录。
_ARTIFACTS: Path = _REPO_ROOT / "artifacts"
#: 静态页面。
_INDEX_HTML: Path = Path(__file__).resolve().parent / "static" / "index.html"
#: 默认行情数据目录（用户在本机放置 TradingView 拉取的 parquet/csv）。
#: 首项经 :func:`~miaosuan.config.kline_data_dir` 读取，可用环境变量
#: ``MIAOSUAN_KLINE_DIR`` 覆盖——**不在本模块直接读 env**（依赖方向铁律）。
_DATA_DIRS: tuple[Path, ...] = (
    Path(kline_data_dir()),
    _REPO_ROOT / "data",
)

_DATA_SUFFIXES = {".parquet", ".csv"}
_EXCLUDED_JSON = {"magic_registry.json", "holdout_seals.json", "config_snapshot.json"}


# ── 工具 ─────────────────────────────────────────────────────────────────────


def _safe_name(name: str, *, suffixes: set[str] | None = None) -> Path:
    """把用户传入的文件名约束为 artifacts 下的单个文件（拒绝路径穿越）。"""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail=f"非法文件名：{name!r}")
    path = (_ARTIFACTS / name).resolve()
    if _ARTIFACTS.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail=f"文件不在产物目录内：{name!r}")
    if suffixes is not None and path.suffix.lower() not in suffixes:
        raise HTTPException(status_code=400, detail=f"仅允许 {sorted(suffixes)}：{name!r}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在：{name}")
    return path


def _cli_process(*args: str) -> list[str]:
    """构造 CLI 子进程命令（用当前解释器，保证与 venv 一致）。

    PyInstaller bundle 模式（``is_bundled()`` 为 True）下：
    ``sys.executable`` 指向 launcher exe 本身，需直接传递子命令参数，
    由 launcher.main 的 ``_run_cli()`` 接管路由。
    """
    if is_bundled():
        return [sys.executable, *args]
    return [sys.executable, "-m", "miaosuan.cli", *args]


def _spec_id(path: Path) -> str:
    """从 spec 文件内容派生稳定短 id（内容寻址：payload + semantics 的 SHA256 前 16 位）。

    与 :data:`StrategySpec.spec_id` 一致：对 payload + semantics + spec_version
    做 JSON 规范序列化后取哈希，而非对文件字节取哈希。这样导出的 .py
    里内嵌的 spec_id 才能与注册表里的条目匹配。
    """
    import hashlib

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 兜底：文件读不出来或解析失败时退回文件哈希
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    payload = data.get("payload") or {}
    semantics = data.get("semantics") or {}
    spec_version = data.get("spec_version", "1.0")
    core = {
        "payload": payload,
        "semantics": semantics,
        "spec_version": spec_version,
    }
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


#: 可选市场画像提示（唯一来源 :data:`EXPECTED_PROFILE_NAMES`，此处不复制字符串）。
_PROFILE_HINT: str = " / ".join(EXPECTED_PROFILE_NAMES)
#: 已知标的提示（唯一来源 ``pipeline._SYMBOL_PROFILE_FALLBACK``，此处不复制字符串）。
_SYMBOL_HINT: str = " / ".join(sorted(_SYMBOL_PROFILE_FALLBACK))


def _resolve_data_file(path: str) -> Path:
    """把行情文件路径约束到已知数据目录内（拒绝任意路径 / 路径穿越）。

    与 :func:`_safe_name`（约束 ``artifacts/`` 下的产物）对应，这里约束的是
    **输入**行情文件：只允许 :data:`_DATA_DIRS` 下的真实文件。
    """
    try:
        resolved = Path(path).resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"非法路径：{path!r}") from exc
    for directory in _DATA_DIRS:
        root = directory.resolve()
        if resolved == root or root in resolved.parents:
            return resolved
    allowed = " / ".join(str(d) for d in _DATA_DIRS)
    raise HTTPException(
        status_code=400, detail=f"行情文件不在允许的数据目录内：{path!r}（允许 {allowed}）"
    )


# ── 起止时间轻量探查 ────────────────────────────────────────────────────────

#: 时间展示格式。**必须与 /api/inspect 同口径**（``%Y-%m-%d`` + UTC），
#: 否则同一文件在下拉框和「读取元信息」里会差一天。
_TIME_FMT = "%Y-%m-%d"


def _fmt_utc(stamp: int) -> str:
    """把 Unix 秒格式化为 UTC 日期（与 :data:`_TIME_FMT` 绑定）。"""
    return time.strftime(_TIME_FMT, time.gmtime(int(stamp)))


def _range_from_table(table: Any) -> tuple[str, str] | None:
    """从只含 ``time`` 列的 Arrow 表取 min/max，格式化为 ``(start, end)``。"""
    import pyarrow.compute as pc

    if table.num_rows == 0:
        return None
    col = table.column("time")
    lo = pc.min(col).as_py()
    hi = pc.max(col).as_py()
    if lo is None or hi is None:
        return None
    return _fmt_utc(int(lo)), _fmt_utc(int(hi))


def _read_time_range_uncached(path: Path) -> tuple[str, str] | None:
    """实际读盘取时间范围；**任何异常一律降级为 ``None``**（不向上抛）。

    * ``.parquet``：用 pyarrow **只读 ``time`` 一列**，再走向量化 min/max。
      刻意不走 :func:`~miaosuan.data.loader.load`（那是 ``/api/inspect`` 的重路径，
      会全量加载并算指纹），否则文件一多 ``/api/data`` 会被拖垮。
    * ``.csv``：尽力而为——同样只读 ``time`` 列；若列名不叫 ``time``（或值是
      日期字符串而非 Unix 秒）则拿不到，返回 ``None``（前端显示 ``—``）。
      取舍说明：CSV 不是主格式（生产落盘一律 parquet），为它引入额外探测
      逻辑不划算，宁可显示 ``—`` 也不给错的时间。
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            return _range_from_table(pq.read_table(str(path), columns=["time"]))
        if suffix == ".csv":
            import pyarrow.csv as pa_csv

            opts = pa_csv.ConvertOptions(include_columns=["time"])
            return _range_from_table(pa_csv.read_csv(str(path), convert_options=opts))
    except Exception:  # noqa: BLE001 - 单文件读失败绝不能让 /api/data 整体 500
        return None
    return None


#: 进程内缓存：``(路径, mtime_ns, size) -> (start, end) | None``。
#: 文件没变（mtime + size 均未变）就不重读 parquet——``/api/data`` 会被前端
#: 频繁调用（挖掘完成 / 数据获取后自动刷新），每次全量读一遍列不划算。
_TIME_RANGE_CACHE: dict[tuple[str, int, int], tuple[str, str] | None] = {}
#: 缓存上限，超出整体清空（简单兜底，避免目录巨多时无限增长）。
_TIME_RANGE_CACHE_MAX = 512


def _peek_time_range(path: Path) -> tuple[str, str] | None:
    """读取行情文件起止时间（带进程内缓存），失败返回 ``None``。"""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
    if key in _TIME_RANGE_CACHE:
        return _TIME_RANGE_CACHE[key]
    value = _read_time_range_uncached(path)
    if len(_TIME_RANGE_CACHE) >= _TIME_RANGE_CACHE_MAX:
        _TIME_RANGE_CACHE.clear()
    _TIME_RANGE_CACHE[key] = value
    return value


def _strategy_magic(source: str) -> str:
    """从策略源码里取 ``STRATEGY_MAGIC`` 常量值；取不到返回空串。

    Args:
        source: 策略 ``.py`` 源码全文。

    Returns:
        六位 magic 的**文本形式**（供表格展示）。源码里该常量是裸 int
        （``STRATEGY_MAGIC = 661801``），这里保留字符串只是为了显示与比较；
        非导出产物（未声明该常量）返回 ``""``。
    """
    for line in source.splitlines():
        if line.startswith("STRATEGY_MAGIC"):
            return line.partition("=")[2].strip().strip('"')
    return ""


def _validate_market_choice(symbol: str, market: str) -> None:
    """在起子进程**之前**拦掉"必然失败"的标的/画像组合。

    根因：:func:`miaosuan.pipeline.resolve_market` 对未知标的**刻意**抛
    :class:`ConfigError`（"未知标的报错而不是猜"，避免用错成本 / 年化因子）。
    但该错误发生在 CLI 子进程里，UI 侧若不前置校验，用户只会看到一段 Python
    堆栈。这里把它前移成 ``400`` + 可操作文案，**判定语义与 pipeline 完全一致**
    （显式 market 一律放行，因此不会挡住非标品种）。

    Args:
        symbol: 表单里的标的代码（可为空）。
        market: 表单里显式选择的市场画像（可为空）。

    Raises:
        HTTPException: 400 —— 既未给 ``market``，标的又不在已知集合内。
    """
    if market:
        return
    if symbol and symbol.upper() in _SYMBOL_PROFILE_FALLBACK:
        return
    if not symbol:
        raise HTTPException(
            status_code=400,
            detail=f"请填写标的（{_SYMBOL_HINT}），或显式选择市场画像（{_PROFILE_HINT}）",
        )
    raise HTTPException(
        status_code=400,
        detail=(
            f"未知标的 {symbol!r}：请填写已知标的（{_SYMBOL_HINT}），"
            f"或显式选择市场画像（{_PROFILE_HINT}）"
        ),
    )


# ── 实时页：妙算 后端**只读**代理 ───────────────────────────────────────


def _realtime_client() -> ShenjiReadOnlyClient:
    """构造只读客户端（地址/超时每次读环境变量，便于运维改配置）。

    默认 ``http://127.0.0.1:1783``（妙算 dashboard），超时 3 秒。
    **只读**白名单在客户端层强制（见 :mod:`miaosuan.adapters.shenji.realtime`）。
    """
    base_url, timeout = env_config()
    return ShenjiReadOnlyClient(base_url=base_url, timeout=timeout)


def _outcome_payload(outcome: RealtimeOutcome, base_url: str) -> dict[str, Any]:
    """把只读结果包装成前端信封（失败时带原因，**不含任何伪造的 0/空值**）。"""
    return {
        "ok": outcome.ok,
        "backend": base_url,
        "reason": outcome.reason,
        "error": outcome.error,
        "status": outcome.status,
        "data": outcome.data,
    }


# ── 挖掘任务（单飞 + 增量日志）───────────────────────────────────────────────


@dataclass
class MineJob:
    """一次 ``mine`` 子进程任务：按行收集输出，供前端增量拉取。"""

    id: str
    spec_out: str
    #: 逐代历史 sidecar 名（相对 artifacts，由 CLI 按 --out 派生）。
    history_out: str
    args: list[str]
    proc: subprocess.Popen[str]
    lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    returncode: int | None = None

    @property
    def running(self) -> bool:
        return self.returncode is None

    def pump(self) -> None:
        """在后台线程里按行读取子进程输出（UTF-8，容错替换）。"""
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.rstrip("\r\n")
            with self.lock:
                self.lines.append(line)
        self.proc.wait()
        with self.lock:
            self.returncode = self.proc.returncode
            self.lines.append(f"[进程退出] code={self.proc.returncode}")


class _JobManager:
    """同一时刻只允许一个 mine 任务（简单互斥）。"""

    def __init__(self) -> None:
        self._job: MineJob | None = None

    @property
    def job(self) -> MineJob | None:
        return self._job

    def start(
        self,
        *,
        data: str,
        budget: str,
        seed: int,
        symbol: str,
        timeframe: str,
        top_k: int,
        n_folds: int,
        market: str = "",
    ) -> MineJob:
        # 前置校验：把 pipeline 的 ConfigError 提前成 400，而不是让子进程甩堆栈。
        _validate_market_choice(symbol, market)
        if self._job is not None and self._job.running:
            raise HTTPException(status_code=409, detail="已有挖掘任务在运行，请等待完成或刷新页面")
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        spec_out = _ARTIFACTS / f"spec_{stamp}.json"
        args = [
            "mine",
            "--data",
            data,
            "--budget",
            budget,
            "--out",
            str(spec_out),
            "--top-k",
            str(top_k),
            "--n-folds",
            str(n_folds),
        ]
        if seed:
            args += ["--seed", str(seed)]
        if symbol:
            args += ["--symbol", symbol]
        if timeframe:
            args += ["--timeframe", timeframe]
        if market:
            args += ["--market", market]
        proc = subprocess.Popen(  # noqa: S603 - 参数来自本机 UI 表单，白名单拼接
            _cli_process(*args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_REPO_ROOT),
        )
        # 逐代历史 sidecar 路径必须与 CLI 的派生规则一致（<out stem>.history.json）。
        history_out = spec_out.with_suffix(".history.json").name
        job = MineJob(
            id=uuid.uuid4().hex[:8],
            spec_out=spec_out.name,
            history_out=history_out,
            args=args,
            proc=proc,
        )
        threading.Thread(target=job.pump, daemon=True).start()
        self._job = job
        return job


_jobs = _JobManager()


# ── Pydantic 请求体 ──────────────────────────────────────────────────────────


class MineRequest(BaseModel):
    data: str
    budget: str = "quick"
    seed: int = 0
    symbol: str = ""
    timeframe: str = ""
    market: str = ""
    top_k: int = 5
    n_folds: int = 5


class ExportRequest(BaseModel):
    spec: str
    name: str = ""
    gate_deadband: float = 0.0


class VerifyRequest(BaseModel):
    file: str
    data: str = ""


class BacktestRequest(BaseModel):
    spec: str
    data: str
    market: str = ""
    rolling_window: int = 0


class FetchRequest(BaseModel):
    """数据获取请求体（模块级，确保 FastAPI 将其识别为 JSON body）。"""

    symbol: str
    timeframe: str = "H1"
    since: int = 0
    source: str = ""  # 单选数据来源类名（空 = 按优先级自动）
    note: str = ""  # 备注，追加至生成的缓存文件名
    market_profile: str = ""  # 市场画像（前端联动；后端仅透传/校验）


class TuneRequest(BaseModel):
    """参数寻优请求体（模块级，确保 FastAPI 将其识别为 JSON body）。"""

    spec: str
    data: str
    param_spaces: list[dict[str, Any]] = []
    n_trials: int = 50
    metric: str = "sharpe"
    seed: int = 42


# ── App ──────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """构造仪表盘 FastAPI 应用。"""
    app = FastAPI(title="妙算仪表盘", docs_url=None, redoc_url=None)

    _STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(_INDEX_HTML)

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── 数据与产物只读视图 ───────────────────────────────────────────────
    @app.get("/api/data")
    def list_data() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for directory in _DATA_DIRS:
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in _DATA_SUFFIXES:
                    continue
                stat = path.stat()
                # 起止时间：轻量只读 time 列（失败降级为 ""，绝不 500）
                rng = _peek_time_range(path)
                out.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_mb": round(stat.st_size / 1e6, 2),
                        "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                        "start": rng[0] if rng else "",
                        "end": rng[1] if rng else "",
                    }
                )
        return out

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        """全局只读配置：词表冻结锁 + 可选画像 / 已知标的（供页头展示真实值）。"""
        return {
            "vocab_version": VOCAB_VERSION,
            "n_features": FORMULA_VOCAB.feature_count,
            "n_operators": len(FORMULA_VOCAB.operator_names),
            "n_tokens": FORMULA_VOCAB.size,
            # 行情目录经 config.kline_data_dir() 解析（可用 MIAOSUAN_KLINE_DIR 覆盖），
            # 前端据此渲染「文件放哪儿」提示，避免前端再硬编码一份而与环境脱节。
            "data_dirs": [str(d) for d in _DATA_DIRS],
            "profiles": list(EXPECTED_PROFILE_NAMES),
            "profile_details": {
                name: {
                    "symbols": p.symbols,
                    "timeframes": p.timeframes,
                    "data_sources": p.data_sources,
                }
                for name, p in all_profiles().items()
            },
            "known_symbols": sorted(_SYMBOL_PROFILE_FALLBACK),
            "miaosuan_version": MIAOSUAN_VERSION,
        }

    @app.get("/api/inspect")
    def inspect_data(path: str) -> dict[str, Any]:
        """加载行情文件并回报真实元信息（品种 / 周期 / bar 数 / 年限 / 指纹）。"""
        file_path = _resolve_data_file(path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"文件不存在：{file_path}")
        try:
            panel = load(str(file_path))
        except Exception as exc:  # noqa: BLE001 - 数据文件千奇百怪，统一转成 400
            raise HTTPException(status_code=400, detail=f"数据加载失败：{exc}") from exc
        stamps = panel.time
        span = int(stamps[-1]) - int(stamps[0]) if len(stamps) > 1 else 0
        fmt = "%Y-%m-%d"
        return {
            "path": str(file_path),
            "name": file_path.name,
            "symbol": ", ".join(str(s) for s in panel.symbols),
            "timeframe": str(panel.timeframe or ""),
            "n_bars": int(panel.n_bars),
            "years": round(span / (365.25 * 86400), 2),
            "fingerprint": panel.fingerprint,
            "market_profile": panel.market_profile_name,
            # 复用 _fmt_utc：与 /api/data 的 start/end 保证同一 UTC 口径
            # （两处若一个用 gmtime 一个用 localtime，同一文件会差一天）。
            "start": _fmt_utc(int(stamps[0])) if len(stamps) else "",
            "end": _fmt_utc(int(stamps[-1])) if len(stamps) else "",
        }

    @app.get("/api/specs")
    def list_specs() -> list[dict[str, Any]]:
        """列出 artifacts 下所有 StrategySpec JSON（按修改时间倒序）。"""
        out: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return out
        for path in sorted(
            _ARTIFACTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            if path.name in _EXCLUDED_JSON:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not (isinstance(data, dict) and "payload" in data and "spec_version" in data):
                continue
            payload = data.get("payload") or {}
            tokens = payload.get("tokens") or []
            evidence = data.get("evidence") or {}
            provenance = data.get("provenance") or {}
            semantics = data.get("semantics") or {}
            snapshot = provenance.get("config_snapshot") or {}
            out.append(
                {
                    "file": path.name,
                    "spec_id": _spec_id(path),
                    "name": data.get("name", ""),
                    "n_tokens": len(tokens),
                    "formula": readable_formula([int(t) for t in tokens][:12]) if tokens else "",
                    "val_score": evidence.get("val_score"),
                    "deflated_sharpe": evidence.get("deflated_sharpe"),
                    "gate_verdict": evidence.get("gate_verdict", ""),
                    "vocab_version": payload.get("vocab_version", ""),
                    "symbol": str(snapshot.get("symbol") or ""),
                    "timeframe": str(semantics.get("timeframe") or ""),
                    "created_at": str(provenance.get("created_at") or ""),
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                }
            )
        return out

    @app.get("/api/strategies")
    def list_strategies() -> list[dict[str, Any]]:
        """列出已导出的 妙算 策略 .py（按修改时间倒序）。

        只认**带** ``STRATEGY_MAGIC`` 的文件：该常量由导出模板强制写入，是"这
        是导出产物"的准确判据。否则 artifacts 下的临时脚本也会被当成策略列出
        （且每个都带 4 条 lint error——因为 lint 规则要求声明该常量），把真实
        导出淹掉。
        """
        out: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return out
        for path in sorted(_ARTIFACTS.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            magic = _strategy_magic(source)
            if not magic:
                continue
            issues = lint_source(source)
            out.append(
                {
                    "file": path.name,
                    "magic": magic,
                    "lint_errors": sum(1 for i in issues if i.severity.value == "ERROR"),
                    "lint_warnings": sum(1 for i in issues if i.severity.value == "WARNING"),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "mtime": time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                }
            )
        return out

    @app.get("/api/spec/{name}")
    def get_spec(name: str) -> dict[str, Any]:
        path = _safe_name(name, suffixes={".json"})
        try:
            data: dict[str, Any] = json.loads(
                path.read_text(encoding="utf-8")
            )  # 显式标注，避免 Any 泄漏
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"spec 解析失败：{exc}") from exc
        payload = data.get("payload") or {}
        tokens = [int(t) for t in (payload.get("tokens") or [])]
        data["formula_full"] = readable_formula(tokens)
        data["spec_id"] = _spec_id(path)
        data["vocab_summary"] = {
            "size": FORMULA_VOCAB.size,
            "features": FORMULA_VOCAB.feature_count,
            "operators": len(FORMULA_VOCAB.operator_names),
        }
        return data

    @app.get("/api/file/{name}")
    def get_file(name: str) -> FileResponse:
        path = _safe_name(name, suffixes={".py", ".json"})
        return FileResponse(path, filename=name)

    # ── 挖掘任务 ─────────────────────────────────────────────────────────
    @app.post("/api/mine")
    def start_mine(req: MineRequest) -> dict[str, Any]:
        job = _jobs.start(
            data=req.data,
            budget=req.budget,
            seed=req.seed,
            symbol=req.symbol,
            timeframe=req.timeframe,
            market=req.market,
            top_k=req.top_k,
            n_folds=req.n_folds,
        )
        return {"job_id": job.id, "spec_out": job.spec_out, "cmd": " ".join(job.args)}

    @app.get("/api/job")
    def job_status(offset: int = 0) -> dict[str, Any]:
        job = _jobs.job
        if job is None:
            return {
                "running": False,
                "lines": [],
                "total": 0,
                "returncode": None,
                "spec_out": "",
                "history_out": "",
            }
        with job.lock:
            lines = job.lines[offset:]
            total = len(job.lines)
            returncode = job.returncode
        return {
            "running": returncode is None,
            "lines": lines,
            "total": total,
            "returncode": returncode,
            "spec_out": job.spec_out,
            "history_out": job.history_out,
        }

    @app.get("/api/history")
    def read_history(name: str) -> dict[str, Any]:
        """读取一次挖掘的逐代历史 sidecar（供训练曲线画图）。

        Args:
            name: artifacts 下的 sidecar 文件名（如 ``spec_20260912_101530.history.json``）。

        Returns:
            sidecar 原文（``points`` 为空列表表示本次无历史，前端优雅降级）。
        """
        path = _safe_name(name, suffixes={".json"})
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"历史文件解析失败：{exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("points"), list):
            raise HTTPException(status_code=400, detail=f"历史文件格式不正确：{name}")
        return payload

    # ── 导出与校验（同步子进程，秒级返回）────────────────────────────────
    def _run_sync(args: list[str]) -> tuple[int, str]:
        proc = subprocess.run(  # noqa: S603 - 白名单拼接，见上
            _cli_process(*args),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_REPO_ROOT),
            timeout=600,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output.strip()

    @app.post("/api/export")
    def export(req: ExportRequest) -> dict[str, Any]:
        spec_path = _safe_name(req.spec, suffixes={".json"})
        args = ["export", "--spec", str(spec_path), "--out-dir", str(_ARTIFACTS)]
        if req.name:
            args += ["--name", req.name]
        if req.gate_deadband:
            args += ["--gate-deadband", str(req.gate_deadband)]
        code, output = _run_sync(args)
        return {"returncode": code, "output": output}

    @app.post("/api/backtest")
    def run_backtest(req: BacktestRequest) -> dict[str, Any]:
        """跑一次全样本回测（子进程调 CLI，沿用既定架构）。

        结果同时落盘成 ``artifacts/backtest_<ts>.json`` 便于追溯，并直接回传
        给前端画图。

        Args:
            req: 含 spec 文件名、行情路径、可选市场画像与滚动窗口。

        Returns:
            回测结果 dict（``meta`` / ``summary`` / ``series`` / ``trades``）。
        """
        spec_path = _safe_name(req.spec, suffixes={".json"})
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = _ARTIFACTS / f"backtest_{stamp}.json"
        args = [
            "backtest",
            "--spec",
            str(spec_path),
            "--data",
            req.data,
            "--out",
            str(out_path),
        ]
        if req.market:
            args += ["--market", req.market]
        if req.rolling_window:
            args += ["--rolling-window", str(req.rolling_window)]
        code, output = _run_sync(args)
        if code != 0:
            raise HTTPException(status_code=400, detail=f"回测失败（code={code}）：\n{output}")
        if not out_path.is_file():
            raise HTTPException(status_code=500, detail=f"回测未产出结果文件：{out_path.name}")
        payload: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
        payload["output"] = output
        payload["file"] = out_path.name
        # 记录回测所用的 spec 文件名，便于 Spec 管理模块关联
        meta = payload.get("meta") or {}
        meta["spec_file"] = req.spec
        payload["meta"] = meta
        # 同时更新落盘的文件（持久化关联关系）
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    @app.get("/api/backtests")
    def list_backtests() -> list[dict[str, Any]]:
        """列出历史回测结果（按修改时间倒序）。"""
        out: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return out
        for path in sorted(
            _ARTIFACTS.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            meta = payload["meta"] if isinstance(payload.get("meta"), dict) else {}
            summary = payload["summary"] if isinstance(payload.get("summary"), dict) else {}
            out.append(
                {
                    "file": path.name,
                    "mtime": path.stat().st_mtime,
                    "symbol": str(meta.get("symbol", "")),
                    "timeframe": str(meta.get("timeframe", "")),
                    "n_bars": int(meta.get("n_bars", 0) or 0),
                    "sharpe": float(summary.get("sharpe", 0.0) or 0.0),
                    "total_return": float(summary.get("total_return", 0.0) or 0.0),
                }
            )
        return out

    @app.post("/api/verify")
    def verify(req: VerifyRequest) -> dict[str, Any]:
        path = _safe_name(req.file, suffixes={".py"})
        args = ["verify", "--file", str(path)]
        if req.data:
            args += ["--data", req.data]
        code, output = _run_sync(args)
        return {"returncode": code, "output": output}

    # ── 实时页：只读代理（**只暴露 GET**；写接口一律不存在）──────────────────
    @app.get("/api/realtime/status")
    def realtime_status() -> dict[str, Any]:
        """引擎运行状态（只读）。"""
        client = _realtime_client()
        return _outcome_payload(safe_call(client.engine_status), client.base_url)

    @app.get("/api/realtime/price")
    def realtime_price() -> dict[str, Any]:
        """当前报价 bid/ask/spread（只读）。"""
        client = _realtime_client()
        return _outcome_payload(safe_call(client.market_price), client.base_url)

    @app.get("/api/realtime/candles")
    def realtime_candles(
        symbol: str = "XAUUSD", timeframe: str = "H1", count: int = 120
    ) -> dict[str, Any]:
        """K 线序列（只读）；``count`` 由客户端钳制到 ``[1, 2000]``。"""
        client = _realtime_client()
        return _outcome_payload(
            safe_call(client.market_candles, symbol, timeframe, count), client.base_url
        )

    @app.get("/api/realtime/signals")
    def realtime_signals() -> dict[str, Any]:
        """最近信号列表（只读）。"""
        client = _realtime_client()
        return _outcome_payload(safe_call(client.signals), client.base_url)

    @app.get("/api/realtime/latest")
    def realtime_latest(strategy: str = "") -> dict[str, Any]:
        """指定策略的最新信号（只读）；缺 ``strategy`` 返回 400。"""
        if not strategy.strip():
            raise HTTPException(status_code=400, detail="必须指定策略名（strategy）")
        client = _realtime_client()
        return _outcome_payload(safe_call(client.signals_latest, strategy), client.base_url)

    # ── 数据获取模块 ─────────────────────────────────────────────────────
    @app.get("/api/acquisition/sources")
    def acquisition_sources(profile: str = "") -> list[dict[str, Any]]:
        """列出已配置的数据来源；若给定市场画像则仅返回其严格匹配的数据源类型。"""
        from ..data.acquisition import list_sources

        allowed: list[str] | None = None
        if profile:
            try:
                from ..market.profiles import get_profile

                allowed = list(get_profile(profile).data_sources)
            except Exception:
                allowed = None
        # 确保所有数据源都能正确显示 source_type，即使是 TypedNetworkSource
        return list_sources(allowed_types=allowed)

    @app.get("/api/acquisition/cached")
    def acquisition_cached() -> list[dict[str, Any]]:
        """列出本地缓存的数据文件（**含数据起止时间**）。

        时间范围复用 :func:`_peek_time_range`：与 ``/api/data``、``/api/inspect``
        自动同口径（同一 ``_fmt_utc``），并共享进程内缓存（不重复读盘）。
        任一项路径缺失 / 文件不存在 / 读失败只让该项为空，**绝不 500**。
        """
        from ..data.acquisition import list_cached

        rows = list_cached()
        for row in rows:
            raw_path = str(row.get("path") or "")
            rng = _peek_time_range(Path(raw_path)) if raw_path else None
            row["start"] = rng[0] if rng else ""
            row["end"] = rng[1] if rng else ""
        return rows

    @app.post("/api/acquisition/fetch")
    def acquisition_fetch(req: FetchRequest) -> dict[str, Any]:
        """触发数据获取（指定单一来源，可选备注）。"""
        from ..data.acquisition import fetch as acquire

        try:
            path = acquire(
                req.symbol,
                req.timeframe,
                since=req.since or None,
                source=req.source or None,
                note=req.note or None,
            )
            return {
                "ok": True,
                "path": str(path),
                "symbol": req.symbol,
                "timeframe": req.timeframe,
                "source": req.source or None,
                "note": req.note or None,
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"数据获取失败：{exc}") from exc

    # ── 参数寻优模块 ─────────────────────────────────────────────────────
    class TuneStatusResponse(BaseModel):
        running: bool = False
        result_id: str = ""
        progress: int = 0
        total: int = 0
        best_score: float = 0.0
        error: str = ""

    _tune_job: dict[str, Any] = {}

    @app.post("/api/tune/start")
    def tune_start(req: TuneRequest) -> dict[str, Any]:
        """启动参数寻优任务。"""
        import threading

        try:
            spec_path = _safe_name(req.spec, suffixes={".json"})
            data_path = _resolve_data_file(req.data)
        except HTTPException as exc:
            raise exc

        def _run() -> None:
            try:
                from ..ir.codec import read_spec
                from ..adapters.base import ParamSpace
                from ..ir.schema import FactorPayload
                from ..tune import TuneConfig, TuneEngine, load_param_space_from_spec

                spec = read_spec(str(spec_path))
                if not isinstance(spec.payload, FactorPayload):
                    _tune_job["error"] = "spec 载荷不是因子体，无法寻优"
                    _tune_job["running"] = False
                    return
                tokens = tuple(spec.payload.tokens)
                param_spaces = [ParamSpace.from_dict(s) for s in req.param_spaces]
                if not param_spaces:
                    # 前端未显式指定参数空间 → 自动从 spec 挖掘语义旋钮
                    # （neutral_band / roll_window / long_only），缺省值贴合该 spec。
                    auto_spaces = load_param_space_from_spec(str(spec_path))
                    if auto_spaces:
                        param_spaces = auto_spaces
                config = TuneConfig(
                    spec_path=str(spec_path),
                    param_spaces=param_spaces,
                    n_trials=req.n_trials,
                    metric=req.metric,
                    seed=req.seed,
                )
                engine = TuneEngine(config)
                result = engine.optimize(tokens, str(data_path), spec)
                _tune_job["result"] = result.to_dict()
                _tune_job["running"] = False
            except Exception as exc:  # noqa: BLE001
                _tune_job["error"] = f"{type(exc).__name__}: {exc}"
                _tune_job["running"] = False

        _tune_job = {"running": True, "result": None, "error": ""}
        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "message": "寻优任务已启动"}

    @app.get("/api/tune/status")
    def tune_status() -> dict[str, Any]:
        """查询寻优任务状态。"""
        return {
            "running": _tune_job.get("running", False),
            "error": _tune_job.get("error", ""),
        }

    @app.get("/api/tune/result")
    def tune_result() -> dict[str, Any]:
        """获取寻优结果。"""
        result: dict[str, Any] | None = _tune_job.get("result")
        if result is None:
            raise HTTPException(status_code=404, detail="尚未完成寻优任务")
        return result

    @app.get("/api/tune/list")
    def tune_list() -> list[dict[str, Any]]:
        """列出历史寻优结果。"""
        out: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return out
        for path in sorted(
            _ARTIFACTS.glob("tune_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append(
                    {
                        "file": path.name,
                        "result_id": data.get("result_id", ""),
                        "n_trials": data.get("n_trials", 0),
                        "best_score": data.get("best", {}).get("score", 0.0),
                        "elapsed_sec": data.get("elapsed_sec", 0),
                        "mtime": time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
        return out

    # ── Spec 管理模块（谱系 / 映射 / 生命周期）────────────────────────────

    def _extract_py_spec_id(source: str) -> str:
        """从 .py 源码注释中提取 spec_id（`IR spec id：xxxx`）。"""
        for line in source.splitlines():
            if "spec id" in line.lower() or "spec_id" in line.lower():
                # 匹配 "IR spec id：5d0841b6db9fdca2" 格式
                for sep in ("：", ":"):
                    if sep in line:
                        val = line.rsplit(sep, 1)[-1].strip().strip("*").strip()
                        if val and all(c in "0123456789abcdef" for c in val[:16]):
                            return val[:16]
        return ""

    def _extract_py_name(source: str) -> str:
        """从 .py 源码提取 STRATEGY_NAME 常量值。"""
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("STRATEGY_NAME"):
                val = stripped.partition("=")[2].strip()
                return val.strip('"').strip("'")
        return ""

    def _extract_py_deployable(source: str) -> bool:
        """从 .py 源码提取 DEPLOYABLE 常量值。"""
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("DEPLOYABLE"):
                val = stripped.partition("=")[2].strip()
                return val in ("True", "true", "1")
        return False

    def _scan_all_specs() -> list[dict[str, Any]]:
        """扫描 artifacts/ 下所有 spec JSON，返回含 spec_id 的完整列表。"""
        result: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return result
        for path in sorted(
            _ARTIFACTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            if path.name in _EXCLUDED_JSON:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not (isinstance(data, dict) and "payload" in data and "spec_version" in data):
                continue
            payload = data.get("payload") or {}
            tokens = payload.get("tokens") or []
            evidence = data.get("evidence") or {}
            provenance = data.get("provenance") or {}
            semantics = data.get("semantics") or {}
            snapshot = provenance.get("config_snapshot") or {}
            # 使用内容寻址 spec_id（与 StrategySpec.spec_id 一致），
            # 而非文件哈希——这样才能与 .py 内嵌的 spec_id 匹配。
            core = {
                "payload": payload,
                "semantics": semantics,
                "spec_version": data.get("spec_version", "1.0"),
            }
            import hashlib as _hl
            spec_id = _hl.sha256(
                json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            result.append({
                "file": path.name,
                "spec_id": spec_id,
                "name": data.get("name", ""),
                "n_tokens": len(tokens),
                "formula": readable_formula([int(t) for t in tokens][:12]) if tokens else "",
                "val_score": evidence.get("val_score"),
                "deflated_sharpe": evidence.get("deflated_sharpe"),
                "gate_verdict": evidence.get("gate_verdict", ""),
                "gate_reasons": evidence.get("gate_reasons", []),
                "vocab_version": payload.get("vocab_version", ""),
                "symbol": str(snapshot.get("symbol") or ""),
                "timeframe": str(semantics.get("timeframe") or ""),
                "created_at": str(provenance.get("created_at") or ""),
                "market": str(provenance.get("market") or ""),
                "budget": str(provenance.get("budget") or ""),
                "git_sha": str(provenance.get("git_sha") or ""),
                "seed": provenance.get("seed"),
                "data_fingerprint": str(provenance.get("data_fingerprint") or ""),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                "payload": payload,
                "semantics": semantics,
                "evidence": evidence,
                "provenance": provenance,
                "notes": data.get("notes", ""),
            })
        return result

    def _scan_all_strategies() -> list[dict[str, Any]]:
        """扫描 artifacts/ + shenji-strategies/ 下所有 .py 策略文件。"""
        result: list[dict[str, Any]] = []
        search_dirs = [_ARTIFACTS, _REPO_ROOT / "shenji-strategies"]
        seen: set[Path] = set()
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for path in sorted(
                directory.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    source = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                magic = _strategy_magic(source)
                if not magic:
                    continue
                issues = lint_source(source)
                spec_id = _extract_py_spec_id(source)
                str_name = _extract_py_name(source)
                deployable = _extract_py_deployable(source)
                # 相对路径
                try:
                    rel = str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
                except ValueError:
                    rel = path.name
                result.append({
                    "file": path.name,
                    "rel_path": rel,
                    "magic": magic,
                    "spec_id": spec_id,
                    "strategy_name": str_name,
                    "deployable": deployable,
                    "lint_errors": sum(1 for i in issues if i.severity.value == "ERROR"),
                    "lint_warnings": sum(1 for i in issues if i.severity.value == "WARNING"),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                    "dir": "shenji-strategies" if "shenji-strategies" in str(path) else "artifacts",
                })
        return result

    def _scan_all_backtests() -> list[dict[str, Any]]:
        """扫描 artifacts/ 下所有回测结果。"""
        result: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return result
        for path in sorted(
            _ARTIFACTS.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
            summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
            # 尝试从 meta.spec_file 反查 spec_id
            spec_file = str(meta.get("spec_file") or meta.get("spec") or "")
            result.append({
                "file": path.name,
                "spec_file": spec_file,
                "symbol": str(meta.get("symbol", "")),
                "timeframe": str(meta.get("timeframe", "")),
                "n_bars": int(meta.get("n_bars", 0) or 0),
                "sharpe": float(summary.get("sharpe", 0.0) or 0.0),
                "total_return": float(summary.get("total_return", 0.0) or 0.0),
                "max_drawdown": float(summary.get("max_drawdown", 0.0) or 0.0),
                "sortino": float(summary.get("sortino", 0.0) or 0.0),
                "n_trades": int(summary.get("n_trades", 0) or 0),
                "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
            })
        return result

    def _scan_all_tunes() -> list[dict[str, Any]]:
        """扫描 artifacts/tune/ 下所有寻优结果。"""
        result: list[dict[str, Any]] = []
        tune_dir = _ARTIFACTS / "tune"
        if not tune_dir.is_dir():
            return result
        for path in sorted(
            tune_dir.glob("tune_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            config = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
            result.append({
                "file": path.name,
                "result_id": data.get("result_id", ""),
                "spec_path": str(config.get("spec_path") or ""),
                "n_trials": data.get("n_trials", 0),
                "best_score": float(data.get("best", {}).get("score", 0.0) or 0.0),
                "elapsed_sec": float(data.get("elapsed_sec", 0) or 0),
                "best_params": dict(data.get("best", {}).get("params", {})),
                "metric": str(config.get("metric") or ""),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
            })
        return result

    @app.get("/api/specs/detail")
    def specs_detail() -> dict[str, Any]:
        """Spec 管理总览：所有 spec + 所有策略 + 所有回测 + 所有寻优 + 映射关系。"""
        specs = _scan_all_specs()
        strategies = _scan_all_strategies()
        backtests = _scan_all_backtests()
        tunes = _scan_all_tunes()

        # 构建映射：spec_id → 关联的策略 / 回测 / 寻优
        for spec in specs:
            sid = spec["spec_id"]
            spec_file = spec["file"]
            # 关联导出的 .py
            spec["exports"] = [
                s for s in strategies
                if s.get("spec_id") == sid or _spec_matches_file(s.get("spec_id", ""), sid, spec_file)
            ]
            # 关联回测
            spec["backtests"] = [
                bt for bt in backtests
                if bt.get("spec_file") == spec_file
            ]
            # 关联寻优
            spec["tunes"] = [
                tu for tu in tunes
                if spec_file in tu.get("spec_path", "")
            ]
            # 谱系信息（从 provenance 提取，当前为空，预留）
            prov = spec.get("provenance") or {}
            spec["parent_spec_id"] = prov.get("parent_spec_id")
            spec["lineage"] = prov.get("lineage", [])
            spec["derivation"] = prov.get("derivation", "mine")

        # 统计摘要
        summary = {
            "total_specs": len(specs),
            "total_strategies": len(strategies),
            "total_backtests": len(backtests),
            "total_tunes": len(tunes),
            "deployable_count": sum(1 for s in specs if s.get("gate_verdict") == "DEPLOYABLE"),
            "blocked_count": sum(1 for s in specs if s.get("gate_verdict") == "BLOCKED"),
            "research_only_count": sum(1 for s in specs if s.get("gate_verdict") == "RESEARCH_ONLY"),
            "exported_count": sum(1 for s in specs if s.get("exports")),
        }

        return {
            "specs": specs,
            "strategies": strategies,
            "backtests": backtests,
            "tunes": tunes,
            "summary": summary,
        }

    def _spec_matches_file(py_spec_id: str, registry_spec_id: str, spec_file: str) -> bool:
        """判断 .py 内嵌的 spec_id 是否匹配某个 spec 文件。"""
        if py_spec_id and registry_spec_id:
            # 如果 .py 有 spec_id，比较前 12 位（_spec_id 返回 12 位）
            return py_spec_id[:12] == registry_spec_id[:12]
        return False

    @app.get("/api/specs/{spec_file}/relations")
    def spec_relations(spec_file: str) -> dict[str, Any]:
        """单个 spec 的完整关系视图：关联的导出 / 回测 / 寻优 / 谱系。"""
        path = _safe_name(spec_file, suffixes={".json"})
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"spec 解析失败：{exc}") from exc

        spec_id = _spec_id(path)
        all_strategies = _scan_all_strategies()
        all_backtests = _scan_all_backtests()
        all_tunes = _scan_all_tunes()

        exports = [
            s for s in all_strategies
            if s.get("spec_id")[:12] == spec_id[:12]
        ]
        backtests = [
            bt for bt in all_backtests
            if bt.get("spec_file") == spec_file
        ]
        tunes = [
            tu for tu in all_tunes
            if spec_file in tu.get("spec_path", "")
        ]

        # 谱系（当前 provenance 没有 lineage 字段，预留）
        prov = data.get("provenance") or {}
        parent_spec_id = prov.get("parent_spec_id")
        lineage = prov.get("lineage", [])
        derivation = prov.get("derivation", "mine")

        return {
            "spec_file": spec_file,
            "spec_id": spec_id,
            "name": data.get("name", ""),
            "exports": exports,
            "backtests": backtests,
            "tunes": tunes,
            "parent_spec_id": parent_spec_id,
            "lineage": lineage,
            "derivation": derivation,
        }

    @app.get("/api/specs/{name}")
    def get_spec_plural(name: str) -> dict[str, Any]:
        """向后兼容：复数形式 /api/specs/{name} 重定向到单数形式逻辑。"""
        return get_spec(name)

    return app
