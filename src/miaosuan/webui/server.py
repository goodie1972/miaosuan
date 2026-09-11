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
from pydantic import BaseModel

from ..adapters.algoforge.generator import readable_formula
from ..core.vocab import FORMULA_VOCAB
from ..market.profiles import EXPECTED_PROFILE_NAMES
from ..pipeline import _SYMBOL_PROFILE_FALLBACK

__all__ = ["create_app"]

#: 仓库根（``src/miaosuan/webui/server.py`` 往上 3 层）。
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
#: 产物目录。
_ARTIFACTS: Path = _REPO_ROOT / "artifacts"
#: 静态页面。
_INDEX_HTML: Path = Path(__file__).resolve().parent / "static" / "index.html"
#: 默认行情数据目录（用户在本机放置 TradingView 拉取的 parquet/csv）。
_DATA_DIRS: tuple[Path, ...] = (
    Path(r"D:\K线数据"),
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
    """构造 CLI 子进程命令（用当前解释器，保证与 venv 一致）。"""
    return [sys.executable, "-m", "miaosuan.cli", *args]


def _spec_id(path: Path) -> str:
    """从文件内容派生稳定短 id（spec JSON 本身不携带 spec_id 键）。"""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


#: 可选市场画像提示（唯一来源 :data:`EXPECTED_PROFILE_NAMES`，此处不复制字符串）。
_PROFILE_HINT: str = " / ".join(EXPECTED_PROFILE_NAMES)
#: 已知标的提示（唯一来源 ``pipeline._SYMBOL_PROFILE_FALLBACK``，此处不复制字符串）。
_SYMBOL_HINT: str = " / ".join(sorted(_SYMBOL_PROFILE_FALLBACK))


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


# ── 挖掘任务（单飞 + 增量日志）───────────────────────────────────────────────

@dataclass
class MineJob:
    """一次 ``mine`` 子进程任务：按行收集输出，供前端增量拉取。"""

    id: str
    spec_out: str
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

    def start(self, *, data: str, budget: str, seed: int, symbol: str,
              timeframe: str, top_k: int, n_folds: int,
              market: str = "") -> MineJob:
        # 前置校验：把 pipeline 的 ConfigError 提前成 400，而不是让子进程甩堆栈。
        _validate_market_choice(symbol, market)
        if self._job is not None and self._job.running:
            raise HTTPException(status_code=409, detail="已有挖掘任务在运行，请等待完成或刷新页面")
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        spec_out = _ARTIFACTS / f"spec_{stamp}.json"
        args = ["mine", "--data", data, "--budget", budget,
                "--out", str(spec_out), "--top-k", str(top_k), "--n-folds", str(n_folds)]
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
        job = MineJob(id=uuid.uuid4().hex[:8], spec_out=spec_out.name, args=args, proc=proc)
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


# ── App ──────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """构造仪表盘 FastAPI 应用。"""
    app = FastAPI(title="妙算仪表盘", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(_INDEX_HTML)

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
                out.append({
                    "path": str(path),
                    "name": path.name,
                    "size_mb": round(path.stat().st_size / 1e6, 2),
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                })
        return out

    @app.get("/api/specs")
    def list_specs() -> list[dict[str, Any]]:
        """列出 artifacts 下所有 StrategySpec JSON（按修改时间倒序）。"""
        out: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return out
        for path in sorted(_ARTIFACTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if path.name in _EXCLUDED_JSON:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not (isinstance(data, dict) and "payload" in data
                    and "spec_version" in data):
                continue
            payload = data.get("payload") or {}
            tokens = payload.get("tokens") or []
            evidence = data.get("evidence") or {}
            out.append({
                "file": path.name,
                "spec_id": _spec_id(path),
                "name": data.get("name", ""),
                "n_tokens": len(tokens),
                "formula": readable_formula([int(t) for t in tokens][:12]) if tokens else "",
                "val_score": evidence.get("val_score"),
                "deflated_sharpe": evidence.get("deflated_sharpe"),
                "gate_verdict": evidence.get("gate_verdict", ""),
                "vocab_version": payload.get("vocab_version", ""),
            })
        return out

    @app.get("/api/strategies")
    def list_strategies() -> list[dict[str, Any]]:
        """列出已导出的 AlgoForge 策略 .py（按修改时间倒序）。"""
        out: list[dict[str, Any]] = []
        if not _ARTIFACTS.is_dir():
            return out
        for path in sorted(_ARTIFACTS.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True):
            text_head = path.read_text(encoding="utf-8", errors="replace")[:4000]
            magic = ""
            for line in text_head.splitlines():
                if line.startswith("STRATEGY_MAGIC"):
                    magic = line.partition("=")[2].strip().strip('"')
                    break
            out.append({
                "file": path.name,
                "magic": magic,
                "size_kb": round(path.stat().st_size / 1024, 1),
                "mtime": time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
            })
        return out

    @app.get("/api/spec/{name}")
    def get_spec(name: str) -> dict[str, Any]:
        path = _safe_name(name, suffixes={".json"})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
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
            data=req.data, budget=req.budget, seed=req.seed,
            symbol=req.symbol, timeframe=req.timeframe, market=req.market,
            top_k=req.top_k, n_folds=req.n_folds,
        )
        return {"job_id": job.id, "spec_out": job.spec_out, "cmd": " ".join(job.args)}

    @app.get("/api/job")
    def job_status(offset: int = 0) -> dict[str, Any]:
        job = _jobs.job
        if job is None:
            return {"running": False, "lines": [], "total": 0, "returncode": None, "spec_out": ""}
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
        }

    # ── 导出与校验（同步子进程，秒级返回）────────────────────────────────
    def _run_sync(args: list[str]) -> tuple[int, str]:
        proc = subprocess.run(  # noqa: S603 - 白名单拼接，见上
            _cli_process(*args), capture_output=True, text=True,
            encoding="utf-8", errors="replace", cwd=str(_REPO_ROOT), timeout=600,
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

    @app.post("/api/verify")
    def verify(req: VerifyRequest) -> dict[str, Any]:
        path = _safe_name(req.file, suffixes={".py"})
        args = ["verify", "--file", str(path)]
        if req.data:
            args += ["--data", req.data]
        code, output = _run_sync(args)
        return {"returncode": code, "output": output}

    return app
