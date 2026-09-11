"""妙算命令行入口（M14）—— **唯一**允许做 IO / 读环境变量的层。

四个子命令构成「模式 A」一条命令闭环：

==========  ================================================================
``mine``    数据 → 切分（封印 hold-out）→ 搜索 → 门禁 → ``StrategySpec`` JSON
``export``  ``StrategySpec`` JSON → AlgoForge 策略 ``.py``（含 lint）
``verify``  对导出文件做 repaint lint +（可选）数值保真度回归
``report``  打印 spec 的证据/溯源摘要
==========  ================================================================

架构约束（依赖方向铁律）：

* :meth:`AppConfig.from_env` **只在**本模块被调用（CLI 边界）；
* ``core`` / ``search`` / ``gate`` / ``adapters`` 都不做 IO；
* 配置快照在这里生成，经 :func:`~miaosuan.pipeline.run_mine` 写进 provenance。

用法示例::

    miaosuan mine --data data/xauusd_h1.parquet --budget standard --out artifacts/spec.json
    miaosuan export --spec artifacts/spec.json --out-dir algoforge-strategies
    miaosuan verify --file algoforge-strategies/20260910_xauusd_v1.py --data data/xauusd_h1.parquet
    miaosuan report --spec artifacts/spec.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer

from .adapters.algoforge import AlgoforgePort
from .adapters.algoforge.lint import lint_source
from .adapters.base import install_stub_modules, uninstall_stub_modules
from .config import AppConfig
from .core.features import compute_features
from .core.vm import StackVM
from .data.loader import load
from .ir.codec import read_spec, write_spec
from .ir.provenance import resolve_git_sha
from .ir.schema import FactorPayload, StrategySpec
from .pipeline import run_mine

__all__ = ["app", "export", "mine", "report", "verify"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="妙算（MiaoSuan）—— 声明式因子挖掘 + AlgoForge 策略导出",
)

#: 合法预算档位。
_BUDGETS: tuple[str, ...] = ("quick", "standard", "deep")


def _echo(text: str = "") -> None:
    """统一输出（便于测试替换 stdout）。"""
    typer.echo(text)


def _die(message: str) -> None:
    """打印错误并以退出码 1 结束。"""
    typer.echo(f"错误：{message}", err=True)
    raise typer.Exit(code=1)


# ── mine ────────────────────────────────────────────────────────────────────
@app.command()
def mine(
    data: str = typer.Option(..., "--data", help="行情数据文件（parquet / csv）"),
    budget: str = typer.Option("standard", "--budget", help="预算档位 quick/standard/deep"),
    market: str = typer.Option("", "--market", help="市场画像名（缺省按数据自带 profile）"),
    seed: int = typer.Option(0, "--seed", help="随机种子（0 = 使用配置默认值）"),
    symbol: str = typer.Option("", "--symbol", help="标的代码（覆盖环境变量）"),
    timeframe: str = typer.Option("", "--timeframe", help="周期（覆盖环境变量）"),
    top_k: int = typer.Option(5, "--top-k", help="汇总候选数"),
    n_folds: int = typer.Option(5, "--n-folds", help="Walk-Forward 折数"),
    out: str = typer.Option("artifacts/spec.json", "--out", help="StrategySpec 输出路径"),
) -> None:
    """挖掘因子：数据 → 切分（封印 hold-out）→ 搜索 → 门禁 → Spec JSON。"""
    if budget not in _BUDGETS:
        _die(f"未知预算档位 {budget!r}，可选 {list(_BUDGETS)}")

    config = AppConfig.from_env()
    overrides: dict[str, Any] = {}
    if symbol:
        overrides["symbol"] = symbol
    if timeframe:
        overrides["timeframe"] = timeframe
    if seed:
        overrides["seed"] = int(seed)
    if overrides:
        config = config.with_overrides(**overrides)

    try:
        panel = load(data)
    except (FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法加载行情数据 {data!r}：{exc}")
    _echo(f"数据：{data}  N={panel.n_symbols} T={panel.n_bars} profile={panel.market_profile_name}")
    _echo(f"指纹：{panel.fingerprint}")

    git_sha = resolve_git_sha()
    outcome = run_mine(
        panel,
        config=config,
        budget_profile=budget,
        top_k=top_k,
        n_folds=n_folds,
        market=market,
        git_sha=git_sha,
    )
    result = outcome.result
    spec = outcome.spec

    write_spec(out, spec)
    _echo(f"市场画像：{outcome.profile.name}  预算档位={budget}")
    _echo("")
    _echo(f"停止原因：{result.stop_reason}  代数={result.generations}  评估={result.n_evaluations}")
    _echo(
        f"多样性：最低 {result.min_diversity:.3f} / 初始 {result.initial_diversity:.3f} "
        f"= {result.diversity_ratio:.3f}"
    )
    _echo(f"开发区 {result.dev_bars} / 全量 {result.n_bars} 根（hold-out 未触碰={result.holdout_untouched}）")
    _echo("")
    for i, cand in enumerate(result.candidates):
        _echo(
            f"[{i}] {cand.decoded}  val={cand.val_score:.4f}  dsr={cand.dsr:.4f}  "
            f"verdict={cand.verdict}"
        )
    _echo("")
    _echo(f"spec 已写入：{out}")
    payload_vocab = (
        spec.payload.vocab_version if isinstance(spec.payload, FactorPayload) else "-"
    )
    _echo(f"spec_id={spec.spec_id}  vocab={payload_vocab}  git={git_sha}")


# ── export ──────────────────────────────────────────────────────────────────
@app.command()
def export(
    spec_path: str = typer.Option(..., "--spec", help="StrategySpec JSON 路径"),
    out_dir: str = typer.Option("artifacts", "--out-dir", help="输出目录"),
    name: str = typer.Option("", "--name", help="覆盖策略名（影响文件名与 STRATEGY_NAME）"),
    gate_deadband: float = typer.Option(
        0.0, "--gate-deadband", help="GATE 死区（R1 保护；0 = 关闭，与回测逐位一致）"
    ),
    ledger: str = typer.Option("", "--ledger", help="magic 账本路径（缺省用默认路径）"),
    date: str = typer.Option("", "--date", help="文件名日期 YYYYMMDD（缺省取当天 UTC）"),
) -> None:
    """导出 AlgoForge 策略 .py（自动分配 magic + 静态检查）。"""
    try:
        spec = read_spec(spec_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法读取 spec 文件 {spec_path!r}：{exc}")
    if name:
        spec = StrategySpec(
            spec_version=spec.spec_version,
            name=name,
            payload=spec.payload,
            semantics=spec.semantics,
            evidence=spec.evidence,
            provenance=spec.provenance,
            notes=spec.notes,
        )
    port = AlgoforgePort(
        ledger_path=ledger or None,
        date=date,
        gate_deadband=float(gate_deadband),
    )
    result = port.compile(spec)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / result.filename
    target.write_text(result.source, encoding="utf-8")

    _echo(f"已导出：{target}")
    _echo(f"magic={result.magic}  name={spec.name}  platform={result.platform}")
    _echo(f"门禁={spec.evidence.gate_verdict}  DEPLOYABLE={spec.evidence.deployable}")
    for param in result.param_space:
        _echo(f"  param {param.name} = {param.default}  [{param.kind}]")
    if result.issues:
        _echo("")
        for issue in result.issues:
            line = "" if issue.line is None else f":{issue.line}"
            _echo(f"  [{issue.severity.value}] {issue.code}{line} {issue.message}")
    _echo("")
    _echo(f"静态检查：{len(result.errors)} ERROR / {len(result.warnings)} WARNING")
    if not result.ok:
        _die("导出文件未通过静态检查")


# ── verify ──────────────────────────────────────────────────────────────────
@app.command()
def verify(
    file: str = typer.Option(..., "--file", help="待检查的导出 .py 文件"),
    data: str = typer.Option("", "--data", help="行情数据（提供则额外做数值保真度回归）"),
) -> None:
    """对导出文件做 repaint lint（+ 可选数值保真度回归）。"""
    try:
        source = Path(file).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        _die(f"无法读取待检查文件 {file!r}：{exc}")
    issues = lint_source(source)
    errors = [i for i in issues if i.severity.value == "ERROR"]
    _echo(f"文件：{file}")
    _echo(f"静态检查：{len(errors)} ERROR / {len(issues) - len(errors)} WARNING")
    for issue in issues:
        line = "" if issue.line is None else f":{issue.line}"
        _echo(f"  [{issue.severity.value}] {issue.code}{line} {issue.message}")
    if errors:
        _die(f"存在 {len(errors)} 个 ERROR 级问题（含 repaint）")

    if not data:
        _echo("未指定 --data，跳过数值保真度回归")
        return

    try:
        panel = load(data)
    except (FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法加载行情数据 {data!r}：{exc}")
    raw = panel.to_raw_dict()
    tokens = _extract_tokens(source)
    if not tokens:
        _echo("无法从文件中解析 _TOKENS，跳过数值保真度回归")
        return
    max_err = _fidelity_error(file, tokens, raw)
    _echo(f"数值保真度：max_abs_err = {max_err:.3e}（阈值 1e-3）")
    if max_err >= 1e-3:
        _die(f"数值保真度不达标：{max_err:.3e} >= 1e-3")


def _extract_tokens(source: str) -> tuple[int, ...]:
    """从导出文件里取出 ``_TOKENS``（不 import 该文件，纯文本解析）。"""
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("_TOKENS"):
            continue
        _, _, rest = stripped.partition("=")
        body = rest.strip().strip("()")
        if not body:
            return ()
        return tuple(int(part) for part in body.split(",") if part.strip())
    return ()


def _fidelity_error(path: str, tokens: tuple[int, ...], raw: dict[str, Any]) -> float:
    """导入导出文件，对比其因子与原生 :class:`StackVM` 的最大绝对误差。"""
    import importlib.util

    stubs = install_stub_modules(AlgoforgePort.platform)
    try:
        spec = importlib.util.spec_from_file_location("_miaosuan_verify_target", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法构造 module spec：{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        features = compute_features(raw)
        native = StackVM().execute(list(tokens), features)
        if native is None:
            raise RuntimeError("原生 StackVM 返回 None，无法比对")

        flat = {k: np.asarray(v)[0] for k, v in raw.items()}
        got = module.compute_factor(
            flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"]
        )
        return float(np.max(np.abs(np.asarray(got) - np.asarray(native)[0])))
    finally:
        uninstall_stub_modules(stubs)


# ── report ──────────────────────────────────────────────────────────────────
@app.command()
def report(
    spec_path: str = typer.Option(..., "--spec", help="StrategySpec JSON 路径"),
) -> None:
    """打印 spec 的证据与溯源摘要。"""
    spec = read_spec(spec_path)
    evidence = spec.evidence
    provenance = spec.provenance
    _echo(f"策略名    ：{spec.name}")
    _echo(f"spec_id   ：{spec.spec_id}（IR v{spec.spec_version}）")
    _echo(f"payload   ：{type(spec.payload).__name__}")
    if isinstance(spec.payload, FactorPayload):
        _echo(f"tokens    ：{list(spec.payload.tokens)}")
        _echo(f"词表版本  ：{spec.payload.vocab_version}")
    _echo(f"语义      ：{spec.semantics.position_fn} / 中性带 {spec.semantics.neutral_band} "
          f"/ 多空 {spec.semantics.long_short} / 预热 {spec.semantics.warmup_bars}")
    _echo("─" * 60)
    _echo(f"验证集得分：{evidence.val_score}（{evidence.wf_folds} 折 WF）")
    _echo(f"试验次数  ：{evidence.n_trials}")
    _echo(f"DSR       ：{evidence.deflated_sharpe}")
    _echo(f"hold-out  ：{evidence.holdout_sharpe if evidence.holdout_sharpe is not None else '未消费'}")
    _echo(f"成本敏感度：{evidence.cost_sensitivity or 'n/a'}")
    _echo(f"门禁结论  ：{evidence.gate_verdict}（deployable={evidence.deployable}）")
    for reason in evidence.gate_reasons:
        _echo(f"            - {reason}")
    _echo("─" * 60)
    _echo(f"git sha   ：{provenance.git_sha}")
    _echo(f"数据指纹  ：{provenance.data_fingerprint}")
    _echo(f"种子      ：{provenance.seed}")
    _echo(f"市场/预算 ：{provenance.market} / {provenance.budget}")
    _echo(f"生成时刻  ：{provenance.created_at}")


def main() -> int:
    """供 ``python -m miaosuan.cli`` 使用；返回进程退出码。"""
    try:
        app()
    except SystemExit as exc:  # typer/click 的退出信号
        return int(exc.code) if isinstance(exc.code, int) else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
