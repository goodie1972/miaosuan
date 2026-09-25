# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""妙算命令行入口（M14）—— **唯一**允许做 IO / 读环境变量的层。

四个子命令构成「模式 A」一条命令闭环：

==========  ================================================================
``mine``    数据 → 切分（封印 hold-out）→ 搜索 → 门禁 → ``StrategySpec`` JSON
``export``  ``StrategySpec`` JSON → 妙算 策略 ``.py``（含 lint）
``verify``  对导出文件做 repaint lint +（可选）数值保真度回归
``report``  打印 spec 的证据/溯源摘要
==========  ================================================================

架构约束（依赖方向铁律）：

* :meth:`AppConfig.from_env` **只在**本模块被调用（CLI 边界）；
* ``core`` / ``search`` / ``gate`` / ``adapters`` 都不做 IO；
* 配置快照在这里生成，经 :func:`~miaosuan.pipeline.run_mine` 写进 provenance。

用法示例::

    miaosuan mine --data data/xauusd_h1.parquet --budget standard --out artifacts/spec.json
    miaosuan export --spec artifacts/spec.json --out-dir shenji-strategies
    miaosuan verify --file shenji-strategies/20260910_xauusd_v1.py --data data/xauusd_h1.parquet
    miaosuan report --spec artifacts/spec.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer

from .adapters.shenji import ShenjiPort
from .adapters.shenji.lint import lint_source
from .adapters.base import install_stub_modules, uninstall_stub_modules
from .config import AppConfig
from .core.features import compute_features
from .core.signal import MIN_TRADE_EXPOSURE
from .core.vm import StackVM
from .data.loader import load
from .errors import MiaoSuanError
from .ir.codec import read_spec, write_spec
from .ir.provenance import resolve_git_sha
from .ir.schema import FactorPayload, StrategySpec
from .market.profiles import get_profile
from .pipeline import run_mine
from .report.equity import run_full_backtest
from .tune import TuneConfig, TuneEngine, load_param_space_from_spec, save_tune_result

__all__ = ["app", "backtest", "export", "mine", "report", "tune", "ui", "verify"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="妙算（MiaoSuan）—— 声明式因子挖掘 + 妙算 策略导出",
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


#: 代际历史 sidecar 的版本号（结构变更时递增，便于前端识别）。
_HISTORY_VERSION = 1


def _history_payload(result: Any, *, spec_out: str, budget: str) -> dict[str, Any]:
    """把逐代统计快照整理成可机器解析的 dict。

    逐代历史只写 sidecar 文件而**不打进 stdout**：stdout 已经承载给人看的
    挖掘结论（``[i] 候选`` 那几行），塞进几十行 ``GEN n ...`` 会把真正要人
    看的结果淹掉，且现有测试对 stdout 有断言。sidecar 是纯新增、机器读、
    不污染可读性。

    Args:
        result: :class:`~miaosuan.search.mine.MineResult`（结构化访问，
            不依赖具体类型，测试可传替身）。
        spec_out: 本次 ``--out`` 的 spec 路径，写进 sidecar 便于追溯。
        budget: 预算档位。

    Returns:
        ``{"version", "spec", "budget", "stop_reason", "generations",
        "n_evaluations", "points"}``；``points`` 每项为
        ``{"generation", "best", "mean", "diversity", "n_evaluations"}``。
        历史为空时 ``points`` 为空列表（前端据此优雅降级，不画空图）。
    """
    points: list[dict[str, Any]] = []
    for snap in getattr(result, "history", None) or ():
        points.append(
            {
                "generation": int(getattr(snap, "generation", 0)),
                "best": float(getattr(snap, "best_fitness", 0.0)),
                "mean": float(getattr(snap, "mean_fitness", 0.0)),
                "diversity": float(getattr(snap, "diversity", 0.0)),
                "n_evaluations": int(getattr(snap, "n_evaluations", 0)),
            }
        )
    return {
        "version": _HISTORY_VERSION,
        "spec": str(spec_out),
        "budget": budget,
        "stop_reason": str(getattr(result, "stop_reason", "")),
        "generations": int(getattr(result, "generations", 0)),
        "n_evaluations": int(getattr(result, "n_evaluations", 0)),
        "points": points,
    }


def _write_history(path: Path, result: Any, *, spec_out: str, budget: str) -> int:
    """写代际历史 sidecar（JSON），返回写入的代数。

    Args:
        path: 目标文件路径；父目录不存在时自动创建。
        result: 同 :func:`_history_payload`。
        spec_out: 同 :func:`_history_payload`。
        budget: 同 :func:`_history_payload`。

    Returns:
        ``points`` 的条数（即真实代数）；为 0 表示本次没有可画的历史。
    """
    payload = _history_payload(result, spec_out=spec_out, budget=budget)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(payload["points"])


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
    history_out: str = typer.Option(
        "",
        "--history-out",
        help="逐代历史 sidecar 路径（缺省由 --out 派生：<stem>.history.json）",
    ),
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
    except (MiaoSuanError, FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法加载行情数据 {data!r}：{exc}")
    _echo(f"数据：{data}  N={panel.n_symbols} T={panel.n_bars} profile={panel.market_profile_name}")
    _echo(f"指纹：{panel.fingerprint}")

    git_sha = resolve_git_sha()
    try:
        outcome = run_mine(
            panel,
            config=config,
            budget_profile=budget,
            top_k=top_k,
            n_folds=n_folds,
            market=market,
            git_sha=git_sha,
        )
    except MiaoSuanError as exc:
        _die(f"挖掘失败：{exc}")
    result = outcome.result
    spec = outcome.spec

    write_spec(out, spec)
    history_path = Path(history_out) if history_out else Path(out).with_suffix(".history.json")
    n_points = _write_history(history_path, result, spec_out=out, budget=budget)
    _echo(f"市场画像：{outcome.profile.name}  预算档位={budget}")
    _echo("")
    _echo(f"停止原因：{result.stop_reason}  代数={result.generations}  评估={result.n_evaluations}")
    _echo(
        f"多样性：最低 {result.min_diversity:.3f} / 初始 {result.initial_diversity:.3f} "
        f"= {result.diversity_ratio:.3f}"
    )
    _echo(
        f"开发区 {result.dev_bars} / 全量 {result.n_bars} 根（hold-out 未触碰={result.holdout_untouched}）"
    )
    _echo("")
    for i, cand in enumerate(result.candidates):
        _echo(
            f"[{i}] {cand.decoded}  val={cand.val_score:.4f}  dsr={cand.dsr:.4f}  "
            f"verdict={cand.verdict}"
        )
    _echo("")
    _echo(f"spec 已写入：{out}")
    if n_points:
        _echo(f"逐代历史：{n_points} 代 → {history_path}")
    else:
        _echo("逐代历史：本次未产出（0 代）—— 训练曲线不显示")
    payload_vocab = spec.payload.vocab_version if isinstance(spec.payload, FactorPayload) else "-"
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
    """导出 妙算 策略 .py（自动分配 magic + 静态检查）。"""
    try:
        spec = read_spec(spec_path)
    except (MiaoSuanError, FileNotFoundError, OSError, ValueError) as exc:
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
    port = ShenjiPort(
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
    except (MiaoSuanError, FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法加载行情数据 {data!r}：{exc}")
    raw = panel.to_raw_dict()
    tokens = _extract_tokens(source)
    if not tokens:
        _echo("无法从文件中解析 _TOKENS，跳过数值保真度回归")
        return
    try:
        max_err = _fidelity_error(file, tokens, raw)
    except (MiaoSuanError, RuntimeError, TypeError, KeyError, IndexError, ValueError) as exc:
        _die(f"数值保真度回归执行失败（{file!r}）：{type(exc).__name__}: {exc}")
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

    stubs = install_stub_modules(ShenjiPort.platform)
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
    try:
        spec = read_spec(spec_path)
    except (MiaoSuanError, FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法读取 spec 文件 {spec_path!r}：{exc}")
    evidence = spec.evidence
    provenance = spec.provenance
    _echo(f"策略名    ：{spec.name}")
    _echo(f"spec_id   ：{spec.spec_id}（IR v{spec.spec_version}）")
    _echo(f"payload   ：{type(spec.payload).__name__}")
    if isinstance(spec.payload, FactorPayload):
        _echo(f"tokens    ：{list(spec.payload.tokens)}")
        _echo(f"词表版本  ：{spec.payload.vocab_version}")
    _echo(
        f"语义      ：{spec.semantics.position_fn} / 中性带 {spec.semantics.neutral_band} "
        f"/ 多空 {spec.semantics.long_short} / 预热 {spec.semantics.warmup_bars}"
    )
    _echo("─" * 60)
    _echo(f"验证集得分：{evidence.val_score}（{evidence.wf_folds} 折 WF）")
    _echo(f"试验次数  ：{evidence.n_trials}")
    _echo(f"DSR       ：{evidence.deflated_sharpe}")
    _echo(
        f"hold-out  ：{evidence.holdout_sharpe if evidence.holdout_sharpe is not None else '未消费'}"
    )
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


# ── backtest ────────────────────────────────────────────────────────────────
@app.command()
def backtest(
    spec_path: str = typer.Option(..., "--spec", help="StrategySpec JSON 路径"),
    data: str = typer.Option(..., "--data", help="行情数据文件（parquet / csv）"),
    market: str = typer.Option("", "--market", help="市场画像名（缺省按数据自带 profile）"),
    rolling_window: int = typer.Option(
        0, "--rolling-window", help="滚动夏普窗口（0 = 按年化 bar 数的 1/12）"
    ),
    out: str = typer.Option("", "--out", help="回测结果 JSON 输出路径（缺省不落盘）"),
) -> None:
    """全样本回测：tokens + 行情 + 画像 → 逐 bar 资金曲线 / 滚动夏普 / 交易统计。

    ⚠ 口径：这里的 Sharpe / Sortino 由**逐 bar 净收益**算出；
    ``spec.evidence.val_score`` 是搜索适应度（复合评分），**不是绩效**，
    本命令不读它、页面也不得把它当收益展示。
    """
    try:
        spec = read_spec(spec_path)
    except (MiaoSuanError, FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法读取 spec 文件 {spec_path!r}：{exc}")
    # 用正向 isinstance 收窄（``_die`` 的返回类型不是 NoReturn，负向收窄不生效）。
    factor_payload = spec.payload
    tokens: tuple[int, ...] = ()
    if isinstance(factor_payload, FactorPayload):
        tokens = tuple(factor_payload.tokens)
    else:
        _die(f"spec 载荷不是因子（{type(factor_payload).__name__}），无法回测")

    try:
        panel = load(data)
    except (MiaoSuanError, FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法加载行情数据 {data!r}：{exc}")

    profile_name = market or panel.market_profile_name or spec.provenance.market
    if not profile_name:
        _die("无法确定市场画像：请用 --market 指定，或确保数据自带 profile")
    try:
        profile = get_profile(profile_name)
    except (KeyError, ValueError) as exc:
        _die(f"未知市场画像 {profile_name!r}：{exc}")

    try:
        result = run_full_backtest(
            tokens,
            panel,
            profile,
            rolling_window=rolling_window or None,
            neutral_band=float(spec.semantics.neutral_band or 0.0) or MIN_TRADE_EXPOSURE,
            long_only=not spec.semantics.long_short,
        )
    except (MiaoSuanError, ValueError, RuntimeError, TypeError, IndexError) as exc:
        _die(f"回测执行失败：{type(exc).__name__}: {exc}")

    payload = result.to_dict()
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    m = result.metrics
    tr = result.trades
    _echo(f"策略      ：{spec.name}  spec_id={spec.spec_id}")
    _echo(f"数据      ：{data}  {result.symbol} {result.timeframe}  {result.n_bars} 根")
    _echo(
        f"市场画像  ：{profile.name}  单边成本率={result.cost_rate:.6f}  年化 bar={result.periods_per_year}"
    )
    _echo("─" * 60)
    _echo(f"累计收益  ：{result.total_return:.6f}")
    _echo(f"年化收益  ：{m.annual_return:.6f}")
    _echo(f"Sharpe    ：{m.sharpe:.4f}")
    _echo(f"Sortino   ：{m.sortino:.4f}")
    _echo(f"最大回撤  ：{m.max_drawdown:.6f}")
    _echo(f"交易数    ：{tr.n_trades}")
    _echo(f"胜率      ：{tr.win_rate:.4f}")
    _echo(f"盈亏比    ：{tr.profit_factor:.4f}（均盈 {tr.avg_win:.6f} / 均亏 {tr.avg_loss:.6f}）")
    _echo("─" * 60)
    _echo(
        "口径提醒  ：Sharpe/Sortino 由逐 bar 净收益算出；"
        f"spec 的 val_score={spec.evidence.val_score} 是搜索适应度，不是绩效"
    )
    if out:
        _echo(f"结果已写入：{out}")


# ── tune ──────────────────────────────────────────────────────────────────
@app.command()
def tune(
    spec: str = typer.Option(..., "--spec", help="策略 Spec JSON 路径"),
    data: str = typer.Option(..., "--data", help="行情数据文件（parquet / csv）"),
    param_spaces_json: str = typer.Option(
        "",
        "--param-spaces",
        help='参数空间 JSON 字符串（如 \'[{"name":"neutral_band","default":0.05,"kind":"float","low":0.01,"high":0.2}]\'）',
    ),
    n_trials: int = typer.Option(50, "--n-trials", help="最大试验次数"),
    metric: str = typer.Option(
        "sharpe", "--metric", help="优化目标指标（sharpe/sortino/calmar/composite）"
    ),
    seed: int = typer.Option(42, "--seed", help="随机种子"),
    out: str = typer.Option("", "--out", help="结果输出路径（缺省自动命名）"),
) -> None:
    """参数寻优：对已导出策略的语义参数做 TPE 采样调优。"""
    import json as _json

    # 加载 spec
    try:
        spec_obj = read_spec(spec)
    except (MiaoSuanError, FileNotFoundError, OSError, ValueError) as exc:
        _die(f"无法读取 spec 文件 {spec!r}：{exc}")

    # 解析因子 token
    from .ir.schema import FactorPayload

    tokens: tuple[int, ...] = ()
    if isinstance(spec_obj.payload, FactorPayload):
        tokens = tuple(spec_obj.payload.tokens)
    else:
        _die("spec 载荷不是因子体，无法寻优（需先挖掘产出 FactorPayload）")

    # 解析参数空间
    raw_spaces = _json.loads(param_spaces_json) if param_spaces_json else []
    from .adapters.base import ParamSpace

    param_spaces = [ParamSpace.from_dict(s) for s in raw_spaces]
    if not param_spaces:
        # 尝试从 spec 自动提取
        auto_spaces = load_param_space_from_spec(spec)
        if auto_spaces:
            param_spaces = auto_spaces

    config = TuneConfig(
        spec_path=spec,
        param_spaces=param_spaces,
        n_trials=n_trials,
        metric=metric,
        seed=seed,
    )

    _echo(f"寻优配置：{len(param_spaces)} 个参数 · {n_trials} 次试验 · metric={metric}")
    _echo(f"数据：{data}  spec：{spec}")

    engine = TuneEngine(config)
    try:
        result = engine.optimize(tokens, data, spec_obj)
    except Exception as exc:
        _die(f"寻优失败：{type(exc).__name__}: {exc}")

    # 输出结果
    _echo("")
    _echo(f"寻优完成：{len(result.trials)} 次试验 · 耗时 {result.elapsed_sec:.1f}s")
    _echo(f"最优评分：{result.best.score:.4f}")
    _echo("最优参数：")
    for k, v in sorted(result.best.params.items()):
        _echo(f"  {k} = {v:.6f}")

    # 落盘
    out_path = Path(out) if out else Path(config.out_dir) / f"tune_{result.result_id}.json"
    save_tune_result(result, out_path)
    _echo(f"结果已写入：{out_path}")


# ── ui ──────────────────────────────────────────────────────────────────────
@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址（仅本机）"),
    port: int = typer.Option(8686, "--port", help="监听端口"),
) -> None:
    """启动本地 Web 仪表盘（模式 A 可视化：挖掘 → 导出 → 校验）。"""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        _die("缺少 Web 依赖，请在项目 venv 中安装：pip install fastapi uvicorn")
        raise AssertionError from exc

    from .webui.server import create_app

    _echo(f"妙算仪表盘已启动：http://{host}:{port}（Ctrl+C 退出）")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


def main() -> int:
    """供 ``python -m miaosuan.cli`` 使用；返回进程退出码。"""
    try:
        app()
    except SystemExit as exc:  # typer/click 的退出信号
        return int(exc.code) if isinstance(exc.code, int) else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
