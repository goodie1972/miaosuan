"""CLI 编排层（M14 支撑）：``search → gate → IR``。

为什么单独一层
--------------
架构要求「``search`` 不依赖 ``gate``、``adapters`` 不依赖 ``cli``/``tune``」。
把「搜索结果 → 门禁证据 → :class:`StrategySpec`」这段组装放在这里，CLI 只负责
参数解析与 IO，编排逻辑可独立测试（不碰 argparse / 文件系统）。

:func:`run_mine` 是「模式 A」一条命令闭环的**唯一**入口：

    数据 → 切分（封印 hold-out）→ 搜索 → 门禁 → StrategySpec

hold-out 说明：本层**不消费** hold-out 封印。:meth:`Evidence.holdout_sharpe` 因此
保持 ``None``，导出文件里会写明「未消费 hold-out」——这正是我们要的诚实。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .config import AppConfig, ConfigError
from .core.vocab import VOCAB_VERSION
from .data.panel import Panel
from .data.split import HoldoutSealRegistry
from .ir.provenance import build_provenance
from .ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
from .market.profiles import FrozenMarketProfile, get_profile
from .search.budget import DEFAULT_PROFILE
from .search.mine import Candidate, MineResult, mine

__all__ = ["MineOutcome", "build_spec", "resolve_market", "run_mine"]

#: 数据文件未自带 ``market_profile_name`` 时，按标的推断画像（显式 ``--market`` 优先）。
#: 只覆盖已知标的；未知标的**报错**而不是猜，避免用错成本/年化因子。
_SYMBOL_PROFILE_FALLBACK: dict[str, str] = {
    "XAUUSD": "FOREX_XAUUSD",
    "BTCUSDT": "CRYPTO_BTC",
}


def resolve_market(panel: Panel, explicit: str, symbol: str) -> str:
    """确定市场画像名：显式 > 数据自带 > 按标的推断（否则报错）。

    Args:
        panel: 行情面板。
        explicit: CLI 显式指定的画像名（可为空串）。
        symbol: 标的代码（``config.symbol``）。

    Returns:
        画像名。

    Raises:
        ConfigError: 三者都拿不到（提示用户显式传 ``--market``）。
    """
    if explicit:
        return explicit
    if panel.market_profile_name:
        return panel.market_profile_name
    inferred = _SYMBOL_PROFILE_FALLBACK.get(symbol.upper())
    if inferred:
        return inferred
    raise ConfigError(
        "无法确定市场画像：数据未自带 market_profile_name，且标的 "
        f"{symbol!r} 没有默认画像。请显式传入 --market"
        "（可选 FOREX_XAUUSD / CRYPTO_BTC / CN_EQUITY_RESEARCH / US_EQUITY_RESEARCH）"
    )


class MineOutcome:
    """一次挖掘编排的完整产物。

    Attributes:
        result: 搜索/门禁原始结果（含可审计证据）。
        spec: 由最优候选组装出的 IR 规格。
        profile: 实际使用的市场画像。
    """

    def __init__(
        self, result: MineResult, spec: StrategySpec, profile: FrozenMarketProfile
    ) -> None:
        """保存编排产物。"""
        self.result = result
        self.spec = spec
        self.profile = profile


def _gate_reasons(verdict_snapshot: str) -> tuple[str, ...]:
    """从 verdict 快照里取出触发原因（硬失败在前，解析失败返回空元组）。

    :meth:`miaosuan.gate.verdict.GateVerdict.snapshot` 的键是
    ``hard_failures`` / ``soft_failures``；``reasons`` 作为兼容别名一并接受。
    绝不因快照损坏而阻断导出——只是丢失原因说明。
    """
    if not verdict_snapshot:
        return ()
    try:
        data = json.loads(verdict_snapshot)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(data, Mapping):
        return ()
    reasons: list[str] = []
    for key in ("hard_failures", "soft_failures", "reasons"):
        values = data.get(key)
        if isinstance(values, (list, tuple)):
            reasons.extend(str(v) for v in values)
    return tuple(reasons)


def build_spec(
    candidate: Candidate,
    result: MineResult,
    *,
    config: AppConfig,
    profile: FrozenMarketProfile,
    budget_profile: str,
    data_fingerprint: str,
    git_sha: str = "unknown",
    n_folds: int = 5,
    created_at: str = "",
    name: str = "",
) -> StrategySpec:
    """把候选 + 搜索结果组装成 :class:`StrategySpec`（搜索 → 门禁 → IR）。

    Args:
        candidate: 门禁判定后的候选。
        result: :func:`~miaosuan.search.mine.mine` 的完整结果（提供 n_trials 等证据）。
        config: 生效配置。
        profile: 市场画像。
        budget_profile: 预算档位名。
        data_fingerprint: 数据指纹（``Panel.fingerprint``）。
        git_sha: 代码 commit sha（CLI 用 ``resolve_git_sha`` 取到后传入）。
        n_folds: Walk-Forward 折数。
        created_at: 生成时刻（留空取当前 UTC）。
        name: 策略名（留空则按 周期/标的 自动生成）。

    Returns:
        IR 规格（可直接交给 :class:`~miaosuan.adapters.algoforge.AlgoforgePort`）。
    """
    spec_name = name or f"{config.timeframe.lower()}_{config.symbol.lower()}_miaosuan"
    cost_sensitivity = {f"{float(k):g}": float(v) for k, v in candidate.sharpes.items()}
    return StrategySpec(
        name=spec_name,
        payload=FactorPayload(
            tokens=tuple(int(t) for t in candidate.tokens),
            vocab_version=VOCAB_VERSION,
        ),
        semantics=Semantics(
            position_fn="tanh",
            neutral_band=0.05,
            long_short=True,
            warmup_bars=499,
            timeframe=config.timeframe,
            roll_window=500,
        ),
        evidence=Evidence(
            n_trials=int(result.n_evaluations),
            wf_folds=int(n_folds),
            val_score=float(candidate.val_score),
            holdout_sharpe=None,  # 不消费 hold-out 封印
            cost_sensitivity=cost_sensitivity,
            deflated_sharpe=float(candidate.dsr),
            gate_verdict=str(candidate.verdict),
            gate_reasons=_gate_reasons(candidate.verdict_snapshot),
        ),
        provenance=build_provenance(
            vocab_version=VOCAB_VERSION,
            data_fingerprint=data_fingerprint,
            seed=int(config.seed),
            market=profile.name,
            budget=budget_profile,
            git_sha=git_sha,
            created_at=created_at,
            config_snapshot=_snapshot(config, budget_profile, n_folds),
        ),
    )


def _snapshot(config: AppConfig, budget_profile: str, n_folds: int) -> dict[str, Any]:
    """构造 provenance 配置快照（在 config 快照上叠加 CLI 级参数）。"""
    snapshot: dict[str, Any] = dict(config.to_snapshot())
    snapshot["budget_profile"] = budget_profile
    snapshot["wf_folds"] = int(n_folds)
    return snapshot


def run_mine(
    panel: Panel,
    *,
    config: AppConfig,
    budget_profile: str = DEFAULT_PROFILE,
    top_k: int = 5,
    n_folds: int = 5,
    gap: int = 20,
    registry: HoldoutSealRegistry | None = None,
    market: str = "",
    git_sha: str = "unknown",
    created_at: str = "",
    name: str = "",
) -> MineOutcome:
    """模式 A 的唯一编排入口：搜索 → 门禁 → IR。

    Args:
        panel: 全量行情面板。
        config: 生效配置（CLI 从 env 构造后传入）。
        budget_profile: 预算档位（``quick`` / ``standard`` / ``deep``）。
        top_k: 汇总候选数。
        n_folds: Walk-Forward 折数。
        gap: 折间间隔。
        registry: hold-out 封印台账（测试用内存实现）。
        market: 市场画像名（留空按 ``panel.market_profile_name`` 查找）。
        git_sha: 代码 sha。
        created_at: 生成时刻。
        name: 策略名。

    Returns:
        :class:`MineOutcome`（搜索结果 + IR 规格 + 市场画像）。

    Raises:
        ValueError: 预算档位非法。
    """
    if budget_profile not in ("quick", "standard", "deep"):
        raise ValueError(
            f"未知预算档位 {budget_profile!r}（可选 quick / standard / deep）"
        )
    profile = get_profile(resolve_market(panel, market, config.symbol))
    result = mine(
        panel,
        config=config,
        registry=registry,
        profile=profile,
        budget_profile=budget_profile,
        top_k=int(top_k),
        n_folds=int(n_folds),
        gap=int(gap),
    )
    spec = build_spec(
        result.best,
        result,
        config=config,
        profile=profile,
        budget_profile=budget_profile,
        data_fingerprint=panel.fingerprint,
        git_sha=git_sha,
        n_folds=int(n_folds),
        created_at=created_at,
        name=name,
    )
    return MineOutcome(result=result, spec=spec, profile=profile)
