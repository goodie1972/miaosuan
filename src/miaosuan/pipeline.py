# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import AppConfig, ConfigError
from .core.features import compute_features
from .core.vocab import VOCAB_VERSION
from .data.panel import Panel
from .data.split import HoldoutSealRegistry, make_split
from .ir.provenance import build_provenance
from .ir.schema import ComboPayload, Evidence, FactorPayload, Semantics, StrategySpec
from .market.profiles import FrozenMarketProfile, get_profile
from .search.budget import DEFAULT_PROFILE, Budget
from .search.combo import COMBO_METHODS, ComboCandidate, combine
from .search.ga import AMFitnessEvaluator
from .search.mine import Candidate, MineResult, compute_target_ret, mine

__all__ = [
    "ComboOutcome",
    "MineOutcome",
    "build_spec",
    "resolve_market",
    "run_combine",
    "run_mine",
]

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


class ComboOutcome:
    """一次多因子组合编排的完整产物（``docs/combo-impl-spec.md`` §3）。

    Attributes:
        candidate: 组合门禁结论（成员 / 权重 / DSR / verdict / **Σ 累加的 n_trials**）。
        spec: 由组合体组装出的 IR 规格（``ComboPayload`` kind）。
        profile: 实际使用的市场画像（成本率与年化因子的来源）。
        label_sources: ``(symbol 来源, timeframe 来源)``——供 CLI 打「标签」行用，
            与 :func:`derive_labels` 的返回值一致。
    """

    def __init__(
        self,
        candidate: ComboCandidate,
        spec: StrategySpec,
        profile: FrozenMarketProfile,
        label_sources: tuple[str, str] = ("配置默认", "配置默认"),
    ) -> None:
        """保存编排产物。"""
        self.candidate = candidate
        self.spec = spec
        self.profile = profile
        self.label_sources = label_sources


def derive_labels(
    config: AppConfig,
    panel: Panel,
    *,
    explicit_timeframe: str = "",
    explicit_symbol: str = "",
) -> tuple[AppConfig, str, str]:
    """按 **CLI 显式 > env 显式 > 数据文件名 > 配置默认** 派生 ``symbol`` / ``timeframe``。

    ``mine`` 与 ``combine`` 共用同一套优先级（组合 spec 的 ``semantics.timeframe``
    必须与单因子一致，否则回测/导出两边对不上）。

    「env 显式」用**值不等于配置默认**判定 —— 不二次读环境变量，守住
    ``config.py`` 是唯一 env 边界的铁律。已知边界：env 恰好设成默认值时会被当未
    设置、落到数据文件名（此时以数据为准反而更正确，spec 与 backtest 一致）。

    Args:
        config: 生效配置（CLI 显式覆盖可能已并入）。
        panel: 数据面板（文件名推断出的 ``symbols`` / ``timeframe``）。
        explicit_timeframe: CLI 的 ``--timeframe`` 原始值（空 = 未传）。
        explicit_symbol: CLI 的 ``--symbol`` 原始值（空 = 未传）。

    Returns:
        ``(配置, symbol 来源, timeframe 来源)``，来源为四个中文标签之一。
    """
    default_cfg = AppConfig()

    if explicit_timeframe:
        tf_source = "CLI 显式"
    elif config.timeframe != default_cfg.timeframe:
        tf_source = "env 显式"
    elif panel.timeframe:
        config = config.with_overrides(timeframe=panel.timeframe)
        tf_source = "数据文件名"
    else:
        tf_source = "配置默认"

    derived_symbol = str(panel.symbols[0]).strip() if panel.symbols else ""
    if explicit_symbol:
        sym_source = "CLI 显式"
    elif config.symbol != default_cfg.symbol:
        sym_source = "env 显式"
    elif derived_symbol and derived_symbol != "SINGLE":
        config = config.with_overrides(symbol=derived_symbol)
        sym_source = "数据文件名"
    else:
        sym_source = "配置默认"

    return config, sym_source, tf_source


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
        IR 规格（可直接交给 :class:`~miaosuan.adapters.shenji.ShenjiPort`）。
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
    budget: Budget | None = None,
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
        budget: 显式预算对象（``None`` = 按 ``budget_profile`` 用
            :meth:`~miaosuan.search.budget.Budget.from_profile` 构造）。
            运行时的 ``pop_size`` / ``elite_size`` / ``island_count`` **只读这一份**
            （见 ``search.mine.mine()`` → ``RpnGA.run(budget)``），``config.search``
            里的同名字段不参与运行——CLI 的 ``--pop-size`` 等覆盖必须经此透传，
            否则只写进 config_snapshot、实际种群仍是档位表里的值。

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
        budget=budget,
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


# ── 多因子组合（combine，docs/combo-impl-spec.md §3）────────────────────────


def _load_candidates(path: str | Path) -> dict[str, Any]:
    """读一份 :func:`mine --dump-candidates` 产出的候选 JSON。

    读取放在本层是规格 §3 明文要求的（``run_combine(candidates_paths, ...)`` 传的是
    路径）；IR codec 的 :func:`~miaosuan.ir.codec.read_spec` /
    :func:`~miaosuan.ir.codec.write_spec` 仍然**只由 CLI 调用**，不受影响。

    Args:
        path: 候选 JSON 路径。

    Returns:
        已解析的顶层字典。

    Raises:
        ConfigError: 文件不存在 / 非法 JSON / 顶层不是对象 —— 三条都带路径。
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(
            f"候选文件不存在：{p}", context={"path": str(p)}
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"候选文件读不了：{p}（{exc}）", context={"path": str(p)}
        ) from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"候选文件不是合法 JSON：{p}（{exc}）", context={"path": str(p)}
        ) from exc
    if not isinstance(doc, dict):
        raise ConfigError(
            f"候选文件顶层必须是对象：{p}（实际 {type(doc).__name__}）",
            context={"path": str(p)},
        )
    return doc


def _select_members(
    docs: Sequence[tuple[Path, dict[str, Any]]],
    *,
    top_per_file: int,
) -> tuple[list[tuple[int, ...]], list[Path]]:
    """按「每文件 rank0..k + 跨文件去重相同 tokens」挑成员（规格 §4）。

    Args:
        docs: ``(候选文件路径, 已解析内容)`` 序列。
        top_per_file: 每个文件取前几名（``k = top_per_file - 1``）。

    Returns:
        ``(成员 token 列表, 按 rank 入选过候选的运行路径列表)`` —— 第二项用于
        n_trials 按**运行**去重。

    Raises:
        ConfigError: 某文件没有任何候选、或去重后不足 2 个成员。
    """
    members: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    selected_runs: list[Path] = []
    for path, doc in docs:
        entries = doc.get("candidates")
        if not isinstance(entries, list) or not entries:
            raise ConfigError(
                f"候选文件没有 candidates：{path}（请用 mine --dump-candidates 重新落盘）",
                context={"path": str(path)},
            )
        rank_hit = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if int(entry.get("rank", 1 << 30)) >= top_per_file:
                continue
            rank_hit = True
            tokens = tuple(int(t) for t in (entry.get("tokens") or ()))
            if not tokens or tokens in seen:
                continue
            seen.add(tokens)
            members.append(tokens)
        if rank_hit:
            selected_runs.append(path)
    if len(members) < 2:
        raise ConfigError(
            f"去重后只剩 {len(members)} 个成员——组合至少需要 2 个**互不相同**的因子"
            f"（当前 top_per_file={top_per_file}）",
            context={"n_members": len(members), "top_per_file": top_per_file},
        )
    return members, selected_runs


def _sum_run_n_trials(
    docs: Sequence[tuple[Path, dict[str, Any]]],
    selected_runs: Sequence[Path],
) -> int:
    """Σ 各**来源运行**的 n_trials（按运行去重，规格 §1.3）。

    运行身份 = 候选文件路径（一次 ``mine`` 落盘一份）。同一条运行无论贡献了几条
    成员都只计一次；没有候选按 rank 入选的运行不计。取值优先用顶层 ``n_trials``
    （任务0 落的），旧文件退回 ``verdict_snapshot.n_trials``（两者同源，都是该次
    ``Evidence.n_trials``）。

    注意：**成员**跨文件去重**不会**减少这里的运行数（见 :func:`_select_members`）——
    同一条公式被两个运行各自挖到，两次搜索的试验数都算，否则 DSR 校正不足。

    权重搜索次数对 **equal 与 ic 都是 0**：两者都是确定性映射，没有在假设空间里
    搜索（§1.2 禁 GA 搜权重），故不额外累加。

    Raises:
        ConfigError: 某来源运行拿不到 n_trials（旧格式且快照缺失）。
    """
    wanted = {str(p) for p in selected_runs}
    total = 0
    for path, doc in docs:
        if str(path) not in wanted:
            continue
        raw = doc.get("n_trials")
        if raw is None:
            for entry in doc.get("candidates") or ():
                if not isinstance(entry, dict):
                    continue
                snap = entry.get("verdict_snapshot")
                # 落盘时已解析成 dict；手写/旧文件可能是 JSON 字符串，两者都认，
                # 否则兜底分支永远走不到，格式差异会变成「缺 n_trials」硬失败。
                if isinstance(snap, str):
                    try:
                        snap = json.loads(snap)
                    except json.JSONDecodeError:
                        snap = None
                if isinstance(snap, dict) and snap.get("n_trials") is not None:
                    raw = snap["n_trials"]
                    break
        if raw is None:
            raise ConfigError(
                f"候选文件缺 n_trials，无法做 DSR 多重检验校正：{path}"
                "（请用当前版本的 mine --dump-candidates 重新落盘）",
                context={"path": str(path)},
            )
        total += max(1, int(raw))
    return max(1, total)


def run_combine(
    candidates_paths: Sequence[str | Path],
    *,
    panel: Panel,
    config: AppConfig,
    method: str = "equal",
    seed: int = 0,
    git_sha: str = "unknown",
    top_per_file: int = 3,
    market: str = "",
    created_at: str = "",
) -> ComboOutcome:
    """多因子组合编排：候选校验 → **seal=False** 重建切分 → dev 区 → 组合 → 门禁 → IR。

    与 :func:`run_mine` 同构，差别只在「搜索」被「读候选 + 组合」替代：

    1. **成员校验**（规格 §3）：同 ``timeframe``、同 ``vocab_version``、同
       ``data_fingerprint``，且 timeframe 必须与 ``--data`` 数据文件名推断出的周期
       **一致**——周期只在「非配置默认」时才可信（任务B 修的正是这条），所以数据
       文件名没带周期时直接拒绝，而不是退回配置默认 H1 猜一个；
    2. **只在开发区评估**：``make_split(..., seal=False)`` 重建切分 + ``slice_view(0, val_end)``，
       与 mine 同款结构；``seal=False`` **不写封印台账**（registry 用纯内存实例，
       连文件句柄都不开），hold-out 结构性读不到；
    3. **门禁完全同款**（§1.7）：把 Σ 累加后的 ``n_trials`` 交给
       :func:`miaosuan.search.combo.combine`，内部走与单因子一致的
       cost_curve → DSR → verdict 顺序。

    Args:
        candidates_paths: ``mine --dump-candidates`` 产出的 JSON 路径（≥1 个）。
        panel: ``--data`` 数据文件的面板（开发区切分与特征都来自它）。
        config: 生效配置（label 派生 + 切分比例 + 种子）。
        method: 权重方法（``equal`` / ``ic``，确定性；**禁 GA 搜权重**）。
        seed: 随机种子（>0 时覆盖 config.seed，参与切分指纹）。
        git_sha: 代码 sha。
        top_per_file: 每个候选文件取前几名（默认 rank0..2）。
        market: 显式市场画像名（留空按数据自带 / 标的推断）。
        created_at: 生成时刻（留空取当前 UTC）。

    Returns:
        :class:`ComboOutcome`（组合候选 + IR 规格 + 市场画像 + 标签来源）。

    Raises:
        ConfigError: 方法非法、候选缺失/跨文件不一致、周期不可信、成员不足。
    """
    if method not in COMBO_METHODS:
        raise ConfigError(
            f"未知权重方法 {method!r}（可选 {list(COMBO_METHODS)}；禁止 GA 搜权重）",
            context={"method": method},
        )
    if int(top_per_file) < 1:
        raise ConfigError(
            f"top_per_file 必须 >= 1，收到 {top_per_file}",
            context={"top_per_file": int(top_per_file)},
        )
    if not candidates_paths:
        raise ConfigError("至少要给一个候选文件（--candidates）")

    # 运行身份 = 候选文件路径：同一文件被传两次仍只算**一次**它的 n_trials，
    # 否则 Σ 会虚高 → DSR 被过度压低 → 平白把能过的候选判成 BLOCKED（过严）。
    unique_paths: list[Path] = []
    _seen_paths: set[Path] = set()
    for raw in candidates_paths:
        resolved = Path(raw).expanduser().resolve()
        if resolved in _seen_paths:
            continue
        _seen_paths.add(resolved)
        unique_paths.append(resolved)

    if seed:
        config = config.with_overrides(seed=int(seed))
    # 标签派生与 mine 同一套优先级（规格：semantics.timeframe 必须与单因子同源）。
    config, sym_source, tf_source = derive_labels(config, panel)

    # ── 1. 成员校验（同周期 / 同词表 / 同数据指纹，且周期必须可信）─────────
    if not panel.timeframe:
        raise ConfigError(
            "数据文件名推不出周期（须形如 {symbol}_{timeframe}），无法确认同周期组合"
            "——请把数据文件重命名成带周期的形式",
            context={"symbols": [str(s) for s in panel.symbols]},
        )
    docs: list[tuple[Path, dict[str, Any]]] = [
        (p, _load_candidates(p)) for p in unique_paths
    ]
    for path, doc in docs:
        if not doc.get("timeframe"):
            raise ConfigError(
                f"候选文件缺 timeframe 字段，无法做同周期校验：{path}"
                "（请用当前版本的 mine --dump-candidates 重新落盘）",
                context={"path": str(path)},
            )

    def _distinct(key: str) -> list[Any]:
        values: list[Any] = []
        for _path, doc in docs:
            value = doc.get(key)
            if value not in values:
                values.append(value)
        return values

    timeframes = _distinct("timeframe")
    if len(timeframes) != 1 or timeframes[0] != panel.timeframe:
        raise ConfigError(
            f"候选周期 {timeframes} 与数据文件周期 {panel.timeframe!r} 不一致——"
            "组合只支持同周期成员，且周期必须与 --data 文件一致",
            context={"candidates": timeframes, "data": panel.timeframe},
        )
    vocabs = _distinct("vocab_version")
    if len(vocabs) != 1:
        raise ConfigError(
            f"候选词表版本不一致：{vocabs}——token 只配同一词表才有意义",
            context={"vocab_versions": vocabs},
        )
    if vocabs[0] != VOCAB_VERSION:
        raise ConfigError(
            f"候选词表 {vocabs[0]} 与当前 {VOCAB_VERSION} 不一致——用旧词表挖出的 token "
            "在当前 VM 下会算错，须先用当前版本重新 mine",
            context={"candidate_vocab": vocabs[0], "current_vocab": VOCAB_VERSION},
        )
    fingerprints = _distinct("data_fingerprint")
    if len(fingerprints) != 1:
        raise ConfigError(
            f"候选数据指纹不一致：{len(fingerprints)} 种——成员必须来自同一份数据",
            context={"fingerprints": fingerprints},
        )
    if not fingerprints[0]:
        # 三个字段里只有 fingerprint 缺失时会「None == None」静默通过，必须单独拦。
        raise ConfigError(
            "候选缺 data_fingerprint（旧格式），无法证明成员同源——"
            "请用当前版本的 mine --dump-candidates 重新落盘",
            context={"path": str(docs[0][0])},
        )

    # ── 2. 挑成员 + Σ n_trials（按运行去重）────────────────────────────────
    members, selected_runs = _select_members(docs, top_per_file=int(top_per_file))
    n_trials_total = _sum_run_n_trials(docs, selected_runs)

    # ── 3. seal=False 重建切分，只在开发区评估（与 mine 同款结构）──────────
    profile = get_profile(resolve_market(panel, market, config.symbol))
    split = make_split(
        panel,
        config.split,
        registry=HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY),
        seed=int(config.seed),
        seal=False,
    )
    assert split.indices is not None
    val_end = split.indices.ranges()["val"][1]
    dev_panel = panel.slice_view(0, val_end)
    features = compute_features(dev_panel.to_raw_dict())
    target_ret = compute_target_ret(dev_panel.open)

    evaluator = AMFitnessEvaluator(
        features,
        target_ret,
        cost_rate=float(profile.cost_model.total("buy")),
        periods_per_year=int(profile.bars_per_year),
    )

    # ── 4. 组合 + 同款门禁 ────────────────────────────────────────────────
    candidate = combine(
        members,
        evaluator=evaluator,
        n_trials=n_trials_total,
        method=method,
    )

    # ── 5. 组装 ComboStrategySpec（kind=combo，SPEC_VERSION 不 bump，§1.6）──
    timeframe = config.timeframe
    symbol = config.symbol
    cost_sensitivity = {
        f"{float(k):g}": float(v) for k, v in candidate.sharpes.items()
    }
    snapshot = _snapshot(config, "combo", int(evaluator.n_folds))
    snapshot["combo"] = {
        "method": str(method),
        "top_per_file": int(top_per_file),
        "n_members": candidate.n_members,
        "sources": [str(p) for p in selected_runs],
    }
    spec = StrategySpec(
        name=f"combo_{timeframe.lower()}_{symbol.lower()}",
        payload=ComboPayload(
            members=candidate.members,
            weights=candidate.weights,
            vocab_version=VOCAB_VERSION,
        ),
        semantics=Semantics(
            position_fn="tanh",
            neutral_band=0.05,
            long_short=True,
            warmup_bars=499,
            timeframe=timeframe,
            roll_window=500,
        ),
        evidence=Evidence(
            n_trials=int(candidate.n_trials),
            wf_folds=int(evaluator.n_folds),
            val_score=float(candidate.val_score),
            holdout_sharpe=None,  # 组合同样不消费 hold-out 封印
            cost_sensitivity=cost_sensitivity,
            deflated_sharpe=float(candidate.dsr),
            gate_verdict=str(candidate.verdict),
            gate_reasons=_gate_reasons(candidate.verdict_snapshot),
        ),
        provenance=build_provenance(
            vocab_version=VOCAB_VERSION,
            data_fingerprint=panel.fingerprint,
            seed=int(config.seed),
            market=profile.name,
            budget="combo",  # 组合没有预算档位——不假装它来自某次搜索
            git_sha=git_sha,
            created_at=created_at,
            config_snapshot=snapshot,
        ),
    )
    return ComboOutcome(
        candidate=candidate,
        spec=spec,
        profile=profile,
        label_sources=(sym_source, tf_source),
    )
