"""``mine`` 端到端流水线单测（M10–M12 / 验收 #1 #2 #5）。

关键断言：

* 流水线端到端跑通：数据 → 切分 → 搜索 → 门禁 → 带评分候选（验收 #1）；
* 停止原因取值为契约内四值之一（验收 #2）；
* **hold-out 结构性未被搜索触碰**：开发区 bar 数 < 全量 bar 数 + 封印恰一次
  + hold-out 起始索引 ≥ 开发区末尾（验收 #5）。
"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.config import AppConfig
from miaosuan.data.panel import PANEL_FIELDS, Panel
from miaosuan.data.split import HoldoutSealRegistry
from miaosuan.errors import HoldoutSealedError
from miaosuan.search.budget import (
    STOP_CONVERGED,
    STOP_EARLY_STOP,
    STOP_MAX_GENERATIONS,
    STOP_WALL_CLOCK,
    Budget,
)
from miaosuan.search.mine import MineResult, compute_target_ret, mine


class _SpyRegistry(HoldoutSealRegistry):
    """记录 ``seal`` 调用的封印台账（证明封印恰一次、且只有切分路径调用）。"""

    def __init__(self) -> None:
        super().__init__(HoldoutSealRegistry.IN_MEMORY)
        self.seal_fingerprints: list[str] = []

    def seal(self, fingerprint: str, *, meta: dict[str, object] | None = None) -> None:
        self.seal_fingerprints.append(fingerprint)
        super().seal(fingerprint, meta=meta)


def _panel(n: int = 1200, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.002, (1, n)), axis=1)
    fields = {
        "open": price.astype(np.float32),
        "high": (price * 1.001).astype(np.float32),
        "low": (price * 0.999).astype(np.float32),
        "close": (price * 1.0005).astype(np.float32),
        "volume": rng.uniform(100, 200, (1, n)).astype(np.float32),
    }
    assert set(fields) == set(PANEL_FIELDS)
    time = (np.arange(n, dtype=np.int64) + 1) * 3600
    return Panel.from_arrays(
        fields, time, symbols=("XAUUSD",), timeframe="H1", market_profile_name="FOREX_XAUUSD"
    )


def _tiny_budget() -> Budget:
    return Budget(
        profile="quick",
        wall_clock_hours=0.5,
        max_generations=4,
        pop_size=20,
        elite_size=3,
        tournament_k=3,
        patience=1000,
        min_delta=0.005,
        explore_generations=2,
        immigrant_trigger=3,
        immigrant_ratio=0.2,
        island_count=2,
        island_migrate_every=2,
        island_migrate_top=2,
        min_hamming=2,
    )


def test_compute_target_ret_matches_am_formula() -> None:
    open_arr = np.array([[10.0, 11.0, 12.1, 13.31]], dtype=np.float64)
    tret = compute_target_ret(open_arr)
    # target_ret[t] = log(open[t+2]/open[t+1])
    assert tret[0, 0] == pytest.approx(np.log(12.1 / 11.0))
    assert tret[0, 1] == pytest.approx(np.log(13.31 / 12.1))
    assert tret[0, -1] == 0.0
    assert tret[0, -2] == 0.0


def test_mine_end_to_end() -> None:
    panel = _panel()
    config = AppConfig(seed=20260910)
    registry = _SpyRegistry()

    result = mine(panel, config=config, registry=registry, budget=_tiny_budget(), top_k=3)

    assert isinstance(result, MineResult)
    # ① 端到端产出带评分候选
    assert result.candidates, "必须产出至少一个候选"
    assert result.best is result.candidates[0]
    for c in result.candidates:
        assert len(c.tokens) == 8
        assert c.verdict in {"DEPLOYABLE", "RESEARCH_ONLY", "BLOCKED"}
        assert c.verdict_snapshot
        assert np.isfinite(c.val_score)
    # ② 停止原因在契约内
    assert result.stop_reason in {STOP_WALL_CLOCK, STOP_EARLY_STOP, STOP_MAX_GENERATIONS, STOP_CONVERGED}
    # ③ 多样性地板
    assert result.diversity_ratio >= 0.4


def test_mine_holdout_untouched() -> None:
    panel = _panel()
    config = AppConfig()
    registry = _SpyRegistry()

    result = mine(panel, config=config, registry=registry, budget=_tiny_budget())

    # 结构性证据：开发区严格小于全量
    assert result.holdout_untouched is True
    assert result.dev_bars < result.n_bars
    # 封印恰一次，且指纹等于切分产出的 hold-out seal
    assert len(registry.seal_fingerprints) == 1
    assert registry.seal_fingerprints[0] == result.split.holdout_seal
    assert result.split.is_sealed()
    # hold-out 区间起始索引 ≥ 开发区末尾（不相交）
    assert result.split.indices is not None
    holdout_start = int(result.split.indices.holdout[0])
    assert holdout_start >= result.dev_bars


def test_mine_second_run_same_registry_raises() -> None:
    panel = _panel()
    config = AppConfig()
    registry = _SpyRegistry()
    mine(panel, config=config, registry=registry, budget=_tiny_budget())
    with pytest.raises(HoldoutSealedError):
        mine(panel, config=config, registry=registry, budget=_tiny_budget())


def test_mine_reproducible_same_seed() -> None:
    panel = _panel()
    config = AppConfig(seed=123)
    a = mine(panel, config=config, registry=_SpyRegistry(), budget=_tiny_budget())
    b = mine(panel, config=config, registry=_SpyRegistry(), budget=_tiny_budget())
    assert a.best.tokens == b.best.tokens
    assert a.stop_reason == b.stop_reason


def test_mine_missing_profile_name_raises() -> None:
    panel = _panel()
    bad = Panel.from_arrays(
        panel.fields, panel.time, symbols=("XAUUSD",), timeframe="H1", market_profile_name=""
    )
    with pytest.raises(ValueError):
        mine(bad, config=AppConfig(), registry=_SpyRegistry(), budget=_tiny_budget())
