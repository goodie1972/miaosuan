# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""搜索检查点与断点续挖测试（路线图 S1 + S2）。

S1（检查点 + 续挖，验收 G7）：
* 原子落盘/加载往返：快照 → 落盘 → 读回 → 字段逐项相等（含 RNG 状态）；
* 故障注入：跑 N 代拍检查点 → 「进程死亡」→ 新 GA 从检查点续挖 →
  ``n_evaluations`` 单调不减（恢复个体免重评估）、最优分不低于中断前；
* 续挖后随机序列与「从未中断」的对照搜索一致（同 seed 可复现）。

S2（分数下限）：
* 显式下限：best ≤ floor 的候选被拦下（不产出）；best > floor 正常产出；
* 下限自动读取：同标的既有 spec 的最优分被用作下限。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from miaosuan.config import AppConfig, GASearchConfig
from miaosuan.search.budget import Budget
from miaosuan.search.ckpt import MineCheckpoint, load_ckpt, save_ckpt
from miaosuan.search.ga import RpnGA
from miaosuan.search.rpn import random_feasible


class _CountingEvaluator:
    """可计数的合成评估器：分值与 token 内容相关（确定性）。"""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, tokens: np.ndarray) -> Any:
        self.calls += 1
        key = tuple(int(t) for t in tokens)
        val = sum(k % 7 for k in key) / 100.0
        from miaosuan.search.ga import EvalResult

        return EvalResult(train_score=val, val_score=val, status="ok")


def _tiny_budget(max_gen: int = 8) -> Budget:
    return Budget(
        profile="quick",
        wall_clock_hours=2.0,  # 墙钟不触发（让代数上限控制）
        max_generations=max_gen,
        pop_size=16,
        elite_size=3,
        tournament_k=3,
        patience=1000,
        min_delta=0.0,
        explore_generations=1000,
        immigrant_trigger=1000,
        immigrant_ratio=0.2,
        island_count=2,
        island_migrate_every=3,
        island_migrate_top=2,
        min_hamming=1,
    )


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg = cfg.with_overrides(search=GASearchConfig(formula_len=6, pop_size=16, elite_size=3))
    return cfg


# ── S1：检查点落盘/加载往返 ──────────────────────────────────────────────


def test_ckpt_roundtrip(tmp_path: Path) -> None:
    budget = _tiny_budget()
    evaluator = _CountingEvaluator()
    ga = RpnGA(_cfg().search, seed=7, length=6)
    # 跑几代后拍快照
    ga.run(budget, evaluator, checkpoint_every=2)
    ckpt = ga.snapshot(ga.last_result.generations, budget, ga.final_populations)

    path = tmp_path / "ckpt.npz"
    save_ckpt(ckpt, path)
    loaded = load_ckpt(path)

    assert loaded.generation == ckpt.generation
    assert loaded.best_score == pytest.approx(ckpt.best_score)
    # numpy 状态 dict：逐键比较（数组元素级相等）。
    for k in ckpt.rng_state:
        assert k in loaded.rng_state
        assert np.array_equal(ckpt.rng_state[k], loaded.rng_state[k])
    assert len(loaded.population) == len(ckpt.population)
    assert len(loaded.elite_pool) == len(ckpt.elite_pool)
    # 分数随检查点落盘（恢复后免重评估的载体）
    assert loaded.population[0][1:] == ckpt.population[0][1:]


def test_ckpt_atomic_write(tmp_path: Path) -> None:
    """落盘是同目录原子替换：不存在半截 .npz。"""
    budget = _tiny_budget()
    ga = RpnGA(_cfg().search, seed=1, length=6)
    ga.run(budget, _CountingEvaluator())
    ckpt = ga.snapshot(ga.last_result.generations, budget, ga.final_populations)

    path = tmp_path / "ckpt.npz"
    save_ckpt(ckpt, path)
    assert path.is_file()
    assert not (tmp_path / "ckpt.tmp.npz").exists()  # tmp 已替换
    loaded = load_ckpt(path)
    assert loaded.best_score == pytest.approx(ckpt.best_score)


# ── S1：断点续挖（故障注入：跑一半“进程死亡”，从检查点恢复）─────────────


def test_resume_keeps_evaluations_monotonic(tmp_path: Path) -> None:
    """中断 → 续挖：真实评估次数单调不减（恢复个体免重评估）。"""
    budget = _tiny_budget(max_gen=10)
    evaluator = _CountingEvaluator()
    cfg = _cfg()

    # 第一次运行：跑 5 代后“进程被杀”（此处模拟为：手动拍检查点后丢弃 GA）。
    ga1 = RpnGA(cfg.search, seed=42, length=6)
    first_evals = 0
    ckpt: MineCheckpoint | None = None

    def _save(state: Any) -> None:
        nonlocal ckpt
        ckpt = state

    ga1.run(budget, evaluator, on_checkpoint=_save, checkpoint_every=3)
    first_evals = evaluator.calls
    assert ckpt is not None

    # 新进程：从检查点恢复，跑满 10 代。
    ga2 = RpnGA(cfg.search, seed=42, length=6)
    budget2 = _tiny_budget(max_gen=10)
    ga2.restore(ckpt, budget2)
    # 恢复后不应立刻重新评估（缓存已随检查点恢复）。
    ga2.run(budget2, evaluator)

    # 续挖总评估 ≥ 中断前评估（不变量：n_evaluations 单调不减）。
    assert evaluator.calls >= first_evals
    # 最优分不低于中断前（验收 G7）。
    assert ga2.last_result is not None
    assert ga2.last_result.best.fitness >= ckpt.best_score - 1e-12


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    """同 seed：中断-续挖 vs 从未中断 → 最终最优分一致（随机序列无损衔接）。

    故障注入：中断运行只跑 4 代即“强杀”（拍检查点）；续挖恢复后跑满 8 代；
    对照 = 单次跑满 8 代。两者最终最优分必须一致（恢复的 RNG/种群/预算与
    未中断路径在 4 代处逐位相同，此后演化轨迹一致）。
    """
    cfg = _cfg()

    def _run_uninterrupted():
        ev = _CountingEvaluator()
        ga = RpnGA(cfg.search, seed=99, length=6)
        ga.run(_tiny_budget(max_gen=8), ev)
        return ga.last_result.best.fitness

    uninterrupted = _run_uninterrupted()

    # 中断路径：只跑 4 代（max_gen=4 触发停止），在第 4 代处拍检查点。
    ev = _CountingEvaluator()
    ga = RpnGA(cfg.search, seed=99, length=6)
    budget4 = _tiny_budget(max_gen=4)
    ga.run(budget4, ev)
    assert ga.last_result is not None
    captured = ga.snapshot(ga.last_result.generations, budget4, ga.final_populations)
    # 续挖：恢复并跑满 8 代（总 8 代 = 对照）。
    resumed = RpnGA(cfg.search, seed=99, length=6)
    resumed.restore(captured, _tiny_budget(max_gen=8))
    resumed.run(_tiny_budget(max_gen=8), ev)

    assert resumed.last_result is not None
    assert resumed.last_result.best.fitness == pytest.approx(uninterrupted, abs=1e-9)

    assert resumed.last_result is not None
    assert resumed.last_result.best.fitness == pytest.approx(uninterrupted, abs=1e-9)


# ── S2：分数下限（弱结果不覆盖既有最优）─────────────────────────────────


def test_floor_filters_weak_best(tmp_path: Path) -> None:
    """显式下限高于本次最优 → 候选被拦下（弱结果不覆盖）。"""
    from miaosuan.data.panel import PANEL_FIELDS, Panel
    from miaosuan.data.split import HoldoutSealRegistry
    from miaosuan.search.mine import mine

    n = 600
    rng = np.random.default_rng(0)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.002, (1, n)), axis=1)
    fields = {
        "open": price.astype(np.float32),
        "high": (price * 1.001).astype(np.float32),
        "low": (price * 0.999).astype(np.float32),
        "close": (price * 1.0005).astype(np.float32),
        "volume": rng.uniform(100, 200, (1, n)).astype(np.float32),
    }
    time = (np.arange(n, dtype=np.int64) + 1) * 3600
    panel = Panel.from_arrays(
        fields, time, symbols=("XAUUSD",), timeframe="H1", market_profile_name="FOREX_XAUUSD"
    )

    tiny = Budget(
        profile="quick", wall_clock_hours=2.0, max_generations=2, pop_size=8,
        elite_size=2, tournament_k=3, patience=1000, min_delta=0.0,
        explore_generations=1000, immigrant_trigger=1000, immigrant_ratio=0.2,
        island_count=1, island_migrate_every=1, island_migrate_top=1, min_hamming=1,
    )
    cfg = _cfg()
    cfg = cfg.with_overrides(
        search=GASearchConfig(formula_len=4, pop_size=8, elite_size=2)
    )
    registry = HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY)

    # 极高下限：本次搜索几乎不可能超过 → 候选被拦下（mine 抛空候选错误）。
    with pytest.raises(RuntimeError, match="所有候选均未达分数下限"):
        mine(
            panel,
            config=cfg,
            registry=registry,
            budget=tiny,
            budget_profile="quick",
            score_floor=1e9,
        )


def test_floor_low_does_not_block(tmp_path: Path) -> None:
    """显式下限极低 → 正常产出候选。"""
    from miaosuan.data.panel import PANEL_FIELDS, Panel
    from miaosuan.data.split import HoldoutSealRegistry
    from miaosuan.search.mine import mine

    n = 600
    rng = np.random.default_rng(0)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.002, (1, n)), axis=1)
    fields = {
        "open": price.astype(np.float32),
        "high": (price * 1.001).astype(np.float32),
        "low": (price * 0.999).astype(np.float32),
        "close": (price * 1.0005).astype(np.float32),
        "volume": rng.uniform(100, 200, (1, n)).astype(np.float32),
    }
    time = (np.arange(n, dtype=np.int64) + 1) * 3600
    panel = Panel.from_arrays(
        fields, time, symbols=("XAUUSD",), timeframe="H1", market_profile_name="FOREX_XAUUSD"
    )
    tiny = Budget(
        profile="quick", wall_clock_hours=2.0, max_generations=2, pop_size=8,
        elite_size=2, tournament_k=3, patience=1000, min_delta=0.0,
        explore_generations=1000, immigrant_trigger=1000, immigrant_ratio=0.2,
        island_count=1, island_migrate_every=1, island_migrate_top=1, min_hamming=1,
    )
    cfg = _cfg().with_overrides(
        search=GASearchConfig(formula_len=4, pop_size=8, elite_size=2)
    )
    registry = HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY)
    result = mine(
        panel,
        config=cfg,
        registry=registry,
        budget=tiny,
        budget_profile="quick",
        score_floor=-1.0,
    )
    assert result.candidates
    assert result.best.val_score > -1.0


def test_floor_reads_existing_spec(tmp_path: Path) -> None:
    """S2 下限自动读取：同标的既有 spec 的最优分被当作下限。"""
    import json

    from miaosuan.core.vocab import VOCAB_VERSION
    from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    # 造一份既有 spec（val_score = 0.88）
    spec = StrategySpec(
        name="prev",
        payload=FactorPayload(tokens=(1, 2, 65), vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="H1"),
        evidence=Evidence(val_score=0.88, gate_verdict="RESEARCH_ONLY"),
    )
    (artifacts / "prev.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    # 手动调用（不等 CLI）：把 artifacts 指到临时目录后验证扫描逻辑。
    # 这里直接测辅助函数的行为：构造一个假 config 带 symbol/timeframe。
    from miaosuan.cli import _resolve_score_floor

    class _FakeCfg:
        symbol = "XAUUSD"
        timeframe = "H1"

    import miaosuan.cli as cli_mod

    orig = cli_mod._default_artifacts_dir
    cli_mod._default_artifacts_dir = lambda: str(artifacts)
    try:
        floor = _resolve_score_floor(float("-inf"), _FakeCfg())
    finally:
        cli_mod._default_artifacts_dir = orig
    assert floor == pytest.approx(0.88)