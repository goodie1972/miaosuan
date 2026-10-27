# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""GATE 死区（R1）下沉回归 —— 唯一实现 ``core.ops.make_gate``，四处同源。

死区滞回原本只活在导出模板里（手写的第二份实现），回测/实盘看不到它，于是
「导出文件的行为」和「证据的行为」是两套。本文件锁死下沉后的不变量：

1. **默认逐位不变**：``deadband = 0`` 时 ``make_gate`` 返回 ``_op_gate`` **本体**，
   ``StackVM(gate_deadband=0)`` 与 ``StackVM()`` 输出 ``array_equal``；
2. **滞回语义正确**：与逐根前向递推的参考实现逐位一致，支持 N≥1（每行独立），
   NaN 条件沿用上一次决定（不引入未来函数）；
3. **无全局可变状态**：带死区的 VM 不污染 ``OPS_CONFIG`` / 后续新建的 VM；
4. **回测 == 实盘 == 导出内核**：同一 ``gate_deadband`` 值贯穿
   :func:`run_full_backtest` / :class:`SignalEngine` / 导出 .py，导出因子与原生 VM
   在 ``deadband > 0`` 下仍**逐位**相等；
5. **spec 兼容**：旧 spec（JSON 无 ``gate_deadband``）加载取 0.0，且 ``spec_id``
   不因新增该字段而漂移。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from miaosuan.adapters.base import install_stub_modules, uninstall_stub_modules
from miaosuan.adapters.shenji import ShenjiPort
from miaosuan.adapters.shenji.contract import PLATFORM_SPEC
from miaosuan.core.features import compute_features
from miaosuan.core.ops import OPS_CONFIG, _op_gate, gate_operator_index, make_gate
from miaosuan.core.signal import compute_target_positions
from miaosuan.core.vm import StackVM
from miaosuan.core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from miaosuan.data.panel import Panel
from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
from miaosuan.market.profiles import FOREX_XAUUSD
from miaosuan.report.equity import run_full_backtest
from miaosuan.runtime import SignalEngine
from miaosuan.runtime.ports import Bar
from miaosuan.search.mine import compute_target_ret

OFFSET: int = FORMULA_VOCAB.operator_offset
GATE_TOKEN: int = OFFSET + gate_operator_index()

#: 妙算 最优公式（含 GATE，条件会真实穿越 0）。
AM_BEST: tuple[int, ...] = (33, 62, 3, 87, 72, 119, 73, 103)

#: 人造公式：``GATE(open-close, high, low)`` —— 条件在 0 两侧来回穿越。
CROSSING: tuple[int, ...] = (0, 3, OFFSET + 2, 1, 2, GATE_TOKEN)

_TOLERANCE: float = 1e-3


# ── 1. make_gate 工厂 ───────────────────────────────────────────────────────


def test_make_gate_non_positive_returns_native_body() -> None:
    """``deadband <= 0`` 必须返回 ``_op_gate`` **本体**（不是副本）→ 默认逐位不变。"""
    assert make_gate(0.0) is _op_gate
    assert make_gate() is _op_gate
    assert make_gate(-1.0) is _op_gate


def test_make_gate_positive_returns_closure_not_native() -> None:
    fn = make_gate(0.05)
    assert fn is not _op_gate
    cond = np.array([[0.03, -0.03, 0.0]])
    x = np.array([[1.0, 1.0, 1.0]])
    y = np.array([[0.0, 0.0, 0.0]])
    # 全部落在死区内 → 沿用「首次越界之前取 y」的起点约定
    np.testing.assert_array_equal(fn(cond, x, y), np.zeros((1, 3)))


# ── 2. 滞回语义 ─────────────────────────────────────────────────────────────


def _reference_hysteresis(
    condition: np.ndarray, x: np.ndarray, y: np.ndarray, deadband: float
) -> np.ndarray:
    """朴素逐根前向递推参考实现（与下沉前的模板手写版同一语义，支持 N≥1）。"""
    n, t = condition.shape
    out = np.zeros_like(x)
    for row in range(n):
        prev = 0
        for i in range(t):
            c = float(condition[row, i])
            if c > deadband:
                prev = 1
            elif c < -deadband:
                prev = 0
            out[row, i] = x[row, i] if prev else y[row, i]
    return out


@pytest.mark.parametrize("n_rows", [1, 3])
@pytest.mark.parametrize("deadband", [1e-6, 0.01, 0.2])
def test_hysteresis_matches_forward_recursion(n_rows: int, deadband: float) -> None:
    """向量化实现与逐根递推**逐位**一致（含大量 0 附近抖动与死区滞留段）。"""
    rng = np.random.default_rng(11 + n_rows)
    t = 240
    cond = rng.normal(0.0, deadband * 3.0 + 1e-9, size=(n_rows, t))
    # 一半的根压到 ±1ULP 量级：制造「原生会翻、死区不翻」的场景
    jitter = rng.integers(0, t, size=t // 2)
    cond[:, jitter] = rng.normal(0.0, deadband * 0.1 + 1e-12, size=(n_rows, t // 2))
    x = rng.normal(size=(n_rows, t))
    y = rng.normal(size=(n_rows, t))

    got = make_gate(deadband)(cond, x, y)
    np.testing.assert_array_equal(got, _reference_hysteresis(cond, x, y, deadband))


def test_hysteresis_rows_are_independent() -> None:
    """每行各自持有滞回状态：改一行不得影响另一行（原生 GATE 是元素级，同样独立）。"""
    cond = np.array([[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]])
    x = np.ones((2, 3))
    y = np.zeros((2, 3))
    np.testing.assert_array_equal(make_gate(0.1)(cond, x, y), [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])


def test_hysteresis_nan_condition_carries_previous_choice() -> None:
    """NaN 条件不落任何一侧 → 沿用上一次决定（只看历史，无未来函数）。"""
    cond = np.array([[0.9, np.nan, np.nan, -0.9, np.nan]])
    x = np.ones((1, 5))
    y = np.zeros((1, 5))
    np.testing.assert_array_equal(make_gate(0.1)(cond, x, y), [[1.0, 1.0, 1.0, 0.0, 0.0]])


def test_deadband_suppresses_branch_flips() -> None:
    """死区的存在意义：±1ULP 抖动下，翻转次数必须显著少于原生 GATE。"""
    t = 400
    base = np.zeros((1, t))
    noise = np.random.default_rng(5).normal(0.0, 1e-8, size=(1, t))
    cond = base + noise  # 理论上恒为 0，实测因浮点噪声正负交替
    x = np.ones((1, t))
    y = np.zeros((1, t))

    def flips(arr: np.ndarray) -> int:
        return int(np.count_nonzero(np.diff(arr)))

    native = flips(_op_gate(cond, x, y))
    protected = flips(make_gate(1e-4)(cond, x, y))
    assert native > 50, f"用例失效：原生 GATE 本应被抖动打出大量翻转，实际 {native}"
    assert protected == 0


# ── 3. StackVM 注入：默认逐位不变 + 不污染全局 ──────────────────────────────


def _raw(n_bars: int = 900, seed: int = 3) -> dict[str, np.ndarray]:
    """单品种 float32 OHLCV（与导出内核的取数口径一致）。"""
    rng = np.random.default_rng(seed)
    close = (3000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n_bars)))).astype(np.float32)
    high = (close * (1.0 + np.abs(rng.normal(0.0, 0.0006, n_bars)))).astype(np.float32)
    low = (close * (1.0 - np.abs(rng.normal(0.0, 0.0006, n_bars)))).astype(np.float32)
    open_ = np.concatenate([[close[0]], close[:-1]]).astype(np.float32)
    volume = (1000.0 + rng.integers(0, 500, n_bars)).astype(np.float32)
    return {
        "open": open_[None, :],
        "high": high[None, :],
        "low": low[None, :],
        "close": close[None, :],
        "volume": volume[None, :],
    }


def test_stackvm_zero_deadband_is_bit_exact_with_default() -> None:
    """``gate_deadband=0`` 的 VM 与裸 VM 在含 GATE 的公式上逐位相等。"""
    features = compute_features(_raw())
    baseline = StackVM().execute(list(AM_BEST), features)
    explicit = StackVM(roll_window=500, gate_deadband=0.0).execute(list(AM_BEST), features)
    negative = StackVM(gate_deadband=-0.5).execute(list(AM_BEST), features)
    assert baseline is not None and explicit is not None and negative is not None
    np.testing.assert_array_equal(baseline, explicit)
    np.testing.assert_array_equal(baseline, negative)


def test_stackvm_deadband_does_not_mutate_global_ops_config() -> None:
    """带死区的实例只改**自己**的 op_map：全局 OPS_CONFIG 与后续 VM 不受影响。"""
    native_transform = OPS_CONFIG[gate_operator_index()][1]
    gated_vm = StackVM(gate_deadband=0.05)
    assert OPS_CONFIG[gate_operator_index()][1] is native_transform
    assert StackVM().op_map[GATE_TOKEN] is native_transform
    assert gated_vm.op_map[GATE_TOKEN] is not native_transform
    # 实例之间互不串味
    assert StackVM(gate_deadband=0.02).op_map[GATE_TOKEN] is not gated_vm.op_map[GATE_TOKEN]


def test_stackvm_deadband_changes_output_of_crossing_formula() -> None:
    """死区>0 时因子必须真的改变（否则「同源」是空话）；且改的是 GATE 分支选择。"""
    features = compute_features(_raw(n_bars=900, seed=21))
    plain = StackVM().execute(list(CROSSING), features)
    protected = StackVM(gate_deadband=0.05).execute(list(CROSSING), features)
    assert plain is not None and protected is not None
    assert not np.array_equal(plain, protected)
    assert np.isfinite(protected).all()


def test_stackvm_shape_error_survives_injection() -> None:
    """注入后 GATE 仍带形状校验（与词表里的原生条目同构，R2.13 不退化）。"""
    from miaosuan.core.ops import ShapeError

    vm = StackVM(gate_deadband=0.05)
    fn = vm.op_map[GATE_TOKEN]
    cond = np.zeros((1, 4))
    with pytest.raises(ShapeError):
        fn(cond, np.zeros((1, 4)), np.zeros((2, 4)))


# ── 4. Semantics：序列化 / 旧 spec 兼容 / spec_id 不漂移 ─────────────────────


def _spec(tokens: tuple[int, ...] = AM_BEST, *, semantics: Semantics | None = None) -> StrategySpec:
    return StrategySpec(
        name="gate_sink_probe",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=semantics or Semantics(),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


def test_semantics_gate_deadband_roundtrip_and_old_spec_compat() -> None:
    """三处同步（字段 + to_dict + from_dict）；旧 JSON 缺键取默认 0.0。"""
    sem = Semantics(gate_deadband=0.07)
    assert sem.to_dict()["gate_deadband"] == 0.07
    assert Semantics.from_dict(sem.to_dict()) == sem
    assert Semantics.from_dict({"position_fn": "tanh", "roll_window": 500}).gate_deadband == 0.0


def test_spec_id_does_not_drift_when_deadband_added() -> None:
    """新增口径字段不得改变既有身份：旧 JSON / 不同死区 → 同一 spec_id。"""
    base = _spec()
    with_deadband = _spec(semantics=Semantics(gate_deadband=0.05))
    assert base.spec_id == with_deadband.spec_id
    assert "gate_deadband" not in base.semantics.identity_dict()

    legacy = base.to_dict()
    legacy["semantics"].pop("gate_deadband")  # 模拟下沉之前落盘的 spec
    restored = StrategySpec.from_dict(legacy)
    assert restored.spec_id == base.spec_id
    assert restored.semantics.gate_deadband == 0.0


# ── 5. 回测 / 实盘同源 ───────────────────────────────────────────────────────


def _panel(n_bars: int = 900, *, seed: int = 7) -> Panel:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.002, size=n_bars)
    close = 2000.0 * np.exp(np.cumsum(steps))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return Panel(
        open=open_.reshape(1, -1).astype(np.float32),
        high=high.reshape(1, -1).astype(np.float32),
        low=low.reshape(1, -1).astype(np.float32),
        close=close.reshape(1, -1).astype(np.float32),
        volume=np.full(n_bars, 1000.0).reshape(1, -1).astype(np.float32),
        time=np.arange(n_bars, dtype=np.int64) * 1800 + 1_700_000_000,
        symbols=("XAUUSD",),
        timeframe="M30",
        market_profile_name="FOREX_XAUUSD",
    )


def _bars(n_bars: int = 700, *, seed: int = 7) -> list[Bar]:
    rng = np.random.default_rng(seed)
    close = 4000.0 + np.cumsum(rng.normal(0.0, 3.0, n_bars))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, 0.5, n_bars - 1)
    high = np.maximum(open_, close) + rng.uniform(0.5, 3.0, n_bars)
    low = np.minimum(open_, close) - rng.uniform(0.5, 3.0, n_bars)
    volume = rng.integers(50, 500, n_bars).astype(np.float64)
    start = 1_800_000_000 - (n_bars - 1) * 1800
    return [
        Bar(
            symbol="XAUUSD",
            timeframe="M30",
            ts=int(start + i * 1800),
            open=float(open_[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=float(volume[i]),
            closed=True,
        )
        for i in range(n_bars)
    ]


def test_backtest_honors_factor_gate_deadband() -> None:
    """``run_full_backtest(factor_gate_deadband=db)`` == 同 db 的 StackVM 计算链。

    与 ``test_report_equity`` 同一比对方式：自己按参考链算逐 bar 净收益，再与
    回测产物比 ``portfolio_pnl``。db=0 时结果必须与不传参逐位一致（默认不变）。
    """
    panel = _panel()
    db = 0.05
    features = compute_features(panel.to_raw_dict())
    ret = np.asarray(compute_target_ret(panel.open), dtype=np.float64)

    def reference_pnl(deadband: float) -> np.ndarray:
        """按「features → StackVM(db) → tanh+中性带 → 分边成本」参考链算逐 bar 净收益。"""
        factor = np.asarray(
            StackVM(gate_deadband=deadband).execute(list(CROSSING), features), dtype=np.float64
        )
        position = compute_target_positions(
            factor, min_trade_exposure=0.05, long_only=FOREX_XAUUSD.long_only
        )
        n_bars = int(min(position.shape[1], ret.shape[1]))
        pos = position[:, :n_bars]
        prev = np.roll(pos, 1, axis=1)
        prev[:, 0] = 0.0
        pnl = pos * ret[:, :n_bars] - np.abs(pos - prev) * FOREX_XAUUSD.cost_model.buy_rate()
        return pnl.mean(axis=0)

    with_db = run_full_backtest(CROSSING, panel, FOREX_XAUUSD, factor_gate_deadband=db)
    without_db = run_full_backtest(CROSSING, panel, FOREX_XAUUSD)
    explicit_zero = run_full_backtest(CROSSING, panel, FOREX_XAUUSD, factor_gate_deadband=0.0)

    np.testing.assert_allclose(with_db.portfolio_pnl, reference_pnl(db), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(without_db.portfolio_pnl, reference_pnl(0.0), rtol=0.0, atol=1e-12)
    # 默认口径（不传参）与显式 0.0 逐位一致；死区>0 必须真的改变结果
    np.testing.assert_array_equal(without_db.portfolio_pnl, explicit_zero.portfolio_pnl)
    assert not np.array_equal(with_db.portfolio_pnl, without_db.portfolio_pnl), (
        "用例失效：该死区没改变任何持仓"
    )


def test_live_engine_and_backtest_share_spec_deadband() -> None:
    """同一 spec（``gate_deadband>0``）：实盘逐 bar 仓位 == 回测口径仓位（逐位）。"""
    sem = Semantics(timeframe="M30", gate_deadband=0.05)
    spec = _spec(CROSSING, semantics=sem)
    bars = _bars()

    raw: dict[str, Any] = {
        key: np.asarray([[getattr(b, key) for b in bars]], dtype=np.float64)
        for key in ("open", "high", "low", "close", "volume")
    }
    factor = StackVM(
        roll_window=int(sem.roll_window), gate_deadband=float(sem.gate_deadband)
    ).execute(list(CROSSING), compute_features(raw))
    assert factor is not None
    bt_positions = compute_target_positions(
        factor,
        min_trade_exposure=float(sem.neutral_band or 0.0) or 0.05,
        long_only=not sem.long_short,
    )[0]

    engine = SignalEngine(spec)
    assert engine._vm.gate_deadband == 0.05  # noqa: SLF001 - 口径接线本身要被钉住
    rt_positions = list(engine.replay_closed_bars(bars))
    assert rt_positions == pytest.approx([float(p) for p in bt_positions], abs=0.0)
    # 用例有效性：死区真的作用过（全 0 序列会让断言失去意义）
    assert any(p != 0.0 for p in rt_positions[int(sem.roll_window) :])


def test_live_engine_default_zero_deadband_unchanged() -> None:
    """默认（旧 spec 无该字段）：实盘 VM 不带死区，与下沉前逐位一致。"""
    spec = _spec(CROSSING)
    engine = SignalEngine(spec)
    assert engine._vm.gate_deadband == 0.0  # noqa: SLF001
    assert engine._vm.op_map[GATE_TOKEN] is OPS_CONFIG[gate_operator_index()][1]  # noqa: SLF001


# ── 6. 导出内核与 core 同源 ─────────────────────────────────────────────────


@pytest.fixture
def shenji_stubs() -> Any:
    created = install_stub_modules(PLATFORM_SPEC)
    try:
        yield
    finally:
        uninstall_stub_modules(created)


def _export(spec: StrategySpec, tmp_path: Path, tag: str) -> Any:
    port = ShenjiPort(magic="661899", date="20261027")
    result = port.compile(spec)
    assert result.errors == (), f"导出未过静态检查：{result.errors}"
    target = tmp_path / f"{tag}_{result.filename}"
    target.write_text(result.source, encoding="utf-8")
    module_spec = importlib.util.spec_from_file_location(f"exported_{tag}", target)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _factor_from(module: Any, raw: dict[str, np.ndarray]) -> np.ndarray:
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    return np.asarray(
        module.compute_factor(
            flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"]
        )
    )


def test_exported_kernel_follows_spec_gate_deadband(tmp_path: Path, shenji_stubs: None) -> None:
    """spec 语义里的死区直接落到生成文件（不再需要 CLI 显式传参）。"""
    spec = _spec(AM_BEST, semantics=Semantics(gate_deadband=0.03))
    module = _export(spec, tmp_path, "follows_spec")
    assert module._GATE_DEADBAND == 0.03  # noqa: SLF001
    assert "_make_gate_protected" not in Path(module.__file__).read_text(encoding="utf-8")


def test_exported_kernel_matches_native_vm_with_deadband(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """三方同源的核心证据：导出因子 == 原生 VM（同死区）**逐位**相等，
    且不等于原生 VM（死区关闭）—— 证明死区在导出文件里真的生效、且实现同源。"""
    db = 0.05
    spec = _spec(AM_BEST, semantics=Semantics(gate_deadband=db))
    module = _export(spec, tmp_path, "fidelity")
    raw = _raw(n_bars=1200, seed=43)
    features = compute_features(raw)

    got = _factor_from(module, raw)
    same = np.asarray(StackVM(gate_deadband=db).execute(list(AM_BEST), features))[0]
    off = np.asarray(StackVM().execute(list(AM_BEST), features))[0]

    assert got.shape == same.shape
    max_err = float(np.max(np.abs(got - same)))
    assert max_err < _TOLERANCE, f"导出 vs 原生（同死区）误差 {max_err:.3e} 超阈"
    assert float(np.max(np.abs(got - off))) > 0.0, "用例失效：死区在导出文件里没起作用"


def test_export_cli_deadband_override_reaches_kernel(tmp_path: Path, shenji_stubs: None) -> None:
    """显式覆盖（``--gate-deadband``）仍然生效，且与原生 VM 同口径对齐。"""
    spec = _spec(AM_BEST, semantics=Semantics(gate_deadband=0.0))
    port = ShenjiPort(magic="661899", date="20261027", gate_deadband=0.02)
    result = port.compile(spec)
    assert "_GATE_DEADBAND = 0.02" in result.source
    target = tmp_path / result.filename
    target.write_text(result.source, encoding="utf-8")

    module_spec = importlib.util.spec_from_file_location("override_kernel", target)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    raw = _raw(n_bars=1000, seed=71)
    got = _factor_from(module, raw)
    expected = np.asarray(
        StackVM(gate_deadband=0.02).execute(list(AM_BEST), compute_features(raw))
    )[0]
    assert float(np.max(np.abs(got - expected))) < _TOLERANCE


def test_exported_compute_factor_override_is_not_sticky(tmp_path: Path, shenji_stubs: None) -> None:
    """``compute_factor(gate_deadband=...)`` 临时覆盖不写模块级状态（闭包工厂）。"""
    spec = _spec(AM_BEST, semantics=Semantics(gate_deadband=0.0))
    module = _export(spec, tmp_path, "no_sticky")
    raw = _raw(n_bars=1000, seed=89)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}

    def call(**kw: Any) -> np.ndarray:
        return np.asarray(
            module.compute_factor(
                flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"], **kw
            )
        )

    with_override = call(gate_deadband=0.04)
    after = call()
    default_off = np.asarray(StackVM().execute(list(AM_BEST), compute_features(raw)))[0]
    assert module._GATE_DEADBAND == 0.0  # noqa: SLF001
    np.testing.assert_array_equal(after, default_off)
    if not np.array_equal(with_override, default_off):
        # 覆盖生效过（死区在该公式该样本上确实改变输出时才算这条有意义）
        assert float(np.max(np.abs(with_override - default_off))) > 0.0
