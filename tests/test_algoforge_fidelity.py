"""M16：数值保真度回归 —— 导出因子 vs 原生 ``StackVM``（max abs err < 1e-3）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from miaosuan.adapters.algoforge import AlgoforgePort
from miaosuan.adapters.algoforge.contract import PLATFORM_SPEC
from miaosuan.adapters.base import install_stub_modules, uninstall_stub_modules
from miaosuan.core.features import compute_features
from miaosuan.core.vm import StackVM
from miaosuan.core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from miaosuan.ir.schema import Evidence, FactorPayload, StrategySpec

OFFSET: int = FORMULA_VOCAB.operator_offset

#: 覆盖：AM 最优公式 + 一元/二元/三元 + GATE + 纯时序算子链
FORMULAS: dict[str, tuple[int, ...]] = {
    "am_best": (33, 62, 3, 87, 72, 119, 73, 103),
    "add_binary": (0, 1, 65),
    "sub_binary": (0, 3, 66),
    "mul_binary": (10, 11, 67),
    "neg_unary": (0, OFFSET + 4),
    "gate_ternary": (0, 1, 3, OFFSET + 7),
    "ts_chain": (0, OFFSET + 22, OFFSET + 54, OFFSET + 8, OFFSET + 38),
    "entropy_trix": (62, 33, 65, OFFSET + 38),
}

_TOLERANCE: float = 1e-3


def _synthetic_raw(n: int = 1400, seed: int = 7) -> dict[str, np.ndarray]:
    """构造与真实盘面量级接近的单品种 OHLCV（2D ``[1, T]``）。"""
    rng = np.random.default_rng(seed)
    close = (3000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n)))).astype(np.float32)
    high = (close * (1.0 + np.abs(rng.normal(0.0, 0.0006, n)))).astype(np.float32)
    low = (close * (1.0 - np.abs(rng.normal(0.0, 0.0006, n)))).astype(np.float32)
    open_ = np.concatenate([[close[0]], close[:-1]]).astype(np.float32)
    volume = (1000.0 + rng.integers(0, 500, n)).astype(np.float32)
    return {
        "open": open_[None, :],
        "high": high[None, :],
        "low": low[None, :],
        "close": close[None, :],
        "volume": volume[None, :],
    }


@pytest.fixture
def algoforge_stubs() -> Any:
    """注入 AlgoForge SDK 占位模块，使导出文件可被 import。"""
    created = install_stub_modules(PLATFORM_SPEC)
    try:
        yield
    finally:
        uninstall_stub_modules(created)


def _spec(tokens: tuple[int, ...]) -> StrategySpec:
    return StrategySpec(
        name="fidelity_probe",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


def _export_module(
    tokens: tuple[int, ...], tmp_path: Path, name: str
) -> tuple[ModuleType, Path]:
    """导出并 import 生成文件，返回 (模块, 路径)。"""
    port = AlgoforgePort(magic="661801", date="20260910")
    result = port.compile(_spec(tokens))
    assert result.errors == (), f"{name} 导出未通过静态检查：{result.errors}"
    target = tmp_path / f"{name}_{result.filename}"
    target.write_text(result.source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(f"exported_{name}", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, target


@pytest.mark.parametrize("name", sorted(FORMULAS), ids=sorted(FORMULAS))
def test_numerical_fidelity_vs_native_vm(
    name: str, tmp_path: Path, algoforge_stubs: None
) -> None:
    tokens = FORMULAS[name]
    module, _path = _export_module(tokens, tmp_path, name)

    raw = _synthetic_raw()
    features = compute_features(raw)
    native = StackVM().execute(list(tokens), features)
    assert native is not None, f"{name}: 原生 StackVM 返回 None"

    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    got = np.asarray(
        module.compute_factor(flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"])
    )
    expected = np.asarray(native)[0]
    assert got.shape == expected.shape, f"{name}: 形状不一致 {got.shape} vs {expected.shape}"

    max_err = float(np.max(np.abs(got - expected)))
    assert max_err < _TOLERANCE, f"{name}: max_abs_err={max_err:.3e} 超过 {_TOLERANCE}"


def test_fidelity_is_exact_for_am_best(tmp_path: Path, algoforge_stubs: None) -> None:
    """AM 最优公式要求逐位一致（内核源码同源，实测误差为 0）。"""
    tokens = FORMULAS["am_best"]
    module, _path = _export_module(tokens, tmp_path, "am_best_exact")
    raw = _synthetic_raw(n=1600, seed=11)
    native = StackVM().execute(list(tokens), compute_features(raw))
    assert native is not None
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    got = module.compute_factor(
        flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"]
    )
    assert float(np.max(np.abs(got - np.asarray(native)[0]))) == 0.0


def test_warmup_bars_are_zero(tmp_path: Path, algoforge_stubs: None) -> None:
    """预热期（前 roll_window-1 根）必须输出 0 —— 因子中性，不出信号。"""
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "warmup")
    raw = _synthetic_raw(n=1200, seed=3)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    got = module.compute_factor(
        flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"]
    )
    assert np.all(got[:498] == 0.0)
    assert got[499:].any()


def test_1d_and_2d_inputs_agree(tmp_path: Path, algoforge_stubs: None) -> None:
    module, _path = _export_module(FORMULAS["add_binary"], tmp_path, "shapes")
    raw = _synthetic_raw(n=900, seed=5)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    as_1d = module.compute_factor(
        flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"]
    )
    as_2d = module.compute_factor(
        raw["open"], raw["high"], raw["low"], raw["close"], raw["volume"]
    )
    assert np.allclose(as_1d, as_2d, atol=1e-6)


def test_generate_signal_uses_only_closed_bars(tmp_path: Path, algoforge_stubs: None) -> None:
    """bar1 语义：剔除未收盘 K 线后，信号不应因「多一根未收盘 K 线」而改变。"""
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "signal")
    raw = _synthetic_raw(n=1200, seed=13)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    candles = [
        {
            "open": float(flat["open"][i]),
            "high": float(flat["high"][i]),
            "low": float(flat["low"][i]),
            "close": float(flat["close"][i]),
            "volume": float(flat["volume"][i]),
        }
        for i in range(len(flat["close"]))
    ]
    strategy = module.FidelityProbeStrategy()
    # 平台契约：on_tick 先 refresh_data() 灌 self.candles，再无参调用 generate_signal()
    strategy.candles = candles
    base = strategy.generate_signal()
    # 末位换成价格离谱的「未收盘」K 线：已收盘集合不变 → 信号必须一字不差
    garbage = {
        "open": float(flat["open"][-1]) * 5.0,
        "high": float(flat["high"][-1]) * 5.0,
        "low": float(flat["low"][-1]) * 0.2,
        "close": float(flat["close"][-1]) * 5.0,
        "volume": float(flat["volume"][-1]) * 100.0,
    }
    strategy.candles = candles[:-1] + [garbage]
    with_forming_bar = strategy.generate_signal()
    assert base is not None
    assert with_forming_bar == base


def test_generate_signal_returns_none_when_neutral(tmp_path: Path, algoforge_stubs: None) -> None:
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "neutral")
    strategy = module.FidelityProbeStrategy()
    strategy.NEUTRAL_BAND = 0.999  # 中性带极大 → 任何仓位都被视为中性
    raw = _synthetic_raw(n=1200, seed=17)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    candles = [
        {
            "open": float(flat["open"][i]),
            "high": float(flat["high"][i]),
            "low": float(flat["low"][i]),
            "close": float(flat["close"][i]),
            "volume": float(flat["volume"][i]),
        }
        for i in range(len(flat["close"]))
    ]
    strategy.candles = candles
    assert strategy.generate_signal() is None


def test_generate_signal_none_on_insufficient_history(tmp_path: Path, algoforge_stubs: None) -> None:
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "short")
    strategy = module.FidelityProbeStrategy()
    raw = _synthetic_raw(n=1200, seed=19)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    candles = [
        {
            "open": float(flat["open"][i]),
            "high": float(flat["high"][i]),
            "low": float(flat["low"][i]),
            "close": float(flat["close"][i]),
            "volume": float(flat["volume"][i]),
        }
        for i in range(100)
    ]
    strategy.candles = candles
    assert strategy.generate_signal() is None


def test_generate_signal_matches_algoforge_contract(
    tmp_path: Path, algoforge_stubs: None
) -> None:
    """P0 回归：``generate_signal`` 必须**无参**，且返回 OrderType 六元组。

    AlgoForge ``BaseStrategy.on_tick`` 先 ``refresh_data()`` 灌 ``self.candles``，
    再**无参**调用 ``generate_signal()``，随后读 ``signal.value``。以前模板生成的是
    ``generate_signal(self, candles) -> dict``，平台侧一调用就 TypeError，
    导出策略根本加载不起来。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "contract")
    strategy = module.FidelityProbeStrategy()

    # ① 无参：除 self 外不得有任何形参
    import inspect

    params = [
        name
        for name in inspect.signature(strategy.generate_signal).parameters
        if name != "self"
    ]
    assert params == [], f"generate_signal 不应有额外形参，实际 {params}"

    # ② 灌 K 线后无参调用，返回六元组且首元是 OrderType 枚举（有 .value）
    raw = _synthetic_raw(n=1200, seed=29)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    strategy.candles = [
        {
            "open": float(flat["open"][i]),
            "high": float(flat["high"][i]),
            "low": float(flat["low"][i]),
            "close": float(flat["close"][i]),
            "volume": float(flat["volume"][i]),
        }
        for i in range(len(flat["close"]))
    ]
    result = strategy.generate_signal()
    assert result is not None
    assert isinstance(result, tuple) and len(result) == 6, f"应返回六元组，实际 {result!r}"
    signal, score_long, score_short, factors_long, factors_short, indicators = result
    assert signal.value in {"BUY", "SELL"}, f"首元必须是 OrderType，实际 {signal!r}"
    assert isinstance(score_long, int) and isinstance(score_short, int)
    assert isinstance(factors_long, list) and isinstance(factors_short, list)
    assert isinstance(indicators, dict)
    # 多头信号时分数记在多头侧，空头侧为 0（与 AlgoForge 既有策略一致）
    if signal.value == "BUY":
        assert score_long > 0 and score_short == 0 and factors_long
    else:
        assert score_short > 0 and score_long == 0 and factors_short


def test_param_space_matches_class_attributes(tmp_path: Path, algoforge_stubs: None) -> None:
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "params")
    strategy = module.FidelityProbeStrategy()
    for entry in module.PARAM_SPACE:
        assert hasattr(strategy, entry["name"]), f"类缺少属性 {entry['name']}"
        assert float(getattr(strategy, entry["name"])) == float(entry["default"])
