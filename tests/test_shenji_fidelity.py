# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M16：数值保真度回归 —— 导出因子 vs 原生 ``StackVM``（max abs err < 1e-3）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from miaosuan.adapters.shenji import ShenjiPort
from miaosuan.adapters.shenji.contract import PLATFORM_SPEC
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
def shenji_stubs() -> Any:
    """注入 妙算 SDK 占位模块，使导出文件可被 import。"""
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
    port = ShenjiPort(magic="661801", date="20260910")
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
    name: str, tmp_path: Path, shenji_stubs: None
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


def test_fidelity_is_exact_for_am_best(tmp_path: Path, shenji_stubs: None) -> None:
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


def test_warmup_bars_are_zero(tmp_path: Path, shenji_stubs: None) -> None:
    """预热期（前 roll_window-1 根）必须输出 0 —— 因子中性，不出信号。"""
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "warmup")
    raw = _synthetic_raw(n=1200, seed=3)
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    got = module.compute_factor(
        flat["open"], flat["high"], flat["low"], flat["close"], flat["volume"]
    )
    assert np.all(got[:498] == 0.0)
    assert got[499:].any()


def test_1d_and_2d_inputs_agree(tmp_path: Path, shenji_stubs: None) -> None:
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


def test_generate_signal_uses_only_closed_bars(tmp_path: Path, shenji_stubs: None) -> None:
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


def test_generate_signal_returns_none_when_neutral(tmp_path: Path, shenji_stubs: None) -> None:
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


def test_generate_signal_none_on_insufficient_history(tmp_path: Path, shenji_stubs: None) -> None:
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


def test_generate_signal_matches_shenji_contract(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """P0 回归：``generate_signal`` 必须**无参**，且返回 OrderType 六元组。

    妙算 ``BaseStrategy.on_tick`` 先 ``refresh_data()`` 灌 ``self.candles``，
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
    # 多头信号时分数记在多头侧，空头侧为 0（与 妙算 既有策略一致）
    if signal.value == "BUY":
        assert score_long > 0 and score_short == 0 and factors_long
    else:
        assert score_short > 0 and score_long == 0 and factors_short


def _exported_strategy_class(module: ModuleType) -> type:
    """取出生成文件里**本模块定义**的策略类。

    不能用「有 generate_signal 属性」直接筛：import 进来的 ``BaseStrategy`` 也带
    该方法，会选错对象（这是踩过的坑）。
    """
    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and getattr(obj, "__module__", None) == module.__name__
            and hasattr(obj, "generate_signal")
        ):
            return obj
    raise AssertionError("生成文件里找不到策略类")


def test_exported_class_is_concrete_not_abstract(tmp_path: Path, shenji_stubs: None) -> None:
    """P0 回归：导出类必须**可实例化**，不能残留未实现的抽象方法。

    背景：模板曾写成 ``class X(BaseStrategy, MT4BridgeBase)``，而 MT4BridgeBase 有
    9 个抽象方法（connect/disconnect/open_order/...）一个都没实现，导致导出类是
    抽象类，妙算 实例化时直接 TypeError，策略根本加载不起来。

    这里的桩**如实还原**了两个基类的抽象面（见 ``contract.stub_abstracts``），
    所以不需要依赖外部 妙算 仓库也能在 CI 里抓到这类问题。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "concrete")
    cls = _exported_strategy_class(module)

    # ① 需要实现的抽象方法**恰好**只有 generate_signal（证明没拖进桥接基类）
    required: set[str] = set()
    for base in cls.__mro__[1:]:
        required |= set(getattr(base, "__abstractmethods__", ()))
    assert required == {"generate_signal"}, f"必须实现的抽象方法异常：{sorted(required)}"

    # ② 全部实现 → 不再是抽象类
    assert set(getattr(cls, "__abstractmethods__", ())) == set(), (
        f"导出类仍是抽象类：{sorted(getattr(cls, '__abstractmethods__', ()))}"
    )

    # ③ 真的能实例化
    strategy = cls()
    assert isinstance(strategy, cls)

    # ④ 兜底：源码里不应再出现 MT4BridgeBase
    assert "MT4BridgeBase" not in _path.read_text(encoding="utf-8")


def test_class_name_attribute_matches_strategy_name(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """P0 回归：运行期的 ``cls.name`` 必须是策略名，而不是基类默认的 ``"base"``。

    源码里有 ``name = "x"`` 还不够，得确认运行期真的取得到 —— 引擎用
    ``scan_strategies().get(name)`` 查的是**类属性值**。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "nameattr")
    cls = _exported_strategy_class(module)
    assert cls.name == "fidelity_probe", f"cls.name={cls.name!r}，应为策略名"
    assert cls.name != "base", "name 落到了基类默认值，池里查不到"
    # 与 STRATEGY_NAME / 模块级常量同源，不能各写各的
    assert cls.name == module.STRATEGY_NAME


def test_exported_class_inherits_only_base_strategy(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """基类列表必须只有 BaseStrategy（与 妙算 现网 27 个策略一致）。"""
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "bases")
    cls = _exported_strategy_class(module)
    base_names = [base.__name__ for base in cls.__mro__[1:] if base.__name__ != "object"]
    assert base_names[0] == "BaseStrategy"
    assert "MT4BridgeBase" not in base_names


def test_param_space_matches_class_attributes(tmp_path: Path, shenji_stubs: None) -> None:
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "params")
    strategy = module.FidelityProbeStrategy()
    for entry in module.PARAM_SPACE:
        assert hasattr(strategy, entry["name"]), f"类缺少属性 {entry['name']}"
        assert float(getattr(strategy, entry["name"])) == float(entry["default"])


# ── get_dynamic_sl_tp 平台契约（ATR / 止损止盈接口统一）─────────────────────


def _candle_list(raw: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """把合成 OHLCV 转成平台 candle 形态（dict 列表）。"""
    flat = {key: np.asarray(value)[0] for key, value in raw.items()}
    n = len(flat["close"])
    return [
        {key: float(flat[key][i]) for key in ("open", "high", "low", "close", "volume")}
        for i in range(n)
    ]


def test_get_dynamic_sl_tp_callable_with_two_positional_args(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """引擎 ``main.py:1901/1953`` 只传 **2 个位置参数** → 必须能这样调。

    ``goodma`` 的第 3 参 ``atr_val`` 没有默认值，引擎一调用就 TypeError、被
    ``except`` 吞掉退化成固定点数——动态 SL/TP 实际从未生效。
    """
    import inspect

    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "sl2args")
    strategy = module.FidelityProbeStrategy()
    strategy.candles = _candle_list(_synthetic_raw(n=1200, seed=31))

    params = inspect.signature(strategy.get_dynamic_sl_tp).parameters
    positional = [
        name
        for name, p in params.items()
        if name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        name
        for name, p in params.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert required == ["direction", "entry_price"], f"必填位置参数异常：{required}"
    assert len(positional) >= 2

    entry = 4000.0
    out = strategy.get_dynamic_sl_tp(module.OrderType.BUY, entry)  # 恰好 2 位置参数
    assert isinstance(out, tuple) and len(out) == 2
    sl, tp = out
    # TP 默认关闭（TP_ATR_MULT=0 → tp=0.0，不是 None）；止损仍在正确一侧
    assert sl < entry, f"买入方向应 sl<entry，实际 {out}"
    assert tp == 0.0 and tp is not None, f"TP 关闭时应返回 0.0（非 None），实际 {out}"
    # 第 3/4 个参数（新式签名）也应当能传
    assert strategy.get_dynamic_sl_tp(module.OrderType.BUY, entry, 12.0, "entry")


def test_get_dynamic_sl_tp_sides_and_direction_normalization(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """买/卖 SL、TP 必须落在**正确一侧**；枚举与字符串必须同侧。

    平台传的是 ``OrderType.BUY`` 枚举，而 ``OrderType.BUY == "BUY"`` 是 ``False``。
    策略若写成 ``if direction == "BUY"``，BUY 会落进 SELL 分支、**止损挂到入场价
    上方**（实盘一开仓即被扫）。这里把两侧与两种写法都钉死。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "slside")
    strategy = module.FidelityProbeStrategy()
    strategy.candles = _candle_list(_synthetic_raw(n=1200, seed=37))
    entry = 4000.0

    buy = strategy.get_dynamic_sl_tp(module.OrderType.BUY, entry)
    sell = strategy.get_dynamic_sl_tp(module.OrderType.SELL, entry)
    # TP 默认关闭（tp=0.0），只钉 SL 侧；tp 两侧都必须是 0.0（非 None）
    assert buy[0] < entry, f"BUY 侧错误：{buy}"
    assert sell[0] > entry, f"SELL 侧错误：{sell}"
    assert buy[1] == 0.0 and buy[1] is not None, f"BUY tp 应为 0.0：{buy}"
    assert sell[1] == 0.0 and sell[1] is not None, f"SELL tp 应为 0.0：{sell}"
    # 枚举 ⇄ 字符串（引擎两处调用分别是枚举与字符串）结果必须一致
    assert strategy.get_dynamic_sl_tp("BUY", entry) == buy
    assert strategy.get_dynamic_sl_tp("SELL", entry) == sell


def test_get_dynamic_sl_tp_derives_from_real_atr_not_hardcoded(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """止损距离必须由**真实 ATR**（已收盘 K 线自算）派生，不是硬编码 15。"""
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "slatr")
    strategy = module.FidelityProbeStrategy()
    strategy.candles = _candle_list(_synthetic_raw(n=1200, seed=41))
    entry = 4000.0

    atr = strategy._current_atr()
    assert atr > 0.0, "自算 ATR 应 > 0"
    sl, _tp = strategy.get_dynamic_sl_tp(module.OrderType.BUY, entry)
    expected = entry - max(atr * strategy.SL_ATR_MULT, strategy.MIN_SL_POINTS)
    assert sl == pytest.approx(expected, rel=1e-12), f"止损未按真实 ATR 计算：{sl} vs {expected}"
    # 与「硬编码 ATR=15 + 兜底倍数 2」的结果不同，证明确实用了自算 ATR
    #（仅在两值本应不同时比较，避免 atr*3 恰好=30 的巧合误报）
    fallback = entry - 2.0 * 15.0
    if abs(expected - fallback) > 1e-9:
        assert sl != pytest.approx(fallback, rel=1e-9)


def test_get_dynamic_sl_tp_no_fixed_tp_when_multiplier_zero(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """``TP_ATR_MULT = 0``（**导出默认值**）→ 无固定止盈（``tp = 0.0`` 非 ``None``）。

    ``0`` 是 ``core.bridge.open_order(sl=0, tp=0)`` 的"未提供"默认语义；
    回测层没有 SL/TP，不设固定止盈最贴近回测口径（趋势跟踪，因子反向出场）。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "sltp0")
    strategy = module.FidelityProbeStrategy()
    strategy.candles = _candle_list(_synthetic_raw(n=1200, seed=43))
    assert strategy.TP_ATR_MULT == 0.0, "导出默认应为 TP_ATR_MULT=0（无固定止盈）"
    sl, tp = strategy.get_dynamic_sl_tp(module.OrderType.BUY, 4000.0)
    assert tp == 0.0 and tp is not None, f"tp 必须是 0.0（不是 None），实际 {tp!r}"
    assert sl < 4000.0
    sl_s, tp_s = strategy.get_dynamic_sl_tp(module.OrderType.SELL, 4000.0)
    assert tp_s == 0.0 and tp_s is not None
    assert sl_s > 4000.0


def test_get_dynamic_sl_tp_sl_is_three_atr(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """``SL_ATR_MULT = 3``（导出默认值）：``atr=10`` 时止损距离 = 30。"""
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "sl3atr")
    strategy = module.FidelityProbeStrategy()
    assert strategy.SL_ATR_MULT == 3.0, "导出默认应为 SL_ATR_MULT=3"

    sl, tp = strategy.get_dynamic_sl_tp(module.OrderType.BUY, 2000.0, 10.0)
    assert sl == pytest.approx(2000.0 - 30.0), f"BUY 止损距离应为 3×ATR=30：{sl}"
    assert tp == 0.0
    sl_s, tp_s = strategy.get_dynamic_sl_tp(module.OrderType.SELL, 2000.0, 10.0)
    assert sl_s == pytest.approx(2000.0 + 30.0), f"SELL 止损距离应为 3×ATR=30：{sl_s}"
    assert tp_s == 0.0


def test_get_dynamic_sl_tp_tp_floor_preserves_ratio(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """C1 回归：TP 启用时，止盈地板必须**独立**于止损地板（不能压成 1:1）。

    旧实现把 ``MIN_SL_POINTS`` 复用作止盈地板 —— ATR→0 时两侧地板相同，
    设计的 ``TP:SL = TP_ATR_MULT:SL_ATR_MULT`` 会被压成 1:1。
    导出默认 ``TP_ATR_MULT = 0``（无固定止盈），因此本用例在实例上**本地
    重新启用止盈**来验证地板与比例逻辑——这也是将来重启 TP 时应遵循的
    ``MIN_TP_POINTS = MIN_SL_POINTS × TP/SL 倍率`` 推导方式。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "sltpratio")
    strategy = module.FidelityProbeStrategy()
    entry = 4000.0

    # 本地启用 TP（不动导出默认值）：倍率与地板按比例重推导
    strategy.TP_ATR_MULT = 4.0
    strategy.MIN_TP_POINTS = strategy.MIN_SL_POINTS * 4.0 / strategy.SL_ATR_MULT

    # 传入极小 ATR 触发两侧地板（atr_val 优先级最高，无需 K 线）
    sl, tp = strategy.get_dynamic_sl_tp(module.OrderType.BUY, entry, 0.001)
    sl_dist = entry - sl
    tp_dist = tp - entry
    assert sl_dist == pytest.approx(strategy.MIN_SL_POINTS)
    assert tp_dist == pytest.approx(strategy.MIN_TP_POINTS)
    assert tp_dist > sl_dist, "止盈地板不得等于止损地板（否则盈亏比被压成 1:1）"
    # 比例断言（而非绝对值）：TP:SL 距离比 == 倍率比
    assert tp_dist / sl_dist == pytest.approx(
        strategy.TP_ATR_MULT / strategy.SL_ATR_MULT
    )


def test_get_dynamic_sl_tp_returns_none_on_invalid_entry(
    tmp_path: Path, shenji_stubs: None
) -> None:
    """入场价非法时返回 ``(None, None)``，**绝不返回 (0, 0)**。

    原因：``_execute_order`` 只判 ``if sl is None``，``0.0`` 会被当「有效值」传进
    ``open_order(sl=0)``——灾难级。返回 ``None`` 才能让两条路径都走自身兜底。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "slbad")
    strategy = module.FidelityProbeStrategy()
    strategy.candles = _candle_list(_synthetic_raw(n=1200, seed=47))
    assert strategy.get_dynamic_sl_tp(module.OrderType.BUY, 0.0) == (None, None)
    assert strategy.get_dynamic_sl_tp(module.OrderType.BUY, float("nan")) == (None, None)


def test_indicator_values_carries_real_atr(tmp_path: Path, shenji_stubs: None) -> None:
    """六元组里的 ``indicator_values['atr']`` 必须是**真实正数**。

    平台兜底 ``athlete.py:113`` 读的就是这个键（缺省才回退硬编码 15）；把真实 ATR
    放进去，即使平台走兜底也不会再用假值。
    """
    module, _path = _export_module(FORMULAS["am_best"], tmp_path, "slind")
    strategy = module.FidelityProbeStrategy()
    strategy.candles = _candle_list(_synthetic_raw(n=1200, seed=51))
    result = strategy.generate_signal()
    assert result is not None
    indicators = result[5]
    assert "atr" in indicators, "indicator_values 必须带 atr"
    atr = indicators["atr"]
    assert isinstance(atr, float) and atr > 0.0, f"atr 应为真实正数，实际 {atr!r}"
    assert atr == pytest.approx(strategy._current_atr(), abs=1e-6)
