# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M15/M16：妙算 导出 —— 命名、magic、契约常量、门禁降级标注。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miaosuan.adapters.shenji import ShenjiPort, build_param_space, readable_formula
from miaosuan.adapters.shenji.contract import REQUIRED_CONSTANTS
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.ir.schema import (
    Evidence,
    FactorPayload,
    ParamPayload,
    Provenance,
    Semantics,
    StrategySpec,
)

AM_BEST: tuple[int, ...] = (33, 62, 3, 87, 72, 119, 73, 103)
SIMPLE: tuple[int, ...] = (0, 1, 65)  # RET, RET5, ADD（不含 GATE）


def _spec(
    tokens: tuple[int, ...] = AM_BEST,
    *,
    verdict: str = "DEPLOYABLE",
    reasons: tuple[str, ...] = (),
) -> StrategySpec:
    return StrategySpec(
        name="h1_xauusd_miaosuan",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(),
        evidence=Evidence(
            n_trials=3280,
            wf_folds=5,
            val_score=2.327,
            cost_sensitivity={"0.5": 2.51, "2.0": 1.95},
            deflated_sharpe=0.87,
            gate_verdict=verdict,
            gate_reasons=reasons,
        ),
        provenance=Provenance(
            git_sha="ce83c09",
            vocab_version=VOCAB_VERSION,
            data_fingerprint="deadbeefcafe0011",
            seed=42,
            market="FOREX_XAUUSD",
            budget="standard",
        ),
    )


def _port(**kwargs: object) -> ShenjiPort:
    defaults: dict[str, object] = {"magic": "661801", "date": "20260910"}
    defaults.update(kwargs)
    return ShenjiPort(**defaults)  # type: ignore[arg-type]


def test_filename_follows_shenji_convention() -> None:
    result = _port().compile(_spec())
    assert result.filename == "20260910_h1_xauusd_miaosuan_v1.py"


def test_magic_is_six_digits_and_66_prefixed() -> None:
    result = _port().compile(_spec())
    assert result.magic == 661801
    assert isinstance(result.magic, int)


def test_magic_is_rendered_as_bare_int_not_string() -> None:
    """STRATEGY_MAGIC 必须是**裸 int**，不能带引号。

    平台侧是 ``magic: int``（core/bridge.py:40）；写成字符串会让
    ``p.magic == magic`` 永不相等 → 接管不到旧仓、产生孤儿单。
    """
    source = _port().compile(_spec()).source
    assert "STRATEGY_MAGIC = 661801" in source
    assert 'STRATEGY_MAGIC = "' not in source
    assert "STRATEGY_MAGIC = '" not in source


def test_strategy_declares_name_equal_to_pool_key() -> None:
    """P0 回归：策略类必须声明 ``name``，且等于策略名（STRATEGY_POOL 的 key）。

    依据（已核 妙算 源码，非转述）：
      docs/strategy_dev_guide.md:28   ``name = "my_strategy"  # settings.STRATEGY_POOL 的 key``
      engine_standalone/main.py:106   ``cls = scan_strategies().get(name)``
      engine_standalone/main.py:529   同上；查不到就 ``Unknown strategy, skip``
      strategies/scanner.py:94-96     字典 key 取自 ``getattr(cls, "name", None)``
      strategies/base.py:20           基类默认 ``name = "base"``

    不声明 name 时签名 / 继承 / magic 全对也不加载 —— 基类默认值顶不上池 key。
    """
    source = _port().compile(_spec()).source
    assert '    name = "h1_xauusd_miaosuan"' in source, (
        "策略类必须声明 name 类属性，且等于 STRATEGY_POOL 的 key"
    )


def test_strategy_name_is_not_the_base_default() -> None:
    """``name`` 不能是基类默认的 ``"base"``（那样池里查不到，等于没写）。"""
    source = _port().compile(_spec()).source
    assert 'name = "base"' not in source


def test_required_constants_present() -> None:
    source = _port().compile(_spec()).source
    for name in REQUIRED_CONSTANTS:
        assert f"{name} = " in source


def test_changelog_is_non_empty_and_traceable() -> None:
    source = _port().compile(_spec()).source
    assert "STRATEGY_CHANGELOG = (" in source
    assert "ce83c09" in source
    assert VOCAB_VERSION in source


def test_export_has_zero_lint_errors() -> None:
    result = _port().compile(_spec())
    assert result.errors == ()
    assert result.ok


def test_deployable_spec_marks_deployable_true() -> None:
    source = _port().compile(_spec(verdict="DEPLOYABLE")).source
    assert "DEPLOYABLE = True" in source
    assert "RESEARCH_ONLY" not in source.split('"""')[2][:2000]


def test_gate_failure_forces_research_only_header() -> None:
    result = _port().compile(
        _spec(verdict="RESEARCH_ONLY", reasons=("WF 胜率 0.40 < 0.60", "DSR 0.62 < 0.95"))
    )
    source = result.source
    assert "DEPLOYABLE = False" in source
    assert "RESEARCH_ONLY" in source
    assert "WF 胜率 0.40 < 0.60" in source
    assert "DSR 0.62 < 0.95" in source
    assert result.errors == ()


def test_gate_formula_gets_deadband_param() -> None:
    result = _port().compile(_spec(AM_BEST))
    names = [p.name for p in result.param_space]
    assert "GATE_DEADBAND" in names
    assert "_gate_protected" in result.source
    assert "GATE_DEADBAND = 0.0" in result.source


def test_non_gate_formula_has_no_deadband_param() -> None:
    result = _port().compile(_spec(SIMPLE))
    names = [p.name for p in result.param_space]
    assert names == ["NEUTRAL_BAND"]
    assert "_gate_protected" not in result.source


def test_gate_deadband_is_configurable() -> None:
    result = _port(gate_deadband=0.05).compile(_spec(AM_BEST))
    assert "_GATE_DEADBAND = 0.05" in result.source
    assert "GATE_DEADBAND = 0.05" in result.source
    deadband = next(p for p in result.param_space if p.name == "GATE_DEADBAND")
    assert deadband.default == 0.05


def test_params_are_class_attributes_not_inline_literals() -> None:
    source = _port().compile(_spec()).source
    assert "NEUTRAL_BAND = 0.05" in source
    assert "PARAM_SPACE = (" in source
    assert "self.NEUTRAL_BAND" in source


def test_warmup_and_neutral_band_documented() -> None:
    source = _port().compile(_spec()).source
    assert "WARMUP_BARS = 499" in source
    assert "ROLL_WINDOW = 500" in source


def test_bar1_semantics_documented_and_implemented() -> None:
    source = _port().compile(_spec()).source
    assert "candles[:-1]" in source
    assert "不重绘" in source or "未来函数" in source


def test_exports_are_deterministic() -> None:
    first = _port().compile(_spec())
    second = _port().compile(_spec())
    assert first.source == second.source
    assert first.filename == second.filename


def test_param_payload_is_rejected() -> None:
    spec = StrategySpec(name="x", payload=ParamPayload(template_id="tpl"))
    with pytest.raises(ValueError, match="仅支持 FactorPayload"):
        _port().compile(spec)


def test_empty_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="tokens 为空"):
        _port().compile(_spec(()))


def test_vocab_mismatch_rejected() -> None:
    spec = _spec()
    mismatched = StrategySpec(
        name=spec.name,
        payload=spec.payload,
        semantics=spec.semantics,
        evidence=spec.evidence,
        provenance=Provenance(vocab_version="v_other"),
    )
    with pytest.raises(ValueError, match="词表版本不一致"):
        _port().compile(mismatched)


def test_readable_formula() -> None:
    assert readable_formula(AM_BEST) == (
        "TRIX_15 → RET_ENTROPY_20 → MA_DIFF → MOMENTUM_5 → GATE → SQRT → JUMP → SCALE"
    )


def test_build_param_space_defaults() -> None:
    space = build_param_space(neutral_band=0.05, gate_deadband=0.0, has_gate=False)
    assert [p.name for p in space] == ["NEUTRAL_BAND"]
    assert space[0].low == 0.0 and space[0].high == 0.5


def test_exported_source_compiles(tmp_path: Path) -> None:
    result = _port().compile(_spec())
    target = tmp_path / result.filename
    target.write_text(result.source, encoding="utf-8")
    compile(target.read_text(encoding="utf-8"), str(target), "exec")


def test_export_declares_dynamic_sl_tp_and_risk_params() -> None:
    """导出文件必须实现动态 SL/TP，且风控参数是可调类属性（非内联字面量）。

    用户拍板：SL=3×ATR、无固定止盈（``TP_ATR_MULT=0``，对齐回测口径）。
    ``MIN_TP_POINTS`` 随推导公式变为 0.0（仅 ``TP_ATR_MULT>0`` 时生效）。
    """
    source = _port().compile(_spec(AM_BEST)).source
    assert "def get_dynamic_sl_tp(" in source
    assert "SL_ATR_MULT = 3.0" in source
    assert "TP_ATR_MULT = 0.0" in source
    assert "MIN_SL_POINTS = 3.0" in source
    assert "MIN_TP_POINTS = 0.0" in source
    assert "ATR_PERIOD = 14" in source
    assert "_ATR_PERIOD = 14" in source


def test_min_tp_points_is_independent_and_scales_with_ratio() -> None:
    """C1：止盈地板必须**独立**于止损地板，且按 TP:SL 倍率等比缩放。

    导出默认 ``TP_ATR_MULT=0``（无固定止盈），推导值 ``MIN_TP_POINTS=0.0``
    不参与计算（``take_dist`` 恒为 0）；这里只断言**比例关系**成立——
    将来重新启用止盈时按该公式重推导即可保住盈亏比。
    """
    from miaosuan.adapters.shenji.generator import (
        MIN_SL_POINTS,
        MIN_TP_POINTS,
        SL_ATR_MULT,
        TP_ATR_MULT,
    )

    assert TP_ATR_MULT == 0.0  # TP 默认关闭
    # 比例断言（非绝对值）：MIN_TP_POINTS 严格按 MIN_SL × TP/SL 倍率推导
    assert pytest.approx(MIN_SL_POINTS * TP_ATR_MULT / SL_ATR_MULT) == MIN_TP_POINTS
    source = _port().compile(_spec(AM_BEST)).source
    assert f"MIN_TP_POINTS = {MIN_TP_POINTS}" in source
    assert "float(self.MIN_TP_POINTS)" in source  # 止盈地板用的是 TP 常量，不是 SL
    # TP 关闭时地板不参与计算：take_dist 恒为 0
    assert "if tp_mult > 0.0 else 0.0" in source


def test_export_sl_tp_extra_params_have_defaults() -> None:
    """第 3/4 个参数必须有默认值——引擎 ``main.py:1901/1953`` 只传 2 个位置参数。"""
    source = _port().compile(_spec(AM_BEST)).source
    assert "atr_val: float | None = None" in source
    assert 'position_type: str = "entry"' in source


def test_export_indicator_values_includes_atr() -> None:
    """``indicator_values`` 必须带真实 ``atr``（平台兜底读的就是这个键）。"""
    source = _port().compile(_spec(AM_BEST)).source
    assert '"atr"' in source
    assert "_atr_from_arrays" in source


def test_non_gate_formula_also_gets_risk_contract() -> None:
    """风控与 GATE 无关：不含 GATE 的公式同样要带 SL/TP 与真实 ATR。"""
    source = _port().compile(_spec(SIMPLE)).source
    assert "def get_dynamic_sl_tp(" in source
    assert "SL_ATR_MULT = 3.0" in source
    assert "_gate_protected" not in source
