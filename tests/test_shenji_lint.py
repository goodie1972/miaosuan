# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M16：repaint / 契约 lint 规则（AF001 ~ AF007）。"""

from __future__ import annotations

from miaosuan.adapters.shenji.lint import lint_source
from miaosuan.adapters.base import LintSeverity


def _codes(source: str) -> list[str]:
    return [issue.code for issue in lint_source(source)]


def _only(source: str, code: str) -> list[str]:
    return [i.code for i in lint_source(source) if i.code == code]


def test_forming_bar_index_is_error() -> None:
    source = "def f(candles):\n    return candles[-1]\n"
    issues = _only(source, "AF001")
    assert issues, "读取未收盘 K 线必须被拦下"
    assert lint_source(source)[0].severity == LintSeverity.ERROR


def test_slice_to_drop_forming_bar_is_allowed() -> None:
    source = "def f(candles):\n    return candles[:-1]\n"
    assert _only(source, "AF001") == []


def test_iloc_minus_one_is_error() -> None:
    assert _only("def f(df):\n    return df.iloc[-1]\n", "AF001")


def test_close_minus_one_is_error() -> None:
    assert _only("def f(close):\n    return close[-1]\n", "AF001")


def test_time_minus_one_is_error() -> None:
    assert _only("def f(time):\n    return time[-1]\n", "AF001")


def test_unrelated_minus_one_is_allowed() -> None:
    assert _only("def f(factor):\n    return factor[-1]\n", "AF001") == []


def test_missing_strategy_constants() -> None:
    codes = _codes("DEPLOYABLE = True\n")
    assert codes.count("AF002") == 4


def test_all_constants_present() -> None:
    source = (
        'STRATEGY_MAGIC = "661801"\n'
        'STRATEGY_NAME = "x"\n'
        'STRATEGY_VERSION = "1"\n'
        "STRATEGY_CHANGELOG = (\"v1 init\",)\n"
    )
    assert "AF002" not in _codes(source)


def test_empty_changelog_is_warning() -> None:
    source = (
        'STRATEGY_MAGIC = "661801"\n'
        'STRATEGY_NAME = "x"\n'
        'STRATEGY_VERSION = "1"\n'
        "STRATEGY_CHANGELOG = ()\n"
    )
    assert _only(source, "AF006")


def test_nondeterminism_is_warning() -> None:
    assert _only("import random\n\ndef f():\n    return random.random()\n", "AF003")


def test_syntax_error_is_error() -> None:
    issues = lint_source("def f(:\n")
    assert issues and issues[0].code == "AF000"
    assert issues[0].severity == LintSeverity.ERROR


def test_gate_without_deadband_warns() -> None:
    # AF004 按 _TOKENS 中是否真含 GATE token（72）判定，样本须与真实导出形态一致
    source = (
        "_TOKENS = (33, 62, 72)\n"
        "_GATE_DEADBAND = 0.0\n"
        'STRATEGY_MAGIC = "1"\nSTRATEGY_NAME = "n"\n'
        'STRATEGY_VERSION = "1"\nSTRATEGY_CHANGELOG = ("v1",)\n'
    )
    assert _only(source, "AF004")


def test_gate_with_deadband_is_quiet() -> None:
    source = (
        "_TOKENS = (33, 62, 72)\n"
        "_GATE_DEADBAND = 0.05\n"
        'STRATEGY_MAGIC = "1"\nSTRATEGY_NAME = "n"\n'
        'STRATEGY_VERSION = "1"\nSTRATEGY_CHANGELOG = ("v1",)\n'
    )
    assert _only(source, "AF004") == []


def test_af004_ignores_formulas_without_gate_token() -> None:
    """无 GATE token 的公式即使文本里含 GATE_DEADBAND 也不得误报（R2-P2）。"""
    source = (
        "_TOKENS = (0, 1, 65)\n"
        "GATE_DEADBAND = 0.0  # generate_signal 里的 getattr 兜底字样\n"
        'STRATEGY_MAGIC = "1"\nSTRATEGY_NAME = "n"\n'
        'STRATEGY_VERSION = "1"\nSTRATEGY_CHANGELOG = ("v1",)\n'
    )
    assert _only(source, "AF004") == []


def test_af005_flags_generate_signal_with_candles_arg() -> None:
    """P0 回归：``generate_signal(self, candles)`` 必须被拦成 ERROR。

    妙算 基类 ``on_tick`` 以**无参**方式调用它，带形参会在平台侧
    直接 TypeError，导出策略加载不起来。
    """
    source = (
        "class S:\n"
        "    def generate_signal(self, candles):\n"
        "        return candles[:-1]\n"
    )
    issues = _only(source, "AF005")
    assert issues, "带 candles 形参的 generate_signal 必须报错"
    assert lint_source(source)[0].severity == LintSeverity.ERROR


def test_af005_also_catches_keyword_and_varargs() -> None:
    """kw-only / *args / **kwargs 同样违反契约（平台不会传任何参数）。"""
    assert _only("class S:\n    def generate_signal(self, *, candles):\n        pass\n", "AF005")
    assert _only("class S:\n    def generate_signal(self, **kw):\n        pass\n", "AF005")
    assert _only("class S:\n    def generate_signal(self, *args):\n        pass\n", "AF005")


def test_af005_accepts_no_arg_method() -> None:
    """无参写法（K 线从 self.candles 读）不应报错。"""
    source = (
        "class S:\n"
        "    def generate_signal(self):\n"
        "        return self.candles[:-1]\n"
    )
    assert _only(source, "AF005") == []
    # 顺带确认 bar1 语义没被 AF001 误伤
    assert _only(source, "AF001") == []


def test_af005_ignores_module_level_function() -> None:
    """模块级同名函数不是策略入口，不应被这条规则拦下（避免误伤辅助函数）。"""
    assert _only("def generate_signal(candles):\n    return candles[:-1]\n", "AF005") == []


def test_af008_flags_missing_default_after_entry_price() -> None:
    """P0 回归：``get_dynamic_sl_tp`` 第 3/4 个参数没有默认值时，引擎 2 参调用会
    TypeError、动态 SL/TP 静默失效（现网 ``goodma`` 即此病）。"""
    source = (
        "class S:\n"
        "    def get_dynamic_sl_tp(self, direction, entry_price, atr_val, position_type='entry'):\n"
        "        return 0, 0\n"
    )
    issues = _only(source, "AF008")
    assert issues, "多出的必填参数必须被拦下"
    assert lint_source(source)[0].severity == LintSeverity.ERROR


def test_af008_flags_swapped_or_too_few_params() -> None:
    """参数个数不足 / 顺序颠倒（引擎按位置传 direction, entry_price）都要报。"""
    assert _only(
        "class S:\n    def get_dynamic_sl_tp(self, entry_price, direction):\n        pass\n", "AF008"
    )
    assert _only(
        "class S:\n    def get_dynamic_sl_tp(self, direction):\n        pass\n", "AF008"
    )
    # kw-only 无默认值同样传不进去
    assert _only(
        "class S:\n    def get_dynamic_sl_tp(self, direction, entry_price, *, atr):\n        pass\n",
        "AF008",
    )


def test_af008_accepts_engine_compatible_signatures() -> None:
    ok_bare = "class S:\n    def get_dynamic_sl_tp(self, direction, entry_price):\n        return 0, 0\n"
    ok_full = (
        "class S:\n"
        "    def get_dynamic_sl_tp(self, direction, entry_price, atr_val=None, position_type='entry'):\n"
        "        return 0, 0\n"
    )
    assert _only(ok_bare, "AF008") == []
    assert _only(ok_full, "AF008") == []


def test_af008_ignores_class_without_the_method() -> None:
    """方法缺失不算违规：引擎会走自身兜底，且并非所有策略都必须实现动态 SL/TP。"""
    assert _only("class S:\n    def generate_signal(self):\n        return None\n", "AF008") == []


def test_issues_sorted_by_severity_then_code() -> None:
    source = (
        "def f(candles):\n"
        "    import random\n"
        "    return candles[-1] + random.random()\n"
    )
    issues = lint_source(source)
    order = [i.severity.value for i in issues]
    assert order == sorted(order, key=lambda s: {"ERROR": 0, "WARNING": 1, "INFO": 2}[s])
