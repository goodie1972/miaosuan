"""M16：repaint lint 规则（AF001 / AF002 / AF003 / AF004）。"""

from __future__ import annotations

from miaosuan.adapters.algoforge.lint import lint_source
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


def test_issues_sorted_by_severity_then_code() -> None:
    source = (
        "def f(candles):\n"
        "    import random\n"
        "    return candles[-1] + random.random()\n"
    )
    issues = lint_source(source)
    order = [i.severity.value for i in issues]
    assert order == sorted(order, key=lambda s: {"ERROR": 0, "WARNING": 1, "INFO": 2}[s])
