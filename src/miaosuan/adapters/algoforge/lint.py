"""AlgoForge 生成文件的静态检查（M16）。

重点规则：

``AF001``（ERROR，repaint）
    禁止用**未收盘 K 线**做决策。任何 ``<K线对象>[-1]`` 形式都会被拦下
    （``candles[:-1]`` 是切片、不是 ``[-1]`` 取值，因此**不会**误报）。

``AF002``（ERROR，契约）
    必须声明 ``STRATEGY_MAGIC`` / ``STRATEGY_NAME`` / ``STRATEGY_VERSION`` /
    ``STRATEGY_CHANGELOG`` 四项，缺一项即失败。

``AF003``（WARNING）
    出现随机数 / 墙上时钟等不确定源，破坏可复现。

``AF004``（WARNING，R1）
    公式含 GATE 但死区保护未启用（``GATE_DEADBAND <= 0``）。

``AF006``（WARNING）
    ``STRATEGY_CHANGELOG`` 为空——生成文件应当自述来源。

检查器是**纯静态**的：只依赖 :mod:`ast`，不 import 被检查文件，因此可以在
导出流水线上无条件执行，也不会被被检查代码的副作用影响。
"""

from __future__ import annotations

import ast

from ..base import LintIssue, LintSeverity
from ...core.vocab import FORMULA_VOCAB, VocabVersionMismatchError
from .contract import GATE_TOKEN_NAME, REQUIRED_CONSTANTS

__all__ = ["CANDLE_NAMES", "has_errors", "lint_source"]

#: 被认定为「K 线 / 行情序列」的名字（小写比较）。对它们取 ``[-1]`` 即视为
#: 读取未收盘 K 线。
CANDLE_NAMES: frozenset[str] = frozenset(
    {
        "candles",
        "candle",
        "bars",
        "bar",
        "klines",
        "kline",
        "rates",
        "ohlc",
        "ohlcv",
        "data",
        "df",
        "frame",
        "close",
        "closes",
        "open",
        "opens",
        "high",
        "highs",
        "low",
        "lows",
        "volume",
        "volumes",
        "time",
        "times",
        "o",
        "h",
        "l",
        "c",
        "v",
    }
)

#: 不确定源（破坏可复现）。
_NONDETERMINISTIC_NAMES: frozenset[str] = frozenset(
    {"random", "rand", "randn", "randint", "uniform", "shuffle", "seed"}
)

_SEVERITY_ORDER: dict[str, int] = {
    LintSeverity.ERROR.value: 0,
    LintSeverity.WARNING.value: 1,
    LintSeverity.INFO.value: 2,
}


def lint_source(source: str) -> list[LintIssue]:
    """对生成文件源码做静态检查。

    Args:
        source: 生成文件的完整源码。

    Returns:
        问题列表（按「级别 → 规则号 → 行号」排序；无问题时为空列表）。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            LintIssue(
                code="AF000",
                severity=LintSeverity.ERROR,
                message=f"生成文件存在语法错误：{exc.msg}",
                line=exc.lineno,
            )
        ]

    issues: list[LintIssue] = []
    issues.extend(_check_required_constants(tree))
    issues.extend(_check_repaint(tree))
    issues.extend(_check_vocab_version(tree))
    issues.extend(_check_nondeterminism(tree))
    issues.extend(_check_gate_deadband(tree, source))
    issues.sort(key=lambda i: (_SEVERITY_ORDER[i.severity.value], i.code, i.line or 0))
    return issues


def has_errors(issues: list[LintIssue]) -> bool:
    """问题列表中是否含有 ERROR 级问题。"""
    return any(i.severity == LintSeverity.ERROR for i in issues)


def _target_names(node: ast.AST) -> set[str]:
    """取出下标/调用目标里出现的所有名字与属性名（小写）。"""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id.lower())
        elif isinstance(child, ast.Attribute):
            names.add(child.attr.lower())
    return names


def _is_negative_one(node: ast.AST) -> bool:
    """判断 AST 节点是否为 ``-1``。

    CPython 把 ``x[-1]`` 解析成 ``UnaryOp(USub, Constant(1))``（而非
    ``Constant(-1)``），而 ``x[:-1]`` 是 ``Slice`` —— 两种形态都必须区分清楚，
    否则 repaint 检查会漏报或误报。
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        return isinstance(operand, ast.Constant) and _as_number(operand.value) == 1
    if isinstance(node, ast.Constant):
        return _as_number(node.value) == -1
    return False


def _as_number(value: object) -> float | None:
    """把常量转成数字；非数字返回 ``None``。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _calls_len_of_candle(node: ast.AST) -> bool:
    """判断节点是否为 ``len(<K线名>)`` 形式的调用。"""
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "len":
        return False
    if len(node.args) != 1:
        return False
    arg = node.args[0]
    if isinstance(arg, ast.Name) and arg.id.lower() in CANDLE_NAMES:
        return True
    if isinstance(arg, ast.Attribute) and arg.attr.lower() in CANDLE_NAMES:
        return True
    return False


def _is_dynamic_last_index(slice_node: ast.AST) -> bool:
    """判断下标/切片是否最终指向『未收盘的最后一根』（repaint）。

    命中情形：
      * ``x[-1]`` / ``x[len(x)-1]`` —— 明确取最后一根未收盘 bar
      * ``x[-1:]`` —— 切片从最后一根开始（含未收盘），等价于取最后一根

    安全情形（不报）：
      * ``x[:-1]`` / ``x[-2:-1]`` —— ``upper=-1`` 是「排除最后一根」，只用已收盘数据
      * ``x[-2]`` / ``x[0]`` / ``x[:k]`` —— 不以最后一根未收盘为终点
    """
    if _is_negative_one(slice_node):
        return True
    if isinstance(slice_node, ast.BinOp) and isinstance(slice_node.op, ast.Sub):
        # len(candles) - 1 之类的动态下标（注意取 .value，否则拿到的是 ast 节点）
        right = slice_node.right
        if isinstance(right, ast.Constant) and _as_number(right.value) == 1 and _calls_len_of_candle(slice_node.left):
            return True
    if isinstance(slice_node, ast.Slice):
        # candles[-1:]：从最后一根开始（含未收盘）→ repaint
        if slice_node.lower is not None and _is_negative_one(slice_node.lower):
            return True
        # candles[:-1] / candles[-2:-1]：upper=-1 排除最后一根 → 安全，不报
    return False


def _check_repaint(tree: ast.Module) -> list[LintIssue]:
    """AF001：禁止读取未收盘 K 线（``xxx[-1]`` / ``xxx[-1:]`` / ``xxx[len(xxx)-1]``）。"""
    issues: list[LintIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not (names := _target_names(node.value)) & CANDLE_NAMES:
            continue
        if _is_dynamic_last_index(node.slice):
            issues.append(
                LintIssue(
                    code="AF001",
                    severity=LintSeverity.ERROR,
                    message=(
                        "读取未收盘 K 线（下标指向最后一根未收盘 bar）—— 存在未来函数"
                        "（repaint），实盘会偷看未来。决策只能使用已收盘 K 线，"
                        "例如 candles[:-1]。"
                    ),
                    line=node.lineno,
                )
            )
    return issues


def _const_str(node: ast.AST) -> str | None:
    """取出字符串常量值；非字符串常量返回 ``None``。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _check_vocab_version(tree: ast.Module) -> list[LintIssue]:
    """AF007：生成文件内嵌词表版本必须与当前内核词表一致（硬约束 H-1 冻结锁）。

    ``verify`` 命令读的是导出的 ``.py`` 文件本身（不带 spec），因此版本锁必须**落进
    文件**：生成文件里写出 ``VOCAB_VERSION`` 常量，lint 在此比对代码派生版本。
    """
    issues: list[LintIssue] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "VOCAB_VERSION"):
            continue
        declared = _const_str(node.value)
        if declared is None:
            issues.append(
                LintIssue(
                    code="AF007",
                    severity=LintSeverity.ERROR,
                    message="VOCAB_VERSION 缺失或未写成字符串常量",
                    line=node.lineno,
                )
            )
            continue
        try:
            FORMULA_VOCAB.verify(declared)
        except VocabVersionMismatchError as exc:
            issues.append(
                LintIssue(
                    code="AF007",
                    severity=LintSeverity.ERROR,
                    message=f"词表版本不匹配（H-1 冻结锁触发）：{exc}",
                    line=node.lineno,
                )
            )
    return issues


def _check_required_constants(tree: ast.Module) -> list[LintIssue]:
    """AF002 / AF006：策略身份常量必须齐全且非空。"""
    defined: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined[node.target.id] = node

    issues: list[LintIssue] = []
    for name in REQUIRED_CONSTANTS:
        if name not in defined:
            issues.append(
                LintIssue(
                    code="AF002",
                    severity=LintSeverity.ERROR,
                    message=f"缺少 AlgoForge 契约常量 {name}",
                    line=None,
                )
            )
    changelog = defined.get("STRATEGY_CHANGELOG")
    if changelog is not None and isinstance(changelog, ast.Assign):
        value = changelog.value
        empty = isinstance(value, ast.Tuple | ast.List) and len(value.elts) == 0
        if empty:
            issues.append(
                LintIssue(
                    code="AF006",
                    severity=LintSeverity.WARNING,
                    message="STRATEGY_CHANGELOG 为空，生成文件应自述来源",
                    line=changelog.lineno,
                )
            )
    return issues


def _check_nondeterminism(tree: ast.Module) -> list[LintIssue]:
    """AF003：不确定源（随机数 / 墙上时钟）。"""
    issues: list[LintIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.lower() in _NONDETERMINISTIC_NAMES:
            issues.append(
                LintIssue(
                    code="AF003",
                    severity=LintSeverity.WARNING,
                    message=f"出现不确定源 {node.id!r}，破坏可复现性",
                    line=node.lineno,
                )
            )
        elif isinstance(node, ast.Attribute) and node.attr.lower() in _NONDETERMINISTIC_NAMES:
            issues.append(
                LintIssue(
                    code="AF003",
                    severity=LintSeverity.WARNING,
                    message=f"出现不确定源 .{node.attr}()，破坏可复现性",
                    line=node.lineno,
                )
            )
    return issues


def _gate_token_id() -> int | None:
    """GATE 算子的 token id；词表中不存在 GATE 时返回 ``None``。"""
    try:
        index = FORMULA_VOCAB.operator_names.index(GATE_TOKEN_NAME)
    except ValueError:
        return None
    return FORMULA_VOCAB.operator_offset + index


def _parse_module_tokens(tree: ast.Module) -> tuple[int, ...] | None:
    """从模块级 ``_TOKENS = (...)`` 常量解析 token 序列；缺失返回 ``None``。"""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "_TOKENS"
                and isinstance(node.value, ast.Tuple)
            ):
                elements = []
                for element in node.value.elts:
                    if not (isinstance(element, ast.Constant) and isinstance(element.value, int)):
                        return None
                    elements.append(int(element.value))
                return tuple(elements)
    return None


def _check_gate_deadband(tree: ast.Module, source: str) -> list[LintIssue]:
    """AF004：公式确实使用 GATE 算子但死区保护未启用时给出警告（R1）。

    判定依据是 ``_TOKENS`` 中是否真的出现 GATE 的 token id——**不能**用
    ``"GATE" in source`` 之类的子串判断：生成文件的 ``generate_signal`` 恒含
    ``GATE_DEADBAND`` 字样（getattr 兜底），按子串判会对所有无 GATE 公式误报。
    """
    tokens = _parse_module_tokens(tree)
    gate_id = _gate_token_id()
    if tokens is None or gate_id is None or gate_id not in tokens:
        return []
    value: float | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "_GATE_DEADBAND"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
                and not isinstance(node.value.value, bool)
            ):
                value = float(node.value.value)
    if value is not None and value > 0.0:
        return []
    return [
        LintIssue(
            code="AF004",
            severity=LintSeverity.WARNING,
            message=(
                "公式含 GATE 算子且死区保护未启用（GATE_DEADBAND=0）："
                "条件穿越 0 时存在 O(1) 跳变，实盘可能因 1-ULP 抖动反复翻分支"
            ),
            line=None,
        )
    ]
