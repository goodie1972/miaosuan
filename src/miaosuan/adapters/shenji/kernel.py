# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""因子内核提取器（M15 支撑模块）—— 把公式用到的算子/特征源码搬进生成文件。

为什么需要它
------------
导出的 ``.py`` 要跑在 神机 环境里，那里**没有** ``miaosuan`` 包，因此不能
``from miaosuan.core.vm import StackVM``。但验收要求「导出因子与原生 VM 的
max abs err < 1e-3」。要同时满足「独立可运行」与「数值一致」，唯一可靠的办法是
**把真实计算源码原样搬过去**——而不是手写一份「等价」实现（手写必然漂移）。

做法
----
1. 用 ``ast`` 解析 :mod:`miaosuan.core.features` / :mod:`miaosuan.core.ops` 源文；
2. 按公式 token 定位根节点（模块级 ``def`` 或列表里的 ``lambda``）；
3. 沿 ``ast.Name`` 引用做**传递闭包**，把依赖的私有 helper 一并取出；
4. 统一加前缀重命名（``ft_`` / ``op_``）——两个模块都定义了 ``_EPS``、
   ``_ema_simple`` 等同名符号，不加前缀会互相覆盖；
5. 静态校验自由变量（:func:`_validate_blocks`）：若还有未定义符号，
   说明闭包没取全，**直接报错**而不是产出一个会 ``NameError`` 的文件。

于是生成文件在数值上与 core 逐字节同源，保真度由构造保证。
"""

from __future__ import annotations

import ast
import builtins
import inspect
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ...core import features as features_module
from ...core import ops as ops_module
from ...core.features import _FEATURE_DEFS
from ...core.ops import OPS_CONFIG
from ...core.vocab import FORMULA_VOCAB

__all__ = ["KernelExtractionError", "KernelPlan", "build_kernel_plan"]

#: 生成文件中由模板提供的符号（不算「未定义」）。
_ALLOWED_FREE_NAMES: frozenset[str] = frozenset(
    {
        "np",
        "Any",
        "Dict",
        "List",
        "Tuple",
        "Optional",
        "Sequence",
        "Iterable",
        "Callable",
        "cast",
        "math",  # 由生成文件的 import 头提供
    }
)


class KernelExtractionError(RuntimeError):
    """内核源码提取失败（依赖闭包不完整或无法定位根节点）。"""


@dataclass(frozen=True)
class KernelPlan:
    """一次内核提取的产物（可直接喂给模板渲染）。

    Attributes:
        blocks: 按依赖顺序排好的源码块（已重命名）。
        feature_entries: ``(token_id, token_name, func_name)``。
        op_entries: ``(token_id, op_name, arity, func_name)``。
        n_features: 用到的特征个数。
        n_ops: 用到的算子个数。
    """

    blocks: tuple[str, ...]
    feature_entries: tuple[tuple[int, str, str], ...]
    op_entries: tuple[tuple[int, str, int, str], ...]

    @property
    def n_features(self) -> int:
        """用到的特征个数。"""
        return len(self.feature_entries)

    @property
    def n_ops(self) -> int:
        """用到的算子个数。"""
        return len(self.op_entries)

    @property
    def op_names(self) -> tuple[str, ...]:
        """用到的算子名（按出现顺序去重）。"""
        seen: list[str] = []
        for _tok, name, _arity, _fn in self.op_entries:
            if name not in seen:
                seen.append(name)
        return tuple(seen)


class _ModuleIndex:
    """单个模块的符号索引：源码段 + 行列定位。"""

    def __init__(self, module: Any, prefix: str) -> None:
        """解析模块源码，建立符号表。

        Args:
            module: 已导入的模块对象。
            prefix: 重命名前缀（``ft`` 或 ``op``）。
        """
        self.prefix = prefix
        self.module = module
        self.source = inspect.getsource(module)
        self.tree = ast.parse(self.source)
        self.segments: dict[str, str] = {}
        self._lineno: dict[str, int] = {}
        self._by_pos: dict[tuple[int, int], ast.AST] = {}
        self._by_lineno: dict[int, list[ast.AST]] = {}

        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                segment = ast.get_source_segment(self.source, node)
                if segment:
                    self.segments[node.name] = segment
                    self._lineno[node.name] = node.lineno
            elif isinstance(node, ast.Assign):
                segment = ast.get_source_segment(self.source, node)
                for target in node.targets:
                    if isinstance(target, ast.Name) and segment:
                        self.segments[target.id] = segment
                        self._lineno[target.id] = node.lineno

        for walked in ast.walk(self.tree):
            lineno = getattr(walked, "lineno", None)
            if lineno is None:  # ast.Module 等无位置信息的节点
                continue
            self._by_pos.setdefault((lineno, getattr(walked, "col_offset", 0)), walked)
            self._by_lineno.setdefault(lineno, []).append(walked)

    def lineno(self, name: str) -> int:
        """符号首次出现的行号（用于稳定排序）。"""
        return self._lineno.get(name, 10**9)

    def rename(self, name: str) -> str:
        """给出符号在生成文件中的新名字。

        规则（**必须保持单射**）：不同原符号若映射到同名，生成文件里会出现
        两个同名的 ``def``，后者遮蔽前者，导出策略直接 ``TypeError`` 不可运行。

        * ``_c_xxx``（特征入口 wrapper）→ ``ft_c_xxx``：保留 ``c`` 标记，与纯计算版区分
        * ``_xxx``（纯计算版）→ ``ft_xxx``
        * 算子侧同理：``_op_xxx`` / ``lambda_xxx`` 原样保留

        ``features`` 模块里 ``_c_X`` 与 ``_X`` 同时存在（wrapper 内部调用纯计算版），
        两者都必须导出，因此前缀去重后必须保留 ``c`` 标记。
        """
        if name.startswith("_c_"):
            return f"{self.prefix}_c_{name[3:]}"
        stripped = name.lstrip("_")
        if self.prefix == "ft":
            return f"ft_{stripped}"
        if stripped.startswith("op_") or stripped.startswith("lambda_"):
            return stripped
        return f"op_{stripped}"

    def resolve(self, fn: Callable[..., Any], hint: str) -> tuple[str, str]:
        """定位可调用对象的根符号。

        Args:
            fn: 特征 compute 函数 / 算子 transform 函数。
            hint: 用于合成名字的提示（算子名或特征名）。

        Returns:
            ``(原符号名, 源码段)``。

        Raises:
            KernelExtractionError: 无法定位（如嵌套定义且无法取源）。
        """
        code = getattr(fn, "__code__", None)
        if code is None:
            return self._resolve_ufunc(fn, hint)
        candidates = self._by_lineno.get(code.co_firstlineno, [])
        node: ast.AST | None = None
        for candidate in candidates:
            if isinstance(candidate, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                node = candidate
                break
        if node is None and candidates:
            node = candidates[0]
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            segment = self.segments.get(node.name) or ast.get_source_segment(self.source, node)
            if not segment:
                raise KernelExtractionError(f"无法取到 {node.name!r} 的源码")
            return node.name, segment
        if isinstance(node, ast.Lambda):
            segment = ast.get_source_segment(self.source, node)
            if not segment:
                raise KernelExtractionError(f"无法取到 {hint!r} 的 lambda 源码")
            symbol = f"_lambda_{hint.lower()}_{node.lineno}"
            self.segments[symbol] = f"{symbol} = {segment}"
            self._lineno[symbol] = node.lineno
            return symbol, self.segments[symbol]
        raise KernelExtractionError(
            f"无法定位 {hint!r} 的定义节点（行 {code.co_firstlineno}）"
        )

    def _resolve_ufunc(self, fn: Callable[..., Any], hint: str) -> tuple[str, str]:
        """处理没有 Python 源码的实现（numpy ufunc，如 ``ABS`` → ``np.absolute``）。

        ufunc 直接指向 numpy 的 C 实现，天然跨平台一致，只需在生成文件里重新
        绑定到 ``np.<name>`` 即可。

        Raises:
            KernelExtractionError: 非 numpy 实现（无法在生成文件中重建）。
        """
        name = getattr(fn, "__name__", "")
        module = getattr(fn, "__module__", "") or ""
        if not name or module.split(".")[0] != "numpy":
            raise KernelExtractionError(
                f"算子 {hint!r} 的实现 {fn!r} 既无 Python 源码也非 numpy ufunc，无法导出"
            )
        symbol = f"_op_np_{hint.lower()}"
        source = f"{symbol} = np.{name}"
        self.segments[symbol] = source
        self._lineno[symbol] = 10**8
        return symbol, source

    def closure(self, roots: Sequence[str]) -> list[str]:
        """沿引用做传递闭包，返回按行号排序的符号列表。"""
        seen: set[str] = set()
        collected: list[str] = []
        queue: list[str] = list(roots)
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            segment = self.segments.get(name)
            if segment is None:
                continue
            seen.add(name)
            collected.append(name)
            for dep in sorted(_load_names(segment)):
                if dep in self.segments and dep not in seen:
                    queue.append(dep)
        return sorted(collected, key=self.lineno)


def _load_names(source: str) -> set[str]:
    """取出源码段中所有「读取」上下文的名字。"""
    tree = ast.parse(textwrap.dedent(source))
    return {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _bound_names(source: str) -> set[str]:
    """取出源码段中所有「绑定」的名字（参数、赋值目标、for 目标等）。"""
    tree = ast.parse(textwrap.dedent(source))
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) or isinstance(node, ast.ExceptHandler) and node.name:
            # ExceptHandler.name 可空，统一收敛成非空字符串再收集
            label = node.name or ""
            if label:
                bound.add(label)
    return bound


def _rename_block(segment: str, mapping: dict[str, str]) -> str:
    """按最长优先策略做词边界重命名（避免 ``_norm`` 误伤 ``_norm_window``）。"""
    ordered = sorted(mapping, key=len, reverse=True)

    def _substitute(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    pattern = re.compile(r"(" + "|".join(re.escape(n) for n in ordered) + r")")
    return pattern.sub(_substitute, segment)


def _assert_injective(kind: str, mapping: Mapping[str, str]) -> None:
    """断言重命名映射是单射（不同原符号不得映射到同名）。

    Args:
        kind: 类别标签（``feature`` / ``op``），用于报错定位。
        mapping: 原符号名 → 新符号名的映射。

    Raises:
        KernelExtractionError: 存在两个原符号映射到同一新名（会导致生成文件
            函数重复定义、后者遮蔽前者，导出策略不可运行）。
    """
    if len(set(mapping.values())) == len(mapping):
        return
    collisions: dict[str, list[str]] = {}
    for original, renamed in mapping.items():
        collisions.setdefault(renamed, []).append(original)
    clashes = {new: sorted(orig) for new, orig in collisions.items() if len(orig) > 1}
    raise KernelExtractionError(
        f"{kind} 符号重命名非单射，以下新名被多个原符号共用："
        + "; ".join(f"{new}={ks}" for new, ks in clashes.items())
    )


def _validate_blocks(blocks: Sequence[str]) -> None:
    """静态校验：生成块里不能出现未定义符号。

    Args:
        blocks: 已重命名的源码块。

    Raises:
        KernelExtractionError: 存在未定义符号（依赖闭包不完整）。
    """
    defined: set[str] = set()
    free: set[str] = set()
    for block in blocks:
        defined |= _bound_names(block)
        free |= _load_names(block)
    missing = free - defined - _ALLOWED_FREE_NAMES - set(dir(builtins))
    if missing:
        raise KernelExtractionError(
            "内核源码存在未定义符号（依赖闭包提取不完整）：" + ", ".join(sorted(missing))
        )


def _unwrap_op(fn: Callable[..., Any]) -> Callable[..., Any]:
    """剥掉 :func:`miaosuan.core.ops._with_shape_check` 的校验包装。

    包装只是形状校验，不参与数值计算，导出时无需保留（形状在栈式求值中天然一致）。
    """
    if getattr(fn, "__name__", "") == "_checked":
        for cell in fn.__closure__ or ():
            value = cell.cell_contents
            if callable(value):
                return cast("Callable[..., Any]", value)
    return fn


def build_kernel_plan(tokens: Sequence[int]) -> KernelPlan:
    """按公式 token 提取内核源码。

    Args:
        tokens: RPN token 序列（含特征与算子）。

    Returns:
        :class:`KernelPlan`。

    Raises:
        KernelExtractionError: 依赖闭包不完整或根节点无法定位。
        IndexError: token 越界。
    """
    offset = FORMULA_VOCAB.operator_offset
    token_names = FORMULA_VOCAB.token_names

    feature_index = _ModuleIndex(features_module, "ft")
    op_index = _ModuleIndex(ops_module, "op")

    feature_roots: list[str] = []
    op_roots: list[str] = []
    feature_entries: list[tuple[int, str, str]] = []
    op_entries: list[tuple[int, str, int, str]] = []

    for raw_token in tokens:
        token = int(raw_token)
        if token < offset:
            token_name = token_names[token]
            compute = _FEATURE_DEFS[token][2]
            symbol, _segment = feature_index.resolve(compute, token_name)
            feature_roots.append(symbol)
            feature_entries.append((token, token_name, feature_index.rename(symbol)))
        else:
            index = token - offset
            op_name, op_fn, arity = OPS_CONFIG[index]
            symbol, _segment = op_index.resolve(_unwrap_op(op_fn), op_name)
            op_roots.append(symbol)
            op_entries.append((token, op_name, int(arity), op_index.rename(symbol)))

    feature_symbols = feature_index.closure(feature_roots)
    op_symbols = op_index.closure(op_roots)

    feature_map = {name: feature_index.rename(name) for name in feature_symbols}
    op_map = {name: op_index.rename(name) for name in op_symbols}
    # 防御：若未来扩词表再次引入同构撞名（如新增与 wrapper 重名的纯计算函数），
    # 立刻在构建期报错，而不是产出一个会 NameError/TypeError 的生成文件。
    _assert_injective("feature", feature_map)
    _assert_injective("op", op_map)

    blocks: list[str] = []
    for name in feature_symbols:
        blocks.append(_rename_block(feature_index.segments[name], feature_map))
    for name in op_symbols:
        blocks.append(_rename_block(op_index.segments[name], op_map))

    _validate_blocks(blocks)
    return KernelPlan(
        blocks=tuple(blocks),
        feature_entries=tuple(feature_entries),
        op_entries=tuple(op_entries),
    )
