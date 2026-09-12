"""平台无关的导出抽象（M16.5）。

定义四个稳定概念，供所有目标平台复用：

* :class:`ParamSpace` —— 单个可调参数的取值空间（供未来 ``tune`` 消费）；
* :class:`PlatformSpec` —— 目标平台的静态描述（文件后缀、导入头、占位模块等）；
* :class:`TargetPort` —— 四件套接口 ``extract / render / lint / compile``；
* :class:`LintIssue` / :class:`ExportResult` —— 静态检查与导出结果。

依赖方向（架构约束）：

* 本模块**只** import :mod:`miaosuan.ir`；
* ``adapters`` **不得** import ``cli`` / ``tune``；``tune`` **不得** import ``adapters``；
* 真实文件 IO **只**发生在 :mod:`miaosuan.cli`，:meth:`TargetPort.compile` 只回传源码。
"""

from __future__ import annotations

import sys
import types
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

from ..ir.schema import StrategySpec

__all__ = [
    "ExportResult",
    "LintIssue",
    "LintSeverity",
    "ParamSpace",
    "PlatformSpec",
    "TargetPort",
    "install_stub_modules",
    "uninstall_stub_modules",
]


class LintSeverity(StrEnum):
    """静态检查问题的严重级别。"""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class LintIssue:
    """一条静态检查问题。

    Attributes:
        code: 稳定规则号（如 ``AF001``），便于测试与门禁断言。
        severity: 严重级别。
        message: 人类可读说明。
        line: 触发行号（1-based）；静态无法定位时为 ``None``。
    """

    code: str
    severity: LintSeverity
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典（键顺序固定，便于快照比对）。"""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "line": self.line,
        }


@dataclass(frozen=True)
class ParamSpace:
    """单个可调参数的取值空间（供 ``tune`` 消费）。

    Attributes:
        name: 参数名，必须与生成代码中的类属性名**逐字一致**。
        default: 默认值；生成代码以它为类属性初值。
        kind: ``float`` / ``int`` / ``choice`` / ``bool``。
        low: 连续/整数参数下界（``choice`` 为 ``None``）。
        high: 连续/整数参数上界（``choice`` 为 ``None``）。
        step: 步长（可为 ``None`` 表示连续）。
        choices: ``choice`` 类型的候选值（按原文顺序，不排序）。
        description: 参数说明，写进生成文件的注释。
    """

    name: str
    default: float
    kind: str = "float"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "name": self.name,
            "default": self.default,
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "choices": list(self.choices),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParamSpace:
        """从字典还原（:meth:`to_dict` 的逆操作）。"""
        raw_choices = data.get("choices") or ()
        return cls(
            name=str(data["name"]),
            default=float(data["default"]),
            kind=str(data.get("kind", "float")),
            low=None if data.get("low") is None else float(data["low"]),
            high=None if data.get("high") is None else float(data["high"]),
            step=None if data.get("step") is None else float(data["step"]),
            choices=tuple(str(c) for c in raw_choices),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class PlatformSpec:
    """目标平台静态描述（数据驱动，不含逻辑）。

    Attributes:
        name: 平台名（如 ``algoforge``）。
        file_suffix: 导出文件后缀。
        module_header: 生成文件顶部的 import 行（按给出顺序渲染）。
        stub_modules: ``{模块名: (导出符号, ...)}`` —— 本地无该平台 SDK 时，
            由 :func:`install_stub_modules` 数据驱动地注入占位模块，
            使生成文件可在离线环境中被 import / exec（数值保真度回归依赖此能力）。
        stub_enums: ``{模块名: {符号名: (成员, ...)}}`` —— 这些占位符号是**枚举**，
            必须造出真 :class:`enum.Enum`（成员值取成员名，与真平台一致），
            否则 ``OrderType.BUY`` 之类的访问会直接 AttributeError。
        stub_abstracts: ``{模块名: {类名: (抽象方法, ...)}}`` —— 这些占位符号是
            **抽象基类**，必须造出真 ABC 并声明对应抽象方法。否则桩会掩盖
            「多继承把导出类变成抽象类、平台实例化不了」这类问题。
        supports_short: 平台是否支持做空。
        base_classes: 生成策略类继承的基类（按给出顺序）。
        default_symbol: 默认交易标的示例值。
    """

    name: str
    file_suffix: str = ".py"
    module_header: tuple[str, ...] = ()
    stub_modules: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    stub_enums: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    stub_abstracts: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    supports_short: bool = False
    base_classes: tuple[str, ...] = ()
    default_symbol: str = "XAUUSD"

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "name": self.name,
            "file_suffix": self.file_suffix,
            "module_header": list(self.module_header),
            "stub_modules": {k: list(v) for k, v in self.stub_modules.items()},
            "stub_enums": {
                k: {sym: list(members) for sym, members in enums.items()}
                for k, enums in self.stub_enums.items()
            },
            "stub_abstracts": {
                k: {sym: list(methods) for sym, methods in abstracts.items()}
                for k, abstracts in self.stub_abstracts.items()
            },
            "supports_short": self.supports_short,
            "base_classes": list(self.base_classes),
            "default_symbol": self.default_symbol,
        }


@dataclass(frozen=True)
class ExportResult:
    """一次导出的完整结果（不含文件写盘，写盘由 CLI 负责）。

    Attributes:
        platform: 平台名。
        filename: 建议文件名（``YYYYMMDD_名称_vN.py``）。
        source: 渲染出的源码全文。
        magic: 分配到的策略 magic（``int``，与 AlgoForge 侧 ``magic: int`` 一致）。
        param_space: 可调参数空间（顺序稳定）。
        issues: 静态检查结果。
        ok: 是否无 ``ERROR`` 级问题。
    """

    platform: str
    filename: str
    source: str
    magic: int | None = None
    param_space: tuple[ParamSpace, ...] = ()
    issues: tuple[LintIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """是否通过静态检查（无 ERROR）。"""
        return not any(i.severity == LintSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> tuple[LintIssue, ...]:
        """所有 ERROR 级问题。"""
        return tuple(i for i in self.issues if i.severity == LintSeverity.ERROR)

    @property
    def warnings(self) -> tuple[LintIssue, ...]:
        """所有 WARNING 级问题。"""
        return tuple(i for i in self.issues if i.severity == LintSeverity.WARNING)


class TargetPort(ABC):
    """目标平台导出端口（四件套）。

    子类需提供 :attr:`platform` 与 :meth:`extract` / :meth:`render` / :meth:`lint`；
    :meth:`compile` 为默认编排（extract → render → lint），通常无需覆写。
    """

    platform: PlatformSpec

    @abstractmethod
    def extract(self, spec: StrategySpec) -> dict[str, Any]:
        """从 :class:`StrategySpec` 提取平台相关的渲染上下文。

        Args:
            spec: 策略规格。

        Returns:
            渲染上下文字典（键由各平台自行约定，但必须包含 ``filename``；
            ``magic`` 为平台特有概念，平台无关层不强制，由具体适配器自行决定）。
        """

    @abstractmethod
    def render(self, spec: StrategySpec, ctx: Mapping[str, Any]) -> str:
        """把 :class:`StrategySpec` 与上下文渲染为平台源码文本。"""

    @abstractmethod
    def lint(self, source: str, spec: StrategySpec | None = None) -> list[LintIssue]:
        """对渲染结果做静态检查（含 repaint 检查）。"""

    def compile(self, spec: StrategySpec) -> ExportResult:  # noqa: A003 - 领域惯用名
        """编排一次导出：extract → render → lint（不写盘）。"""
        ctx = self.extract(spec)
        source = self.render(spec, ctx)
        issues = tuple(self.lint(source, spec))
        raw_space = ctx.get("param_space", ())
        space = tuple(ParamSpace.from_dict(p) if isinstance(p, Mapping) else p for p in raw_space)
        return ExportResult(
            platform=self.platform.name,
            filename=str(ctx["filename"]),
            source=source,
            magic=_as_int_or_none(ctx.get("magic")),
            param_space=space,
            issues=issues,
        )


def _as_int_or_none(value: Any) -> int | None:
    """把上下文里的 magic 归一成 ``int``；缺失或非数字返回 ``None``。

    Args:
        value: 原始值（``int`` / 数字字符串 / ``None`` / ``""``）。

    Returns:
        整数 magic；无法解析时 ``None``。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _make_stub(
    symbol: str,
    members: tuple[str, ...] = (),
    abstracts: tuple[str, ...] = (),
) -> Any:
    """造一个占位符号：抽象基类 / 枚举 / 普通空类，按声明选择。

    优先级：``abstracts`` > ``members`` > 空类。

    Args:
        symbol: 占位符号名。
        members: 枚举成员名；非空则造真 :class:`enum.Enum`（成员值取成员名）。
        abstracts: 抽象方法名；非空则造真 ABC 并声明这些方法为
            :func:`abc.abstractmethod`，使其 ``__abstractmethods__`` 与真平台一致。

    Returns:
        占位类型。还原抽象面是关键：桩若不给抽象方法，就测不出「多继承把导出类
        变成抽象类」这类只有真平台才暴露的问题。
    """
    if abstracts:
        namespace: dict[str, Any] = {"__stub__": True}
        for name in abstracts:
            def _unimplemented(self: Any, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError
            _unimplemented.__name__ = name
            _unimplemented.__qualname__ = f"{symbol}.{name}"
            namespace[name] = abstractmethod(_unimplemented)
        return ABCMeta(symbol, (), namespace)
    if members:
        return Enum(symbol, {name: name for name in members})
    return type(symbol, (), {"__stub__": True})


def install_stub_modules(platform: PlatformSpec) -> tuple[str, ...]:
    """按 :attr:`PlatformSpec.stub_modules` 注入占位模块（数据驱动）。

    仅在本机没有目标平台 SDK 时生效：已存在于 ``sys.modules`` 的模块不会被覆盖，
    因此在真实 AlgoForge 环境里调用本函数是**无操作**。

    Args:
        platform: 平台描述。

    Returns:
        本次新建并注册的模块名元组（供 :func:`uninstall_stub_modules` 清理）。
    """
    created: list[str] = []
    for dotted, exports in platform.stub_modules.items():
        parts = dotted.split(".")
        # 逐级建包：strategies.base 需要先存在 strategies
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name in sys.modules:
                continue
            module = types.ModuleType(name)
            module.__doc__ = f"妙算离线占位模块（{platform.name}）：{name}"
            if depth < len(parts):
                module.__path__ = []
            else:
                enums = platform.stub_enums.get(dotted, {})
                abstracts = platform.stub_abstracts.get(dotted, {})
                for symbol in exports:
                    setattr(
                        module,
                        symbol,
                        _make_stub(
                            symbol,
                            members=enums.get(symbol, ()),
                            abstracts=abstracts.get(symbol, ()),
                        ),
                    )
            sys.modules[name] = module
            created.append(name)
            if depth > 1:
                setattr(sys.modules[".".join(parts[: depth - 1])], parts[depth - 1], module)
    return tuple(created)


def uninstall_stub_modules(names: tuple[str, ...]) -> None:
    """移除 :func:`install_stub_modules` 注入的占位模块。

    Args:
        names: :func:`install_stub_modules` 的返回值。
    """
    for name in reversed(names):
        sys.modules.pop(name, None)
