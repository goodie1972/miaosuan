"""中间表示（IR）—— 挖掘层与目标平台之间的唯一契约（M13）。

* :mod:`miaosuan.ir.schema` —— :class:`StrategySpec` 及其四段结构；
* :mod:`miaosuan.ir.codec` —— 确定性序列化 / 反序列化；
* :mod:`miaosuan.ir.provenance` —— 可追溯信封（git sha / 词表版本 / 数据指纹 / 种子）。

IR 只描述「是什么」，不含任何平台语法；平台差异全部下沉到
:mod:`miaosuan.adapters`。
"""

from .codec import decode_spec, encode_spec, read_spec, write_spec
from .provenance import build_provenance, resolve_git_sha, utc_now_iso
from .schema import (
    SPEC_VERSION,
    Evidence,
    FactorPayload,
    ParamPayload,
    Provenance,
    Semantics,
    StrategySpec,
)

__all__ = [
    "SPEC_VERSION",
    "Evidence",
    "FactorPayload",
    "ParamPayload",
    "Provenance",
    "Semantics",
    "StrategySpec",
    "build_provenance",
    "decode_spec",
    "encode_spec",
    "read_spec",
    "resolve_git_sha",
    "utc_now_iso",
    "write_spec",
]
