"""数据指纹（sha256，治 R14）—— 妙算 ``data/fingerprint.py``。

架构 §9.4 数据契约：**数据指纹 = sha256(对 OHLCV + time + symbols 的有序字节序列化)**，
写入 ``provenance.data_fingerprint``，用于把「同一次实验用了哪份数据」钉死、可复现。

设计要点：

* **确定性**：一律 ``np.ascontiguousarray`` 后取 ``.tobytes()``；字段名按**字典序**排序，
  与插入顺序无关；每段前写入 ``name|dtype|shape`` 头，避免「不同形状/类型拼出相同字节流」。
* **长度前缀**：逐段写入 8 字节小端长度，防止拼接歧义（``"ab"+"c"`` vs ``"a"+"bc"``）。
* **纯函数、无 IO**：本模块只做字节序列化与哈希，不读文件、不读环境变量。
* **不 import ``panel``**：只接受裸数组/字节，避免与 :mod:`miaosuan.data.panel` 形成循环依赖。

指纹字符串形如 ``"sha256:<64 hex>"``；``split_fingerprint`` / ``holdout_fingerprint``
在此基础上叠加切分参数，供 hold-out 封印台账使用。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np

__all__ = [
    "arrays_fingerprint",
    "holdout_fingerprint",
    "panel_fingerprint",
    "split_fingerprint",
]

#: 指纹前缀（写入产物时保留，便于人眼识别算法）
_PREFIX = "sha256:"

#: symbols 分隔符（信息分隔符，几乎不可能出现在品种名中）
_SYMBOL_SEP = b"\x1f"


def _digest(parts: Iterable[bytes]) -> str:
    """对若干字节段做带长度前缀的 sha256。"""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "little"))
        hasher.update(part)
    return hasher.hexdigest()


def _array_part(name: str, arr: np.ndarray) -> bytes:
    """把一个具名数组序列化为「头 + 原始字节」。"""
    contiguous = np.ascontiguousarray(arr)
    header = f"{name}|{contiguous.dtype.str}|{tuple(contiguous.shape)}".encode()
    return header + b"\x00" + contiguous.tobytes()


def arrays_fingerprint(
    named_arrays: Mapping[str, np.ndarray],
    *,
    extra_tokens: Iterable[str] = (),
    prefix: bool = True,
) -> str:
    """对若干具名数组（+ 可选文本 token）计算确定性 sha256。

    :param named_arrays: ``{字段名: 数组}``；字段名将按字典序排序后参与哈希。
    :param extra_tokens: 额外参与哈希的文本片段（如版本串、口径名）。
    :param prefix: 是否加 ``"sha256:"`` 前缀。
    """
    parts: list[bytes] = []
    for name in sorted(named_arrays):
        parts.append(_array_part(name, named_arrays[name]))
    for token in extra_tokens:
        parts.append(f"token:{token}".encode())
    digest = _digest(parts)
    return f"{_PREFIX}{digest}" if prefix else digest


def panel_fingerprint(
    fields: Mapping[str, np.ndarray],
    time: np.ndarray,
    symbols: Iterable[str],
) -> str:
    """Panel 数据指纹 = sha256(OHLCV + time + symbols)（架构 §9.4）。

    :param fields: ``{"open": [N, T], "high": ..., ...}``。
    :param time: ``[T]`` Unix 秒时间戳。
    :param symbols: 品种名序列（顺序敏感）。
    """
    named: dict[str, np.ndarray] = dict(fields)
    named["__time__"] = np.ascontiguousarray(time)
    named["__symbols__"] = np.frombuffer(
        _SYMBOL_SEP.join(s.encode("utf-8") for s in symbols), dtype=np.uint8
    )
    return arrays_fingerprint(named)


def split_fingerprint(
    panel_fp: str,
    *,
    ratios: tuple[float, float, float],
    purge_gap: int,
    embargo: int,
    seed: int,
) -> str:
    """把「数据指纹 + 切分参数」合成切分指纹（同数据同 seed → 同指纹 → 可复现）。"""
    token = (
        f"{panel_fp}|ratios={ratios[0]},{ratios[1]},{ratios[2]}"
        f"|purge={purge_gap}|embargo={embargo}|seed={seed}"
    )
    return f"{_PREFIX}{_digest([token.encode('utf-8')])}"


def holdout_fingerprint(split_fp: str, holdout_panel_fp: str) -> str:
    """hold-out 指纹 = sha256(切分指纹 + hold-out 面板指纹)，供一次性封印台账使用。"""
    token = f"holdout|split={split_fp}|panel={holdout_panel_fp}".encode()
    return f"{_PREFIX}{_digest([token])}"
