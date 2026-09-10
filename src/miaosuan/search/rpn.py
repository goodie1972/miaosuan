"""RPN 表示与遗传算子（架构 §1.3 AD-2、M10）。

**表示**：定长 token 数组 ``L=8``（Q6-A）。token id 分段与 :mod:`miaosuan.core.vocab`
完全一致：feature ∈ ``[0, F-1]``，operator ∈ ``[F, F+O-1]``（``F=65``、``O=62``、``vocab=127``）。

**合法性（可求值）**由**栈深约束**保证：生成时对每一步做可行性剪枝，使得

* 任意前缀都可求值（栈深在放置每个 token 后 ≥ 1）；
* 末态栈深恰为 1（一个完整表达式）。

算子对栈深的净影响 = ``1 - arity``：feature(0) → ``+1``；unary(1) → ``0``；
binary(2) → ``-1``；ternary(3, ``GATE``/``IF_GT``) → ``-2``。

**可行性过滤**另复用 :func:`miaosuan.core.vm.validate_formula_structure` 的「感染模型」：
恒正算子（``TS_RANK``/``ABS`` …）后连续传播算子会令因子退化为 beta，属于**软违规**，
生成/变异时可选择性规避（不改变可求值性）。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..core.ops import OPS_CONFIG
from ..core.vm import validate_formula_structure
from ..core.vocab import FORMULA_VOCAB

__all__ = [
    "ARITY",
    "FEATURE_COUNT",
    "FEATURE_IDS",
    "FORMULA_LEN",
    "OPERATOR_IDS",
    "OPERATOR_OFFSET",
    "VOCAB_SIZE",
    "Individual",
    "crossover",
    "decode",
    "depth_profile",
    "hamming",
    "is_feasible",
    "random_feasible",
    "repair",
    "structure_violations",
]

#: 默认公式长度（Q6-A 保持 8）
FORMULA_LEN: int = 8

FEATURE_COUNT: int = FORMULA_VOCAB.feature_count  # 65
OPERATOR_OFFSET: int = FORMULA_VOCAB.operator_offset  # 65
VOCAB_SIZE: int = FORMULA_VOCAB.size  # 127
TOKEN_NAMES: tuple[str, ...] = FORMULA_VOCAB.token_names

#: token id → arity（仅算子）
ARITY: dict[int, int] = {
    OPERATOR_OFFSET + i: int(cfg[2]) for i, cfg in enumerate(OPS_CONFIG)
}
#: feature token id
FEATURE_IDS: tuple[int, ...] = tuple(range(FEATURE_COUNT))
#: operator token id
OPERATOR_IDS: tuple[int, ...] = tuple(sorted(ARITY))
#: arity → operator token ids
_OPS_BY_ARITY: dict[int, tuple[int, ...]] = {
    a: tuple(t for t in OPERATOR_IDS if ARITY[t] == a) for a in sorted(set(ARITY.values()))
}

_CLASS_NAMES: tuple[str, ...] = ("feat", "unary", "binary", "ternary")


def is_feature(token: int) -> bool:
    """token 是否为特征位。"""
    return 0 <= int(token) < FEATURE_COUNT


def is_operator(token: int) -> bool:
    """token 是否为算子位。"""
    return int(token) in ARITY


def arity_of(token: int) -> int:
    """算子 token 的 arity；特征位返回 0；未知 token 抛 :class:`KeyError`。"""
    t = int(token)
    if 0 <= t < FEATURE_COUNT:
        return 0
    return ARITY[t]


def _token_class(token: int) -> str | None:
    """token 的类别名（``feat``/``unary``/``binary``/``ternary``），未知返回 ``None``。"""
    t = int(token)
    if 0 <= t < FEATURE_COUNT:
        return "feat"
    a = ARITY.get(t)
    if a == 1:
        return "unary"
    if a == 2:
        return "binary"
    if a == 3:
        return "ternary"
    return None


def _random_token_of_class(rng: np.random.Generator, cls: str) -> int:
    if cls == "feat":
        return int(rng.integers(0, FEATURE_COUNT))
    arity = {"unary": 1, "binary": 2, "ternary": 3}[cls]
    choices = _OPS_BY_ARITY[arity]
    return int(choices[int(rng.integers(0, len(choices)))])


def depth_profile(tokens: Iterable[int]) -> list[int] | None:
    """返回逐 token 处理后的栈深序列；若不可求值返回 ``None``。

    不可求值包括：token 越界、或算子弹出时栈深不足。
    """
    depth = 0
    profile: list[int] = []
    for tok in tokens:
        t = int(tok)
        if 0 <= t < FEATURE_COUNT:
            depth += 1
        elif t in ARITY:
            a = ARITY[t]
            if depth < a:
                return None
            depth = depth - a + 1
        else:
            return None
        profile.append(depth)
    return profile


def is_feasible(tokens: Iterable[int]) -> bool:
    """公式是否可求值（任意前缀合法 + 末态栈深恰为 1）。"""
    prof = depth_profile(tokens)
    if prof is None:
        return False
    return prof[-1] == 1


def decode(tokens: Iterable[int]) -> str:
    """token 序列 → 人类可读公式（``A → B → ...``）。"""
    names: list[str] = []
    for tok in tokens:
        t = int(tok)
        names.append(TOKEN_NAMES[t] if 0 <= t < VOCAB_SIZE else f"tok{t}")
    return " → ".join(names)


def hamming(a: Iterable[int], b: Iterable[int]) -> int:
    """两个等长 token 序列的汉明距离（逐位不等个数）。"""
    av = np.asarray(list(a), dtype=np.int64)
    bv = np.asarray(list(b), dtype=np.int64)
    if av.shape != bv.shape:
        raise ValueError("hamming: 长度不一致")
    return int(np.count_nonzero(av != bv))


def structure_violations(tokens: Iterable[int]) -> list[str]:
    """软违规（感染模型）：恒正算子后连续传播算子等。空列表 = 无违规。"""
    return validate_formula_structure([int(t) for t in tokens], TOKEN_NAMES)


def _allowed_classes(depth: int, remaining: int) -> list[tuple[str, int]]:
    """在当前栈深 ``depth``、其后还剩 ``remaining`` 个 token 时的可行类别与目标深度。"""
    allowed: list[tuple[str, int]] = []
    # feature：+1
    if depth + 1 <= remaining + 1:
        allowed.append(("feat", depth + 1))
    # unary：0
    if depth >= 1 and depth <= remaining + 1:
        allowed.append(("unary", depth))
    # binary：-1
    if depth >= 2 and (depth - 1) <= remaining + 1:
        allowed.append(("binary", depth - 1))
    # ternary：-2
    if depth >= 3 and (depth - 2) <= remaining + 1:
        allowed.append(("ternary", depth - 2))
    return allowed


def random_feasible(
    rng: np.random.Generator,
    length: int = FORMULA_LEN,
    *,
    feature_prob: float = 0.55,
    avoid_infection: bool = False,
    max_tries: int = 32,
) -> np.ndarray:
    """随机生成一个**可求值**的定长公式（栈深约束 + 每步可行性剪枝）。

    :param feature_prob: 每步选「特征」的先验权重（其余均分给 unary/binary/ternary）。
    :param avoid_infection: 若为 True，尽量规避感染模型软违规（有界重试）。
    """
    for _ in range(max(1, max_tries if avoid_infection else 1)):
        tokens = np.empty(int(length), dtype=np.int64)
        depth = 0
        for i in range(int(length)):
            remaining = int(length) - 1 - i
            allowed = _allowed_classes(depth, remaining)
            if not allowed:  # pragma: no cover - 剪枝保证非空
                raise RuntimeError("random_feasible: 无可行类别（不应发生）")
            weights = []
            for cls, _nd in allowed:
                weights.append(feature_prob if cls == "feat" else (1.0 - feature_prob))
            w = np.asarray(weights, dtype=np.float64)
            w = w / w.sum()
            idx = int(rng.choice(len(allowed), p=w))
            cls, new_depth = allowed[idx]
            tokens[i] = _random_token_of_class(rng, cls)
            depth = new_depth
        if not avoid_infection or not structure_violations(tokens):
            return tokens
    return tokens  # pragma: no cover - 有界重试后兜底


def repair(tokens: Iterable[int], rng: np.random.Generator, length: int | None = None) -> np.ndarray:
    """把任意 token 序列修复为**可求值**公式（保留尽可能多的合法前缀）。

    从左到右扫描：若当前 token 的类别在「可行性剪枝允许集」内则保留，否则就地重采样一个
    允许类别的随机 token。由剪枝性质保证末态栈深恰为 1。
    """
    src = [int(t) for t in tokens]
    n = int(length) if length is not None else len(src)
    out = np.empty(n, dtype=np.int64)
    depth = 0
    for i in range(n):
        remaining = n - 1 - i
        allowed = _allowed_classes(depth, remaining)
        allowed_map = dict(allowed)
        tok = src[i] if i < len(src) else -1
        cls = _token_class(tok) if tok >= 0 else None
        if cls is not None and cls in allowed_map:
            out[i] = tok
            depth = allowed_map[cls]
        else:
            idx = int(rng.integers(0, len(allowed)))
            pick_cls, new_depth = allowed[idx]
            out[i] = _random_token_of_class(rng, pick_cls)
            depth = new_depth
    return out


def crossover(
    a: Iterable[int],
    b: Iterable[int],
    rng: np.random.Generator,
    *,
    length: int | None = None,
) -> np.ndarray:
    """均匀交叉 + RPN 合法性修复（修复后一定可求值）。"""
    av = np.asarray(list(a), dtype=np.int64)
    bv = np.asarray(list(b), dtype=np.int64)
    n = int(length) if length is not None else int(av.shape[0])
    if av.shape[0] != bv.shape[0]:
        raise ValueError("crossover: 父代长度不一致")
    mask = rng.random(n) < 0.5
    child = np.where(mask, av, bv).astype(np.int64)
    if is_feasible(child):
        return child
    return repair(child, rng, length=n)


def mutate(
    tokens: Iterable[int],
    rng: np.random.Generator,
    *,
    p_point: float = 0.5,
    p_reset: float = 0.2,
    n_points: int = 2,
) -> np.ndarray:
    """变异算子（保证输出**可求值**）。

    * **单点替换**（``p_point``）：把随机位置换成同类 token（feature→feature；
      算子→同 arity 算子），栈深不变 → 一定合法；
    * **重置**（``p_reset``）：整条重采样为随机可行个体，用于跳出局部最优。
    """
    src = np.asarray(list(tokens), dtype=np.int64)
    n = int(src.shape[0])
    if rng.random() < p_reset:
        return random_feasible(rng, n)
    child = src.copy()
    for _ in range(int(n_points)):
        i = int(rng.integers(0, n))
        cls = _token_class(int(child[i]))
        if cls is None:
            child[i] = _random_token_of_class(rng, "feat")
        else:
            child[i] = _random_token_of_class(rng, cls)
    return child


@dataclass(eq=False)
class Individual:
    """种群个体（定长 RPN 公式 + 适配度）。

    :param tokens: ``[L]`` int64 token 序列（可求值）。
    :param train_score: 开发集训练分（AM 口径 ``train_score``）。
    :param val_score: 开发集验证分（AM 口径 ``val_score``；GA 目标）。
    :param fitness: GA 适配度（默认取 ``val_score``）。
    :param status: ``"ok"`` / ``"none"`` / ``"const"`` / ``"unevaluated"``。
    :param birth_gen: 出生代（用于精英年龄/诊断）。
    """

    tokens: np.ndarray
    train_score: float = float("-inf")
    val_score: float = float("-inf")
    fitness: float = float("-inf")
    status: str = "unevaluated"
    birth_gen: int = 0

    def __post_init__(self) -> None:
        self.tokens = np.asarray(self.tokens, dtype=np.int64)

    def key(self) -> tuple[int, ...]:
        """token 元组（用于去重 / 缓存）。"""
        return tuple(int(t) for t in self.tokens)

    def hamming(self, other: Individual) -> int:
        """与另一个体的汉明距离。"""
        return hamming(self.tokens, other.tokens)

    @property
    def decoded(self) -> str:
        """人类可读公式。"""
        return decode(self.tokens)

    def is_feasible(self) -> bool:
        """公式是否可求值。"""
        return is_feasible(self.tokens)

    def is_evaluated(self) -> bool:
        """是否已评估。"""
        return self.status != "unevaluated"
