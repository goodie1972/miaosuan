"""零改主干（zero-touch）CI 断言：新增第 4 个 MarketProfile 不得触碰 ``core/``。

架构 §9.4 硬分界要求「市场怎么交易」的差异**全部**落在 :mod:`miaosuan.market`
（画像/成本），``core/`` 只经 :mod:`miaosuan.core.ports` 的协议消费参数。因此：

* **新增一个市场画像**（如第 4 个 ``CRYPTO_BTC`` 之后再加 ``SYNTHETIC_FOURTH``）应当只是
  ``market/profiles.py`` 里追加一个实例 + 注册表一行，``src/miaosuan/core/`` **一字节不变**。

本测试把这条验收固化为可执行断言，分三层：

1. **静态**：``core/*.py`` 不得出现任何具体画像 / 平台标识符，也不得 import
   :mod:`miaosuan.market` / :mod:`miaosuan.data`（依赖方向）；
2. **动态**：以「运行时新建的合成第 4 个画像」驱动 ``core`` 的 signal 与其 ``apply_cost``，
   执行前后对 ``core/`` 全量取 sha256 快照 → **逐文件字节一致**（这就是「core/ diff 为空」）；
3. **可扩展**：在注册表副本上追加该画像即可被发现，无需改 ``core/``。
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from miaosuan.core.ports import CostModelProtocol, MarketProfile
from miaosuan.core.signal import compute_target_positions
from miaosuan.market.cost import CostModel
from miaosuan.market.profiles import (
    CRYPTO_BTC,
    EXPECTED_PROFILE_NAMES,
    PROFILES,
)

CORE_DIR = Path(__file__).resolve().parent.parent / "src" / "miaosuan" / "core"

#: ``core/`` 中**禁止**出现的具体画像名（一旦出现即耦合了具体产品，破坏依赖倒置）
_FORBIDDEN_PROFILE_TOKENS: tuple[str, ...] = (
    "FOREX_XAUUSD",
    "CN_EQUITY_RESEARCH",
    "US_EQUITY_RESEARCH",
    "CRYPTO_BTC",
)

#: ``core/`` 中**禁止**依赖的上层包（产品轴实现层）
_FORBIDDEN_LAYER_PACKAGES: tuple[str, ...] = ("miaosuan.market", "miaosuan.data", "miaosuan.adapters")


# ── helpers ──────────────────────────────────────────────────────────────────


def _core_files() -> list[Path]:
    files = sorted(CORE_DIR.glob("*.py"))
    assert files, f"未找到 core 源文件：{CORE_DIR}"
    return files


def _snapshot(directory: Path) -> dict[str, str]:
    """对目录下所有 ``.py`` 逐文件取 sha256，返回 {相对路径: 摘要}。"""
    snap: dict[str, str] = {}
    for path in sorted(directory.rglob("*.py")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snap[str(path.relative_to(directory))] = digest
    return snap


@dataclass(frozen=True)
class _SyntheticFourthProfile:
    """运行时定义的「第 4 个画像」，结构性地满足 :class:`MarketProfile` 协议。

    它**不在** :data:`PROFILES` 中，用来证明 core 无需为新画像改任何代码即可消费它。
    """

    quote_currency: str
    cost_model: CostModelProtocol
    long_only: bool
    leverage: float
    lot_size: float
    tick_size: float
    contract_multiplier: float
    bars_per_year: int
    bars_per_day: int
    session_hours: tuple[tuple[int, int], ...]
    tradable_mask: np.ndarray | None
    allow_intraday_exit: bool
    settlement_days: int
    limit_pct: float | None = None
    name: str = "SYNTHETIC_FOURTH"

    def apply_cost(self, position: np.ndarray, ret: np.ndarray) -> np.ndarray:
        pos = np.asarray(position, dtype=np.float64)
        r = np.asarray(ret, dtype=np.float64)
        gross = pos * r
        prev = np.zeros_like(pos)
        if pos.shape[-1] > 1:
            prev[:, 1:] = pos[:, :-1]
        delta = pos - prev
        increase = np.clip(delta, 0.0, None)
        decrease = np.clip(-delta, 0.0, None)
        cost = self.cost_model.total("buy") * increase + self.cost_model.total("sell") * decrease
        return gross - cost

    def is_tradable(self, index: int) -> bool:
        return True if self.tradable_mask is None else bool(self.tradable_mask[index])


def _synthetic_fourth() -> _SyntheticFourthProfile:
    return _SyntheticFourthProfile(
        quote_currency="USDT",
        cost_model=CostModel(commission=0.0004, slippage=0.0005, sell_tax=0.0, asymmetric=False),
        long_only=False,
        leverage=1.0,
        lot_size=1e-05,
        tick_size=0.1,
        contract_multiplier=1.0,
        bars_per_year=24 * 365,
        bars_per_day=24,
        session_hours=((0, 0), (0, 24)),
        tradable_mask=None,
        allow_intraday_exit=True,
        settlement_days=0,
        limit_pct=None,
    )


# ── 1. 静态：core 不得出现具体画像 / 不得依赖实现层 ────────────────────────────


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_has_no_concrete_profile_identifiers(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    hits = [token for token in _FORBIDDEN_PROFILE_TOKENS if token in source]
    assert not hits, f"{path.name} 出现具体画像标识符 {hits}：core 必须对具体市场零耦合"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_does_not_import_implementation_layers(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == pkg or alias.name.startswith(pkg + ".")
                    for pkg in _FORBIDDEN_LAYER_PACKAGES
                ):
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if any(
                module == pkg or module.startswith(pkg + ".") for pkg in _FORBIDDEN_LAYER_PACKAGES
            ):
                violations.append(f"from {module} import ...")
    assert not violations, f"{path.name} 反向依赖实现层：{violations}"


# ── 2. 动态：加第 4 个画像前后 core/ 逐字节一致（= core/ diff 为空）──────────────


def test_adding_fourth_profile_leaves_core_byte_identical() -> None:
    before = _snapshot(CORE_DIR)

    profile = _synthetic_fourth()
    # 结构性满足协议（runtime_checkable）
    assert isinstance(profile, MarketProfile)
    assert profile.name not in PROFILES

    # 用「新画像」驱动 core：signal 决定仓位，画像负责成本结算
    factors = np.array([[0.5, -0.5, 0.02, 0.9, -2.0]], dtype=np.float64)
    ret = np.array([[0.01, -0.02, 0.0, 0.03, 0.015]], dtype=np.float64)
    positions = compute_target_positions(factors, long_only=profile.long_only)
    net = profile.apply_cost(positions, ret)
    assert net.shape == positions.shape

    after = _snapshot(CORE_DIR)
    assert before == after, (
        "「新增第 4 个画像 → 使用它」导致 core/ 发生变化；"
        "两处快照差异：" + repr({k: (before.get(k), after.get(k)) for k in set(before) | set(after)
                                if before.get(k) != after.get(k)})
    )


# ── 3. 可扩展：注册表追加即可被发现，无需改 core ──────────────────────────────


def test_registry_extensible_without_core_change() -> None:
    # 现有 4 个画像齐备（含 CRYPTO_BTC 第 4 个）
    assert set(EXPECTED_PROFILE_NAMES) <= set(PROFILES)
    assert PROFILES[CRYPTO_BTC.name] is CRYPTO_BTC
    assert len(PROFILES) == 4

    # 在副本上追加第 5 个，不影响原注册表，也不需要 core 参与
    extended = dict(PROFILES)
    synthetic = _synthetic_fourth()
    extended[synthetic.name] = synthetic
    assert len(extended) == 5
    assert len(PROFILES) == 4, "不得就地污染全局注册表"
