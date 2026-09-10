"""差分对拍（parity）测试的共享装置 —— 加载冻结的 AlphaMaster 作为 Oracle。

架构 §6.3：妙算（numpy）与冻结 AM（torch）对拍。AM 仓库**只读**，妙算绝不修改它。

Oracle 的两级策略
------------------

1. **冻结快照（始终可用）**：``tests/fixtures/am_vocab_snapshot.json`` 记录了 AM 词表的
   版本字符串与有序 token 名称。它由 AM 侧 ``compute_vocab_version`` 派生后逐字符冻结，
   作为稳定、离线、无需 torch 的对拍基准。

2. **活体对拍（可选，装了 torch 时启用）**：直接 ``import model_core.vocab`` 读取真实 AM
   模块，比对版本字符串与 token 名称。

关于 torch 桩
-------------

``vocab`` 的恒等性**只**取决于字符串 token 名称的有序拼接 + ``hashlib.sha256``，
与 torch 的数值行为**完全无关**。AM 的 ``model_core/{registry,features,ops}.py`` 仅在
模块顶层 ``import torch``（用于类型注解，导入期不执行任何 torch 运算）。因此当本机
未安装 torch 时，本装置注入一个**最小 torch 桩**以完成「读名称」这一动作 ——
它不影响任何被断言的值。若已安装真实 torch，则优先使用真实 torch。
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ── 路径 ───────────────────────────────────────────────────────────────────

#: 冻结 AM 仓库（只读 Oracle）。可用环境变量覆盖（CI / 异机）。
_DEFAULT_AM_ROOT = Path(r"D:\backup\BaoBao\PythonProgram\AlphaMaster-main")

#: 冻结快照
_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "am_vocab_snapshot.json"

#: M5 对拍基准（torch Oracle 生成）
_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_M5_NPZ = _FIXTURES_DIR / "m5_baseline.npz"
_M5_JSON = _FIXTURES_DIR / "m5_baseline.json"
_E2E_NPZ = _FIXTURES_DIR / "e2e_xauusd_baseline.npz"
_E2E_JSON = _FIXTURES_DIR / "e2e_xauusd_meta.json"


def am_root() -> Path:
    """返回冻结 AM 仓库根目录（允许环境变量 ``MIAOSUAN_AM_ROOT`` 覆盖）。"""
    env = os.environ.get("MIAOSUAN_AM_ROOT")
    return Path(env) if env else _DEFAULT_AM_ROOT


# ── torch 桩 ───────────────────────────────────────────────────────────────


class _AnyStub:
    """万能桩对象：任何属性访问/调用/下标都返回自身。"""

    def __call__(self, *args: Any, **kwargs: Any) -> _AnyStub:  # noqa: ARG002
        return self

    def __getattr__(self, name: str) -> _AnyStub:
        return _AnyStub()

    def __getitem__(self, key: Any) -> _AnyStub:  # noqa: ARG002
        return _AnyStub()

    def __repr__(self) -> str:  # pragma: no cover - 展示
        return "<torch-stub>"


class _TorchStubModule(types.ModuleType):
    """torch 桩模块：仅用于让 ``import torch`` 成功（导入期不执行 torch 运算）。"""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return _AnyStub()


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - 取决于环境
        return False
    return True


def _ensure_torch() -> tuple[bool, Any]:
    """确保 torch 可导入。

    :returns: ``(stubbed, restore)`` —— ``stubbed`` 表示是否用了桩；
              ``restore`` 是恢复 ``sys.modules['torch']`` 的可调用对象。
    """
    if _torch_available():
        return False, (lambda: None)

    previous = sys.modules.get("torch")
    sys.modules["torch"] = _TorchStubModule("torch")

    def _restore() -> None:
        if previous is not None:
            sys.modules["torch"] = previous
        else:
            sys.modules.pop("torch", None)

    return True, _restore


# ── 加载 AM vocab ──────────────────────────────────────────────────────────


def _load_am_vocab() -> tuple[Any | None, bool, str]:
    """加载 AM 的 ``model_core.vocab`` 模块。

    :returns: ``(module_or_None, stubbed, reason)``。``module`` 为 ``None`` 时
              ``reason`` 说明原因（供测试 ``skip``）。
    """
    root = am_root()
    if not (root / "model_core" / "vocab.py").is_file():
        return None, False, f"AM Oracle 不可用：{root} 下无 model_core/vocab.py"

    stubbed, restore = _ensure_torch()
    inserted = str(root)
    # 保证冻结的 AM 仓库**零写入**：导入期不落 __pycache__ 字节码
    prev_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, inserted)
    try:
        import importlib

        module = importlib.import_module("model_core.vocab")
    except Exception as exc:  # pragma: no cover - 环境相关
        restore()
        return None, False, f"加载 AM vocab 失败：{exc!r}"
    finally:
        if inserted in sys.path:
            sys.path.remove(inserted)
        sys.dont_write_bytecode = prev_dont_write
        restore()

    return module, stubbed, ""


# ── 装置 ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def frozen_snapshot() -> dict[str, Any]:
    """冻结的 AM 词表快照（离线 Oracle）。"""
    with _FIXTURE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def am_oracle() -> tuple[Any | None, bool, str]:
    """AM 活体 vocab 模块（或 ``None``）及其是否使用 torch 桩。"""
    return _load_am_vocab()


@pytest.fixture(scope="session")
def am_vocab(am_oracle: tuple[Any | None, bool, str]) -> Any:
    """AM 活体 vocab 模块；不可用时跳过相关测试。"""
    module, _stubbed, reason = am_oracle
    if module is None:
        pytest.skip(reason or "AM Oracle 不可用")
    return module


@pytest.fixture(scope="session")
def am_used_torch_stub(am_oracle: tuple[Any | None, bool, str]) -> bool:
    """本次对拍是否使用了 torch 桩（供测试输出披露）。"""
    return am_oracle[1]


# ── M5 对拍基准（vm / signal / backtest / evaluator）────────────────────────


@pytest.fixture(scope="session")
def m5_npz() -> dict[str, np.ndarray]:
    """M5 数值基准（torch Oracle 生成）。"""
    if not _M5_NPZ.is_file():
        pytest.skip(f"缺少 M5 基准：{_M5_NPZ}（先跑 scripts/gen_m5_baseline.py）")
    with np.load(_M5_NPZ) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def m5_meta() -> dict[str, Any]:
    """M5 元信息与标量结果（torch Oracle 生成）。"""
    if not _M5_JSON.is_file():
        pytest.skip(f"缺少 M5 元信息：{_M5_JSON}")
    return json.loads(_M5_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def e2e_npz() -> dict[str, np.ndarray]:
    """端到端 XAUUSD 基准（原始面板 + 特征 + 因子）。"""
    if not _E2E_NPZ.is_file():
        pytest.skip(f"缺少端到端基准：{_E2E_NPZ}（先跑 scripts/gen_e2e_xauusd_baseline.py）")
    with np.load(_E2E_NPZ) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def e2e_meta() -> dict[str, Any]:
    """端到端元信息（形状 / 公式 / 统计量）。"""
    if not _E2E_JSON.is_file():
        pytest.skip(f"缺少端到端元信息：{_E2E_JSON}")
    return json.loads(_E2E_JSON.read_text(encoding="utf-8"))
