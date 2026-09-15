# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""从**冻结的 AlphaMaster** 实时提取「名称/顺序冻结清单」，产出
``tests/fixtures/frozen_token_order.json``。

作用：把 AM 的 token 名称、顺序、arity、类别**一次性冻结**为 fixture。此后 M3/M4
的回归测试**只读 fixture、不再依赖 AM 可导入**（既不依赖 torch，也不依赖 AM 仓库
是否在原位）。这是后续 numpy 化移植的「名称/顺序锁」。

运行（需要能 import AM 的解释器，即装有 torch 的环境）：

    C:\\Users\\Administrator\\.workbuddy\\binaries\\python\\envs\\default\\Scripts\\python.exe \\
        scripts/gen_frozen_fixture.py

可用环境变量覆盖 AM 仓库位置：``MIAOSUAN_AM_ROOT``。

注意：脚本对 AM 仓库**零写入**（临时关闭字节码写入）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 妙算仓库根（本文件位于 <root>/scripts/）
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_AM_ROOT = Path(r"D:\backup\BaoBao\PythonProgram\AlphaMaster-main")
_OUT = _ROOT / "tests" / "fixtures" / "frozen_token_order.json"


def _am_root() -> Path:
    env = os.environ.get("MIAOSUAN_AM_ROOT")
    return Path(env) if env else _DEFAULT_AM_ROOT


def extract(am_root: Path) -> dict:
    """导入 AM 的 model_core.{features,ops,vocab} 并提取词表结构。"""
    vocab_py = am_root / "model_core" / "vocab.py"
    if not vocab_py.is_file():
        raise SystemExit(f"[ERROR] 未找到 AM vocab：{vocab_py}")

    prev_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(am_root))
    try:
        from model_core.features import FEATURE_REGISTRY
        from model_core.ops import OPERATOR_REGISTRY
        from model_core.vocab import FORMULA_VOCAB, VOCAB_SCHEMA_TAG, VOCAB_VERSION
    finally:
        sys.path.remove(str(am_root))
        sys.dont_write_bytecode = prev_dont_write

    features = [
        {"name": spec.name, "category": spec.category} for spec in FEATURE_REGISTRY.feature_specs
    ]
    operators = [
        {"name": spec.name, "arity": int(spec.arity)} for spec in OPERATOR_REGISTRY.operator_specs
    ]
    feature_names = list(FORMULA_VOCAB.feature_names)
    operator_names = list(FORMULA_VOCAB.operator_names)

    # 自洽性校验：名称/顺序必须与 registry 视图一致
    assert [f["name"] for f in features] == feature_names, "feature 视图不一致"
    assert [o["name"] for o in operators] == operator_names, "operator 视图不一致"

    return {
        "_comment": (
            "从冻结的 AlphaMaster 实时提取的「名称/顺序冻结清单」。"
            "由 scripts/gen_frozen_fixture.py 生成，请勿手工编辑。"
            "M3/M4 回归测试只读本文件，不依赖 AM 可导入。"
        ),
        "source": "AlphaMaster-main/model_core/{features,ops,vocab}.py",
        "vocab_version": VOCAB_VERSION,
        "vocab_schema_tag": VOCAB_SCHEMA_TAG,
        "feature_count": len(feature_names),
        "operator_count": len(operator_names),
        "vocab_size": len(feature_names) + len(operator_names),
        "feature_names": feature_names,
        "operator_names": operator_names,
        "features": features,
        "operators": operators,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 AM 名称/顺序冻结清单")
    parser.add_argument("--am-root", type=Path, default=None, help="AM 仓库根目录")
    parser.add_argument("--out", type=Path, default=_OUT, help="输出 JSON 路径")
    args = parser.parse_args()

    am_root = args.am_root or _am_root()
    data = extract(am_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"[OK] AM 根: {am_root}")
    print(
        f"[OK] features={data['feature_count']} operators={data['operator_count']} "
        f"size={data['vocab_size']} VOCAB_VERSION={data['vocab_version']}"
    )
    print(f"[OK] 写出: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
