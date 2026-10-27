# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""工作区根解析回归测试——**打包态必须指回真实仓库，而不是 PyInstaller 临时目录**。

历史坑：onefile 打包后模块 ``__file__`` 是 ``...\\Temp\\_MEIxxxx\\miaosuan\\config.py``，
按源码布局 ``parents[2]`` 反推仓库根会得到 ``...\\Temp``。后果不是崩溃而是**静默空列表**：
exe 里「规范 / 已导出策略 / 回测历史」全空，且退出 exe 后产物随临时目录一起消失。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from miaosuan import config


def _make_workspace(root: Path) -> Path:
    """在 ``root`` 落下工作区标记（``settings.yaml`` + ``data/``）并返回该路径。"""
    (root / "data").mkdir(parents=True)
    (root / "settings.yaml").write_text("webui:\n  port: 8686\n", encoding="utf-8")
    return root


def test_explicit_root_env_wins(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``MIAOSUAN_ROOT`` 是唯一硬覆盖：设了就必须用它，哪怕在源码运行。"""
    root = _make_workspace(tmp_path / "workspace")
    monkeypatch.setenv("MIAOSUAN_ROOT", str(root))
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert config._resolve_repo_root().resolve() == root.resolve().resolve()


def test_malformed_root_env_is_ignored(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """指向不存在目录的 ``MIAOSUAN_ROOT`` 必须被丢弃——否则产物写进虚无。"""
    monkeypatch.setenv("MIAOSUAN_ROOT", str(tmp_path / "nope"))
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert config._resolve_repo_root() == Path(config.__file__).resolve().parents[2]


def test_frozen_build_searches_up_from_exe_and_finds_workspace(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """打包态：从 exe 所在目录逐级向上找工作区，命中即返回。"""
    root = _make_workspace(tmp_path / "miaosuan")
    exe = root / "launcher" / "dist" / "miaosuan_launcher.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    monkeypatch.setenv("MIAOSUAN_ROOT", "")
    # CWD 故意落在工作区之外：证明锚点来自 exe 位置，而不是恰好站在仓库里。
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)

    assert config._resolve_repo_root().resolve() == root.resolve()


def test_frozen_without_workspace_falls_back_to_cwd(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exe 被单独拷走（找不到工作区）：退回 CWD，产物至少落在用户可见处而非 Temp。"""
    orphan = tmp_path / "Desktop"
    orphan.mkdir()
    exe = orphan / "miaosuan_launcher.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    monkeypatch.setenv("MIAOSUAN_ROOT", "")
    monkeypatch.chdir(orphan)

    assert config._resolve_repo_root().resolve() == orphan.resolve()


def test_source_layout_root_is_repo(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """源码态（未打包、无覆盖）：仓库根仍是 ``src/miaosuan/config.py`` 往上两层。"""
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setenv("MIAOSUAN_ROOT", "")
    root = config._resolve_repo_root()
    assert (root / "src" / "miaosuan" / "config.py").is_file()


def test_directory_accessors_are_absolute_under_repo_root() -> None:
    """产物目录访问器必须已被解析成 ``REPO_ROOT`` 下的绝对路径（CWD 无关）。"""
    for fn in (config.artifacts_dir, config.strategies_dir, config.results_dir):
        path = Path(fn())
        assert path.is_absolute(), fn
        assert config.REPO_ROOT in path.parents, (fn, path)


def test_kline_dir_is_anchored_not_cwd_relative() -> None:
    """行情目录同样不得返回随 CWD 漂移的相对路径。

    双击 exe 时 CWD 是 ``launcher\\dist``：若把设置里的 ``data/cache`` 原样返回，
    「本地数据 / 已缓存文件」在打包版里就是空的（且不报错）。
    """
    resolved = Path(config.kline_data_dir({"MIAOSUAN_KLINE_DIR": "data/cache"}))
    assert resolved.is_absolute()
    assert resolved == config.REPO_ROOT / "data" / "cache"
    assert Path(config.kline_data_dir()).is_absolute()


def test_persistent_state_files_are_anchored_at_repo_root() -> None:
    """两份**持久化台账**不得随 CWD / 源码位置漂移。

    - hold-out 封印台账漂移 = 同一指纹可被反复窥探（一次性封印失效）；
    - magic 账本漂移 = 每次启动从 01 重新分配，策略号撞车。
    """
    from miaosuan.adapters.shenji import magic_registry
    from miaosuan.data import split
    from miaosuan.tune.engine import TuneConfig, _default_tune_out_dir

    for path in (split.DEFAULT_SEAL_PATH, magic_registry.DEFAULT_LEDGER_PATH):
        assert path.is_absolute(), path
        assert config.REPO_ROOT in path.parents, path
        assert str(path).startswith(str(config.REPO_ROOT / "data" / "state")), path

    cfg = TuneConfig(spec_path="x", param_spaces=[])
    assert Path(cfg.out_dir).is_absolute()
    assert cfg.out_dir == _default_tune_out_dir()


def test_frozen_launcher_declares_bundled_spawn_form(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """打包 exe 必须自报 ``MIAOSUAN_BUNDLED=1``，否则 WebUI 会 spawn 出错误的命令行。

    ``server._cli_process()`` 只在 bundled 时用 ``[exe, "verify", ...]``；若只在启动器
    模式置位，用 ``exe ui``（CLI 模式起的仪表盘）就会得到
    ``exe -m miaosuan.cli verify ...``——exe 认不出参数，落回启动器模式再起一个服务并
    弹 MessageBox，「验证」按钮永久卡死（实测踩过）。
    """
    from importlib.util import module_from_spec, spec_from_file_location

    main_py = Path(__file__).resolve().parents[1] / "launcher" / "main.py"
    spec = spec_from_file_location("miaosuan_launcher_main", main_py)
    assert spec and spec.loader
    launcher_main = module_from_spec(spec)
    spec.loader.exec_module(launcher_main)

    monkeypatch.setenv("MIAOSUAN_ROOT", "")
    monkeypatch.delenv("MIAOSUAN_BUNDLED", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    launcher_main._anchor_bundled_env()

    assert os.environ["MIAOSUAN_BUNDLED"] == "1"
    assert Path(os.environ["MIAOSUAN_ROOT"]).resolve() == config.REPO_ROOT.resolve()


def test_source_launcher_does_not_claim_bundled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """源码运行（未 frozen）不得置位 ``MIAOSUAN_BUNDLED``：那时 spawn 必须走 ``-m miaosuan.cli``。"""
    from importlib.util import module_from_spec, spec_from_file_location

    main_py = Path(__file__).resolve().parents[1] / "launcher" / "main.py"
    spec = spec_from_file_location("miaosuan_launcher_main_src", main_py)
    assert spec and spec.loader
    launcher_main = module_from_spec(spec)
    spec.loader.exec_module(launcher_main)

    monkeypatch.setenv("MIAOSUAN_ROOT", "")
    monkeypatch.delenv("MIAOSUAN_BUNDLED", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)

    launcher_main._anchor_bundled_env()

    assert "MIAOSUAN_BUNDLED" not in os.environ


def test_frozen_launcher_rejects_unknown_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """打包 exe 收到不认识的参数必须就地退出码 2，绝不落回启动器模式。

    落回启动器模式 = 在用户眼前弹 MessageBox、再起一个占 8686 的服务，而 UI 那一侧的
    子进程永远等不到退出码——「验证」按钮因此卡死（实测踩过）。
    """
    from importlib.util import module_from_spec, spec_from_file_location

    main_py = Path(__file__).resolve().parents[1] / "launcher" / "main.py"
    spec = spec_from_file_location("miaosuan_launcher_main_unknown", main_py)
    assert spec and spec.loader
    launcher_main = module_from_spec(spec)
    spec.loader.exec_module(launcher_main)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["exe", "-m", "miaosuan.cli", "verify", "--file", "x.py"])

    assert launcher_main.main() == 2
