#!/usr/bin/env python3
"""One-shot project cleanup: delete all junk files and temp directories.

Categories:
1. Root-level .txt/.log files (test output captures)
2. Root-level one-off scripts (build_*, check_*, compare_*, merge_*, test_duk*, etc.)
3. Temp scripts from recent work (clean_*, generate_*, finalize_*, make_fake*, etc.)
4. All .pytest_* directories (test artifacts)
5. .mypy_cache/, .ruff_cache/, __pycache__/
6. tmp/ directory contents (but keep the dir itself with .gitkeep)
"""
import shutil
from pathlib import Path

ROOT = Path(r"D:\backup\BaoBao\PythonProgram\miaosuan")

KEEP_FILES = {
    ".gitignore", ".pre-commit-config.yaml", "LICENSE", "Makefile",
    "pyproject.toml", "README.md", "requirements.lock", "settings.yaml",
    "start_miaosuan.bat", "launcher.py", "cleanup_project.py",
}

deleted_files = 0
deleted_dirs = 0
deleted_bytes = 0

# === 1. Delete root-level junk files ===
for f in ROOT.iterdir():
    if not f.is_file() or f.name in KEEP_FILES:
        continue
    if f.name.startswith(".git"):
        continue
    sz = f.stat().st_size
    f.unlink()
    deleted_files += 1
    deleted_bytes += sz

# === 2. Delete .pytest_* directories ===
for d in ROOT.iterdir():
    if d.is_dir() and d.name.startswith(".pytest"):
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
        deleted_dirs += 1
        deleted_bytes += sz

# === 3. Delete .mypy_cache/ and .ruff_cache/ ===
for cache in [".mypy_cache", ".ruff_cache", "__pycache__"]:
    d = ROOT / cache
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
        deleted_dirs += 1

# === 4. Clean tmp/ directory ===
tmp_dir = ROOT / "tmp"
if tmp_dir.is_dir():
    for f in tmp_dir.iterdir():
        if f.is_file():
            sz = f.stat().st_size
            f.unlink()
            deleted_files += 1
            deleted_bytes += sz
        elif f.is_dir():
            shutil.rmtree(f, ignore_errors=True)
            deleted_dirs += 1
    # Add .gitkeep
    gitkeep = tmp_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")

# === 5. Delete nested __pycache__ in src/ and tests/ ===
for pycache in ROOT.rglob("__pycache__"):
    if ".venv" in str(pycache) or ".git" in str(pycache):
        continue
    shutil.rmtree(pycache, ignore_errors=True)

print(f"Cleanup complete!")
print(f"  Files deleted: {deleted_files}")
print(f"  Directories deleted: {deleted_dirs}")
print(f"  Space freed: {deleted_bytes / 1024 / 1024:.1f} MB")
