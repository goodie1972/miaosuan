# 妙算（MiaoSuan）Makefile —— 本地与 CI 的统一入口
#
# 目标（架构 §2、T01 验收）：
#   make env    创建虚拟环境并安装依赖（新机器一次成功）
#   make smoke  最小自检（能 import 妙算包并跑通核心自检）
#   make test   运行全部测试
#   make parity 与冻结 AlphaMaster 的差分对拍（vocab 版本恒等）
#   make lint   静态检查（ruff + mypy）
#
# Windows 用 .venv/Scripts/python.exe，POSIX 用 .venv/bin/python，自动探测。

ifeq ($(OS),Windows_NT)
    PYTHON     := .venv/Scripts/python.exe
    PIP        := .venv/Scripts/python.exe -m pip
    VENV_FLAG  := .venv/Scripts/activate
else
    PYTHON     := .venv/bin/python
    PIP        := .venv/bin/python -m pip
    VENV_FLAG  := .venv/bin/activate
endif

# 与本机自带的 Python 3.13 对齐；若环境中有 3.11 可覆盖：make env PYTHON_BOOT=python3.11
PYTHON_BOOT ?= python

# Oracle python（装有真实 torch，用于重新生成冻结基准）；可用环境变量覆盖。
ORACLE_PYTHON ?= C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe

.DEFAULT_GOAL := help
.PHONY: help env smoke test parity lint format fixtures clean

help:
	@echo "妙算（MiaoSuan）可用目标："
	@echo "  make env     创建 .venv 并安装依赖（含 dev + pre-commit 钩子）"
	@echo "  make smoke   最小自检：import 妙算包 + vocab 版本自检"
	@echo "  make test    运行全部测试"
	@echo "  make parity  与冻结 AlphaMaster 的差分对拍"
	@echo "  make lint    ruff check + mypy"
	@echo "  make format  ruff format"
	@echo "  make fixtures 用 Oracle(torch) 重新生成冻结基准/清单（需 ORACLE_PYTHON）"
	@echo "  make clean   清理缓存（保留 .venv）"

# ── 环境 ───────────────────────────────────────────────────────────────────
env:
	$(PYTHON_BOOT) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	-$(PYTHON) -m pre_commit install
	@echo "环境就绪：$(VENV_FLAG)"

# ── 自检 ───────────────────────────────────────────────────────────────────
smoke:
	$(PYTHON) -m pytest -q tests/test_smoke.py

# ── 测试 ───────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests

parity:
	$(PYTHON) -m pytest tests/parity

# ── 静态检查 ───────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy

format:
	$(PYTHON) -m ruff format src tests

# ── 冻结基准（Oracle/torch 重新生成；日常不需要，仅当 AM 冻结快照更新时）────────
fixtures:
	$(ORACLE_PYTHON) scripts/gen_frozen_fixture.py
	$(ORACLE_PYTHON) scripts/gen_ops_baseline.py
	$(ORACLE_PYTHON) scripts/gen_feature_baseline.py
	$(ORACLE_PYTHON) scripts/gen_m5_baseline.py
	$(ORACLE_PYTHON) scripts/gen_e2e_xauusd_baseline.py

clean:
	-$(PYTHON) -c "import shutil,glob,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"
	-$(PYTHON) -c "import shutil;[shutil.rmtree(p,ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.mypy_cache','.hypothesis']]"
