"""最小自检（``make smoke``）。

断言：妙算包可导入、核心 vocab 可派生、配置与日志基础设施可工作。
不依赖 numpy/torch，纯标准库即可通过，保证新机器 ``make env && make smoke`` 一次成功。
"""

from __future__ import annotations

import io
import json

import miaosuan
from miaosuan.config import AppConfig, BudgetConfig
from miaosuan.errors import ConfigError, MiaoSuanError
from miaosuan.logging_setup import JsonLineFormatter, configure_logging, new_run_id

FROZEN_VERSION = "v9217a2c0d91a"


def test_package_imports() -> None:
    assert miaosuan.__version__
    assert miaosuan.TOOL_NAME == "miaosuan"


def test_core_vocab_selfcheck() -> None:
    from miaosuan.core.vocab import FORMULA_VOCAB, VOCAB_VERSION

    assert VOCAB_VERSION == FROZEN_VERSION
    assert FORMULA_VOCAB.feature_count == 65
    assert len(FORMULA_VOCAB.operator_names) == 62
    assert FORMULA_VOCAB.size == 127


def test_config_defaults_and_snapshot() -> None:
    cfg = AppConfig()
    snap = cfg.to_snapshot()
    assert snap["seed"] == 20260910
    assert snap["search"]["formula_len"] == 8
    # 预算档位
    assert BudgetConfig.standard().wall_clock_hours == 2.0


def test_config_rejects_unknown_key() -> None:
    try:
        AppConfig.from_mapping({"nope": 1})
    except ConfigError as exc:
        assert exc.code == "E-CONFIG"
    else:  # pragma: no cover
        raise AssertionError("未知配置键应抛 ConfigError")


def test_error_hierarchy() -> None:
    assert issubclass(ConfigError, MiaoSuanError)
    err = ConfigError("坏配置", context={"k": 1})
    assert err.to_dict()["code"] == "E-CONFIG"


def test_logging_json_line_has_run_id() -> None:
    run_id = new_run_id()
    buf = io.StringIO()
    logger = configure_logging(run_id=run_id, stream=buf)

    import logging

    record = logging.LogRecord("miaosuan.test", logging.INFO, __file__, 1, "hello", None, None)
    record.run_id = run_id
    formatter = JsonLineFormatter(run_id=run_id)
    buf.write(formatter.format(record) + "\n")
    logger.info("world")

    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    payloads = [json.loads(line) for line in lines]
    assert all(p.get("run_id") == run_id for p in payloads)
    assert any(p["msg"] == "hello" for p in payloads)
    assert any(p["msg"] == "world" for p in payloads)
