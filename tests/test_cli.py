# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M14：CLI 四个子命令 —— mine / export / verify / report。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from miaosuan.cli import _extract_tokens, _history_payload, _write_history, app
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.ir.codec import write_spec
from miaosuan.ir.schema import (
    Evidence,
    FactorPayload,
    Provenance,
    Semantics,
    StrategySpec,
)

runner = CliRunner()

AM_BEST: tuple[int, ...] = (33, 62, 3, 87, 72, 119, 73, 103)


def _spec(verdict: str = "DEPLOYABLE") -> StrategySpec:
    return StrategySpec(
        name="h1_xauusd_miaosuan",
        payload=FactorPayload(tokens=AM_BEST, vocab_version=VOCAB_VERSION),
        semantics=Semantics(),
        evidence=Evidence(
            n_trials=3280,
            wf_folds=5,
            val_score=2.327,
            cost_sensitivity={"2.0": 1.95},
            deflated_sharpe=0.87,
            gate_verdict=verdict,
            gate_reasons=("DSR 0.62 < 0.95",) if verdict != "DEPLOYABLE" else (),
        ),
        provenance=Provenance(
            git_sha="ce83c09",
            vocab_version=VOCAB_VERSION,
            data_fingerprint="deadbeefcafe",
            seed=42,
            market="FOREX_XAUUSD",
            budget="standard",
        ),
    )


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    target = tmp_path / "spec.json"
    write_spec(target, _spec())
    return target


def _ga_history(n: int) -> list[object]:
    """造 n 代真实 :class:`GAStats`（用真类型，避免替身掩盖字段名写错）。"""
    from miaosuan.search.ga import GAStats

    return [
        GAStats(
            generation=g,
            best_fitness=1.0 + g,
            mean_fitness=0.5 + g * 0.5,
            diversity=0.4 - g * 0.05,
            population_size=100,
            n_evaluations=(g + 1) * 10,
            n_immigrants=0,
            n_diversity_rejects=0,
        )
        for g in range(n)
    ]


def _fake_result(n_generations: int) -> SimpleNamespace:
    """只带 :func:`_history_payload` 所需字段的结果替身。"""
    return SimpleNamespace(
        history=_ga_history(n_generations),
        stop_reason="MAX_GENERATIONS",
        generations=n_generations,
        n_evaluations=n_generations * 10,
    )


def test_history_payload_carries_real_generations() -> None:
    """逐代历史必须能被序列化出**非空**的真实序列（训练曲线的数据源）。

    回归防线：UI 的训练曲线只认这个结构。字段名一旦被改（如 best_fitness →
    best），前端拿到的是一堆 0，画出来的曲线看着"正常"但全是假的 —— 所以这里
    用真正的 :class:`GAStats` 实例来验，而不是字典替身。
    """
    payload = _history_payload(_fake_result(4), spec_out="artifacts/spec.json", budget="quick")

    assert payload["version"] >= 1
    assert payload["spec"] == "artifacts/spec.json"
    assert payload["budget"] == "quick"
    assert payload["generations"] == 4
    assert len(payload["points"]) == 4
    first, last = payload["points"][0], payload["points"][-1]
    assert first["generation"] == 0 and last["generation"] == 3
    assert last["best"] == 4.0 and last["mean"] == 2.0
    assert last["n_evaluations"] == 40
    # 多样性必须跟着走（图注里要报收敛情况）
    assert abs(last["diversity"] - 0.25) < 1e-9


def test_history_payload_degrades_to_empty_points() -> None:
    """无历史时 ``points`` 必须是空列表 —— 前端据此走空状态，不画空图。

    两种情况都要覆盖：history 是空列表；以及结果对象根本没有 history 字段
    （老版本 / 部分失败路径）。两者都不能抛异常。
    """
    empty = _history_payload(_fake_result(0), spec_out="s.json", budget="quick")
    assert empty["points"] == []

    no_attr = _history_payload(
        SimpleNamespace(stop_reason="WALL_CLOCK", generations=0, n_evaluations=0),
        spec_out="s.json",
        budget="quick",
    )
    assert no_attr["points"] == []
    assert no_attr["stop_reason"] == "WALL_CLOCK"


def test_write_history_creates_sidecar_json(tmp_path: Path) -> None:
    """sidecar 必须落成**可解析**的 JSON，且路径按 ``<out stem>.history.json`` 派生。"""
    target = tmp_path / "nested" / "spec.history.json"
    n = _write_history(target, _fake_result(3), spec_out="artifacts/spec.json", budget="standard")

    assert n == 3
    assert target.is_file()  # 父目录不存在时要自动创建
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert [p["generation"] for p in data["points"]] == [0, 1, 2]


def _csv(tmp_path: Path, n: int = 400, name: str = "XAUUSD_H1.csv") -> Path:
    """造一份可加载的 CSV 行情（确定性随机游走）。"""
    import pandas as pd

    rng = np.random.default_rng(11)
    steps = rng.normal(0.0, 0.002, size=n)
    close = 2000.0 * np.exp(np.cumsum(steps))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    frame = pd.DataFrame(
        {
            "time": 1_700_000_000 + np.arange(n) * 3600,
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_backtest_command_writes_result(tmp_path: Path, spec_file: Path) -> None:
    """``backtest`` 必须产出**可解析**的结果 JSON，且带口径提醒。"""
    data = _csv(tmp_path)
    out = tmp_path / "bt.json"
    result = runner.invoke(
        app,
        ["backtest", "--spec", str(spec_file), "--data", str(data), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["symbol"] == "XAUUSD"
    assert payload["meta"]["n_bars"] == 400
    # 成本率来自画像 CostModel，不是硬编码
    assert payload["meta"]["cost_rate"] == pytest.approx(0.0003)
    assert payload["series"]["equity"][0] == pytest.approx(1.0)
    assert len(payload["series"]["equity"]) == 400
    # 页面上绝不能把 val_score 当绩效：summary 里不该出现它
    assert "val_score" not in payload["summary"]
    assert "口径提醒" in result.output
    assert "不是绩效" in result.output


def test_backtest_rejects_non_factor_payload(tmp_path: Path) -> None:
    """非因子载荷（参数型）要明确报错，而不是回测出一条假曲线。"""
    from miaosuan.ir.schema import ParamPayload, Provenance, Semantics

    spec = StrategySpec(
        name="param_only",
        payload=ParamPayload(template_id="tpl_demo", params={"a": 1.0}),
        semantics=Semantics(),
        provenance=Provenance(git_sha="ce83c09", data_fingerprint="deadbeefcafe", seed=42),
    )
    spec_file = tmp_path / "param_spec.json"
    write_spec(spec_file, spec)

    result = runner.invoke(
        app,
        ["backtest", "--spec", str(spec_file), "--data", str(_csv(tmp_path, n=50))],
    )
    assert result.exit_code == 1
    assert "不是因子" in result.output


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("mine", "export", "verify", "report"):
        assert name in result.output


def test_export_command(tmp_path: Path, spec_file: Path) -> None:
    out_dir = tmp_path / "strategies"
    result = runner.invoke(
        app,
        [
            "export",
            "--spec",
            str(spec_file),
            "--out-dir",
            str(out_dir),
            "--ledger",
            str(tmp_path / "magic.json"),
            "--date",
            "20260910",
        ],
    )
    assert result.exit_code == 0, result.output
    produced = list(out_dir.glob("*.py"))
    assert len(produced) == 1
    assert produced[0].name == "20260910_h1_xauusd_miaosuan_v1.py"
    assert "661801" in result.output
    assert "0 ERROR" in result.output


def test_export_writes_magic_ledger(tmp_path: Path, spec_file: Path) -> None:
    ledger = tmp_path / "magic.json"
    runner.invoke(
        app,
        ["export", "--spec", str(spec_file), "--out-dir", str(tmp_path), "--ledger", str(ledger)],
    )
    assert ledger.exists()
    assert "661801" in ledger.read_text(encoding="utf-8")


def test_export_respects_name_override(tmp_path: Path, spec_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export",
            "--spec",
            str(spec_file),
            "--out-dir",
            str(tmp_path),
            "--ledger",
            str(tmp_path / "m.json"),
            "--date",
            "20260910",
            "--name",
            "renamed_strategy",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "20260910_renamed_strategy_v1.py").exists()


def test_export_gate_deadband_switch(tmp_path: Path, spec_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export",
            "--spec",
            str(spec_file),
            "--out-dir",
            str(tmp_path),
            "--ledger",
            str(tmp_path / "m.json"),
            "--gate-deadband",
            "0.05",
        ],
    )
    assert result.exit_code == 0, result.output
    written = next(tmp_path.glob("*.py")).read_text(encoding="utf-8")
    assert "_GATE_DEADBAND = 0.05" in written


def test_verify_command_on_exported_file(tmp_path: Path, spec_file: Path) -> None:
    export = runner.invoke(
        app,
        [
            "export",
            "--spec",
            str(spec_file),
            "--out-dir",
            str(tmp_path),
            "--ledger",
            str(tmp_path / "m.json"),
        ],
    )
    assert export.exit_code == 0
    target = next(tmp_path.glob("*.py"))
    result = runner.invoke(app, ["verify", "--file", str(target)])
    assert result.exit_code == 0, result.output
    assert "0 ERROR" in result.output
    assert "跳过数值保真度回归" in result.output


def test_verify_rejects_repaint_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad_strategy.py"
    bad.write_text(
        'STRATEGY_MAGIC = "661801"
'
        'STRATEGY_NAME = "bad"
'
        'STRATEGY_VERSION = "1"
'
        'STRATEGY_CHANGELOG = ("v1",)
'
        "def generate_signal(candles):
"
        "    return candles[-1]
",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["verify", "--file", str(bad)])
    assert result.exit_code == 1
    assert "AF001" in result.output


def test_report_command(spec_file: Path) -> None:
    result = runner.invoke(app, ["report", "--spec", str(spec_file)])
    assert result.exit_code == 0, result.output
    assert "DEPLOYABLE" in result.output
    assert VOCAB_VERSION in result.output
    assert "未消费" in result.output
    assert "ce83c09" in result.output


def test_report_shows_gate_reasons(tmp_path: Path) -> None:
    target = tmp_path / "spec.json"
    write_spec(target, _spec(verdict="RESEARCH_ONLY"))
    result = runner.invoke(app, ["report", "--spec", str(target)])
    assert result.exit_code == 0
    assert "RESEARCH_ONLY" in result.output
    assert "DSR 0.62 < 0.95" in result.output


def test_mine_rejects_unknown_budget(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["mine", "--data", str(tmp_path / "nope.csv"), "--budget", "turbo"]
    )
    assert result.exit_code == 1
    assert "未知预算档位" in result.output


def test_extract_tokens_helper() -> None:
    assert _extract_tokens("_TOKENS = (33, 62, 3)
") == (33, 62, 3)
    assert _extract_tokens("_TOKENS = ()
") == ()
    assert _extract_tokens("nothing here
") == ()
