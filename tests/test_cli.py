"""M14：CLI 四个子命令 —— mine / export / verify / report。"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from miaosuan.cli import _extract_tokens, app
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
        'STRATEGY_MAGIC = "661801"\n'
        'STRATEGY_NAME = "bad"\n'
        'STRATEGY_VERSION = "1"\n'
        'STRATEGY_CHANGELOG = ("v1",)\n'
        "def generate_signal(candles):\n"
        "    return candles[-1]\n",
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
    assert _extract_tokens("_TOKENS = (33, 62, 3)\n") == (33, 62, 3)
    assert _extract_tokens("_TOKENS = ()\n") == ()
    assert _extract_tokens("nothing here\n") == ()
