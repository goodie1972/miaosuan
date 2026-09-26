# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M14：CLI 四个子命令 —— mine / export / verify / report。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from miaosuan.cli import (
    _extract_tokens,
    _history_payload,
    _resolve_search_space,
    _write_candidates,
    _write_history,
    app,
)
from miaosuan.config import AppConfig
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.data.loader import load
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


def test_import_command(tmp_path: Path, spec_file: Path) -> None:
    """Test the new import command that extracts StrategySpec from exported .py file."""
    # First export a spec to a .py file
    out_dir = tmp_path / "strategies"
    result = runner.invoke(
        app,
        [
            "export",
            "--spec",
            str(spec_file),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    
    # Get the generated .py file
    strategy_file = next(out_dir.glob("*.py"))
    
    # Now test importing it back
    result = runner.invoke(
        app,
        [
            "import",
            "--file",
            str(strategy_file),
        ],
    )
    assert result.exit_code == 0, result.output
    
    # The output should be valid JSON matching the original spec
    import json
    imported_spec = json.loads(result.output.strip())
    original_spec = json.loads(spec_file.read_text(encoding="utf-8"))
    
    # Compare key fields (excluding generated_at which will differ)
    assert imported_spec["spec_version"] == original_spec["spec_version"]
    assert imported_spec["name"] == original_spec["name"]
    assert imported_spec["payload"]["kind"] == original_spec["payload"]["kind"]
    assert imported_spec["payload"]["tokens"] == original_spec["payload"]["tokens"]
    assert imported_spec["payload"]["vocab_version"] == original_spec["payload"]["vocab_version"]
    
    # Test with --out option
    out_spec_file = tmp_path / "imported_spec.json"
    result = runner.invoke(
        app,
        [
            "import",
            "--file",
            str(strategy_file),
            "--out",
            str(out_spec_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_spec_file.exists()
    
    # Verify the written file matches
    written_spec = json.loads(out_spec_file.read_text(encoding="utf-8"))
    assert written_spec["spec_version"] == original_spec["spec_version"]
    assert written_spec["name"] == original_spec["name"]


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


# ── mine 搜索空间覆盖：--pop-size / --elite-size / --formula-len / --n-islands ──


class _StopMine(Exception):
    """run_mine 桩的提前终止信号（不跑真 GA，也不需要真 spec 产物）。"""


class _RecordingRunMine:
    """记录 ``run_mine`` 收到的 ``config`` / ``budget`` 后立刻终止。

    覆盖是否**真的生效**只能在这一层验：``run_mine`` 收到的 ``budget`` 就是
    ``search.mine`` 里 ``RpnGA.run`` 读的那份（``config.search.pop_size`` 运行时
    不被读取）。跑完整 GA 太慢，故在此截断。
    """

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, panel: Any, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        raise _StopMine


def _capture_run_mine(monkeypatch: pytest.MonkeyPatch) -> _RecordingRunMine:
    recorder = _RecordingRunMine()
    monkeypatch.setattr("miaosuan.cli.run_mine", recorder)
    return recorder


def test_search_space_defaults_are_not_overridden() -> None:
    """不传任何覆盖参数 → config 与预算档位表都保持默认（0 = 不覆盖）。"""
    config, budget = _resolve_search_space(
        AppConfig(),
        budget_profile="standard",
        pop_size=0,
        elite_size=0,
        formula_len=0,
        n_islands=0,
    )
    # config 线：GASearchConfig 默认
    assert config.search.pop_size == 256
    assert config.search.elite_size == 32
    assert config.search.formula_len == 8
    assert config.search.n_islands == 4
    # 运行线：standard 档位表
    assert (budget.pop_size, budget.elite_size, budget.island_count) == (256, 32, 4)
    assert budget.profile == "standard"


def test_search_space_overrides_apply_to_both_lines() -> None:
    """覆盖值必须同时落到 config.search（进 snapshot）与 Budget（真跑的那份）。"""
    config, budget = _resolve_search_space(
        AppConfig(),
        budget_profile="deep",
        pop_size=512,
        elite_size=64,
        formula_len=12,
        n_islands=2,
    )
    assert (config.search.pop_size, config.search.elite_size) == (512, 64)
    assert config.search.formula_len == 12
    assert config.search.n_islands == 2
    # 运行时这份必须被改写（否则只写进产物、实际仍是 deep 档 384）
    assert (budget.pop_size, budget.elite_size, budget.island_count) == (512, 64, 2)
    assert budget.profile == "deep"
    # 产物 config_snapshot 必须如实反映覆盖后的值
    snapshot = config.to_snapshot()
    assert snapshot["search"]["pop_size"] == 512
    assert snapshot["search"]["elite_size"] == 64
    assert snapshot["search"]["formula_len"] == 12
    assert snapshot["search"]["n_islands"] == 2


def test_search_space_partial_override_keeps_profile_values() -> None:
    """只覆盖 pop 时：elite/islands 仍用档位表；config 未被指定的字段不动。"""
    config, budget = _resolve_search_space(
        AppConfig(),
        budget_profile="quick",
        pop_size=200,
        elite_size=0,
        formula_len=0,
        n_islands=0,
    )
    assert budget.pop_size == 200
    assert budget.elite_size == 12  # quick 档位表
    assert budget.island_count == 1  # quick 档位表（1 = 关闭分岛）
    assert config.search.pop_size == 200
    assert config.search.formula_len == 8  # 未指定 → 不覆盖


def test_search_space_rejects_elite_greater_than_pop() -> None:
    with pytest.raises(typer.Exit):
        _resolve_search_space(
            AppConfig(),
            budget_profile="standard",
            pop_size=16,
            elite_size=32,
            formula_len=0,
            n_islands=0,
        )


def test_search_space_rejects_islands_greater_than_pop() -> None:
    with pytest.raises(typer.Exit):
        _resolve_search_space(
            AppConfig(),
            budget_profile="standard",
            pop_size=4,
            elite_size=2,
            formula_len=0,
            n_islands=8,
        )


def test_search_space_rejects_negative_values() -> None:
    defaults: dict[str, int] = {
        "pop_size": 0,
        "elite_size": 0,
        "formula_len": 0,
        "n_islands": 0,
    }
    for flag in defaults:
        with pytest.raises(typer.Exit):
            _resolve_search_space(
                AppConfig(),
                budget_profile="standard",
                **{**defaults, flag: -1},
            )


def test_mine_cli_overrides_reach_run_mine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI 覆盖参数要原样传到 ``run_mine(config=..., budget=...)``。"""
    recorder = _capture_run_mine(monkeypatch)
    data = _csv(tmp_path)
    result = runner.invoke(
        app,
        [
            "mine",
            "--data",
            str(data),
            "--budget",
            "deep",
            "--pop-size",
            "512",
            "--elite-size",
            "64",
            "--formula-len",
            "12",
            "--n-islands",
            "2",
            "--out",
            str(tmp_path / "spec.json"),
        ],
    )
    # 桩 raise _StopMine → CliRunner 捕获为非 0 退出码；关键是 kwargs 已记录。
    assert recorder.kwargs is not None, result.output
    config = recorder.kwargs["config"]
    budget = recorder.kwargs["budget"]
    assert (budget.pop_size, budget.elite_size, budget.island_count) == (512, 64, 2)
    assert (config.search.pop_size, config.search.elite_size) == (512, 64)
    assert config.search.formula_len == 12
    assert config.search.n_islands == 2
    assert recorder.kwargs["budget_profile"] == "deep"


def test_mine_cli_without_overrides_passes_profile_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不传覆盖参数 → 传下去的是档位表预算，config 保持 GASearchConfig 默认。"""
    recorder = _capture_run_mine(monkeypatch)
    result = runner.invoke(
        app,
        ["mine", "--data", str(_csv(tmp_path)), "--budget", "deep",
         "--out", str(tmp_path / "spec.json")],
    )
    assert recorder.kwargs is not None, result.output
    config = recorder.kwargs["config"]
    budget = recorder.kwargs["budget"]
    assert (budget.pop_size, budget.elite_size, budget.island_count) == (384, 48, 4)
    # config 线不受档位影响：pop 仍是 GASearchConfig 默认（两条线互不覆盖的证据）
    assert (config.search.pop_size, config.search.formula_len) == (256, 8)


def test_mine_rejects_elite_gt_pop_before_loading_data(tmp_path: Path) -> None:
    """非法组合必须在**读数据之前**报错（文件不存在也能先给出参数错误）。"""
    result = runner.invoke(
        app,
        ["mine", "--data", str(tmp_path / "nope.csv"), "--pop-size", "16",
         "--elite-size", "32"],
    )
    assert result.exit_code == 1
    assert "不得大于" in result.output
    assert "无法加载" not in result.output


def test_mine_rejects_negative_override_flag(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["mine", "--data", str(tmp_path / "nope.csv"), "--pop-size=-1"]
    )
    assert result.exit_code == 1
    assert "--pop-size" in result.output
    assert "不覆盖" in result.output


def test_mine_rejects_islands_exceeding_pop(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["mine", "--data", str(tmp_path / "nope.csv"), "--pop-size", "4",
         "--elite-size", "2", "--n-islands", "8"],
    )
    assert result.exit_code == 1
    assert "--n-islands" in result.output


# ── mine --dump-candidates：top-k 候选落盘 sidecar ────────────────────────────


def _fake_candidates() -> list[Any]:
    """两个结构真实的 :class:`Candidate`（真类型，避免替身掩盖字段名写错）。"""
    from miaosuan.search.mine import Candidate

    return [
        Candidate(
            tokens=(33, 62, 3, 87, 72, 119, 73, 103),
            decoded="A → B",
            train_score=1.5,
            val_score=1.25,
            status="ok",
            sharpes={1.0: 2.0, 2.0: 1.1},
            sharpe_2x=1.1,
            wf_win_rate=0.8,
            dsr=0.62,
            verdict="DEPLOYABLE",
            verdict_snapshot='{"hard_failures": [], "soft_failures": [], "status": "DEPLOYABLE"}',
        ),
        Candidate(
            tokens=(1, 2, 3, 4, 5, 6, 7, 8),
            decoded="C",
            train_score=0.9,
            val_score=0.5,
            status="ok",
            sharpes={},
            sharpe_2x=0.1,
            wf_win_rate=0.4,
            dsr=0.2,
            verdict="RESEARCH_ONLY",
            verdict_snapshot="not-a-json",
        ),
    ]


class _FakeOutcomeRunMine:
    """返回可走完整 ``mine`` 流程的假 outcome（真 spec + 真候选，不跑 GA）。"""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, panel: Any, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        from miaosuan.search.mine import MineResult

        self.kwargs = kwargs
        result = MineResult(
            candidates=_fake_candidates(),
            best=_fake_candidates()[0],
            split=SimpleNamespace(),
            stop_reason="MAX_GENERATIONS",
            generations=3,
            n_evaluations=30,
            initial_diversity=0.5,
            min_diversity=0.2,
            dev_bars=300,
            n_bars=400,
            wall_clock_seconds=1.0,
        )
        return SimpleNamespace(
            result=result,
            spec=_spec(),
            profile=SimpleNamespace(name="FOREX_XAUUSD"),
        )


def _capture_full_run(monkeypatch: pytest.MonkeyPatch) -> _FakeOutcomeRunMine:
    stub = _FakeOutcomeRunMine()
    monkeypatch.setattr("miaosuan.cli.run_mine", stub)
    return stub


def test_dump_candidates_writes_sidecar_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--dump-candidates <path>`` 落盘 top-k：结构含溯源字段 + 逐条候选。"""
    _capture_full_run(monkeypatch)
    data = _csv(tmp_path)
    out = tmp_path / "spec.json"
    cand_path = tmp_path / "nested" / "cands.json"  # 父目录须自动创建
    result = runner.invoke(
        app,
        ["mine", "--data", str(data), "--out", str(out),
         "--dump-candidates", str(cand_path)],
    )
    assert result.exit_code == 0, result.output
    assert cand_path.is_file()

    payload = json.loads(cand_path.read_text(encoding="utf-8"))
    # 溯源字段：把候选和这次运行对上（下游多因子组合要用）
    assert payload["version"] == 1
    assert payload["budget"] == "standard"
    assert payload["symbol"] == "XAUUSD"
    assert payload["timeframe"] == "H1"
    assert payload["market"] == "FOREX_XAUUSD"
    assert payload["spec"] == str(out)
    assert payload["n_candidates"] == 2
    # combo 的输入契约四字段（docs/combo-impl-spec.md §2）
    assert payload["vocab_version"] == VOCAB_VERSION
    assert payload["vocab_version"] == json.loads(out.read_text(encoding="utf-8"))[
        "payload"
    ]["vocab_version"]
    assert payload["data_fingerprint"] == load(str(data)).fingerprint
    assert payload["n_trials"] == 30  # = result.n_evaluations（Evidence.n_trials 同源）
    assert payload["semantics"] == {
        "timeframe": "H1",
        "position_fn": "tanh",
        "neutral_band": 0.05,
        "long_short": True,
    }

    got = payload["candidates"]
    assert [c["rank"] for c in got] == [0, 1]
    assert got[0]["tokens"] == [33, 62, 3, 87, 72, 119, 73, 103]
    assert got[0]["val_score"] == pytest.approx(1.25)
    assert got[0]["dsr"] == pytest.approx(0.62)
    assert got[0]["verdict"] == "DEPLOYABLE"
    # 快照 JSON 字符串须被解析成**对象**（下游不必二次 json.loads）
    assert isinstance(got[0]["verdict_snapshot"], dict)
    assert got[0]["verdict_snapshot"]["status"] == "DEPLOYABLE"
    assert got[0]["cost_sensitivity"] == {"1": 2.0, "2": 1.1}
    # 快照损坏 → 保留原文字符串，绝不丢整条候选
    assert got[1]["verdict_snapshot"] == "not-a-json"
    assert "候选落盘：2 条" in result.output


def test_dump_candidates_not_requested_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不传 ``--dump-candidates`` → 一个候选文件都不写（默认行为零变化）。"""
    _capture_full_run(monkeypatch)
    out = tmp_path / "spec.json"
    result = runner.invoke(
        app, ["mine", "--data", str(_csv(tmp_path)), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert not list(tmp_path.glob("*.candidates.json"))
    assert not list(tmp_path.glob("cands*.json"))
    assert "候选落盘" not in result.output
    # spec / history 两个既有 sidecar 不受影响
    assert out.is_file()
    assert (tmp_path / "spec.history.json").is_file()


def test_write_candidates_keeps_entry_on_broken_snapshot(tmp_path: Path) -> None:
    """非 JSON 快照 → 条目照写、原样保留字符串（失败可见且不丢数据）。"""
    path = tmp_path / "c.json"
    n = _write_candidates(
        path, SimpleNamespace(candidates=_fake_candidates()),
        spec_out="s.json", spec_id="abc", budget="quick",
        symbol="XAUUSD", timeframe="H1", market="FOREX_XAUUSD", git_sha="deadbeef",
        vocab_version=VOCAB_VERSION, data_fingerprint="f" * 64,
        semantics={"timeframe": "H1", "position_fn": "tanh",
                   "neutral_band": 0.05, "long_short": True},
        n_trials=3280,
    )
    assert n == 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["spec_id"] == "abc"
    assert payload["git_sha"] == "deadbeef"
    assert payload["candidates"][1]["verdict_snapshot"] == "not-a-json"
    # §2 四字段在直接调用路径同样落盘（combine 的校验输入）
    assert payload["vocab_version"] == VOCAB_VERSION
    assert payload["data_fingerprint"] == "f" * 64
    assert payload["n_trials"] == 3280
    assert payload["semantics"]["neutral_band"] == 0.05


def test_write_candidates_writes_empty_shell_when_no_candidates(
    tmp_path: Path,
) -> None:
    """0 候选也写空壳文件：下游据此区分「跑过但没产出」与「压根没跑」。"""
    path = tmp_path / "c.json"
    n = _write_candidates(
        path, SimpleNamespace(candidates=[]),
        spec_out="s.json", spec_id="x", budget="quick",
        symbol="XAUUSD", timeframe="H1", market="FOREX_XAUUSD", git_sha="g",
        vocab_version=VOCAB_VERSION, data_fingerprint="",
        semantics={"timeframe": "H1", "position_fn": "tanh",
                   "neutral_band": 0.05, "long_short": True},
        n_trials=0,
    )
    assert n == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_candidates"] == 0
    assert payload["candidates"] == []
    assert payload["n_trials"] == 0


# ── mine 标签派生：timeframe / symbol 不许再标错 ──────────────────────────────


def test_mine_derives_timeframe_and_symbol_from_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不带 ``--timeframe/--symbol`` → 从数据文件名推导（不再恒标 h1/xauusd）。"""
    recorder = _capture_run_mine(monkeypatch)
    data = _csv(tmp_path, name="BTCUSDT_M15.csv")
    result = runner.invoke(
        app, ["mine", "--data", str(data), "--out", str(tmp_path / "s.json")]
    )
    assert recorder.kwargs is not None, result.output
    config = recorder.kwargs["config"]
    assert config.timeframe == "M15"
    assert config.symbol == "BTCUSDT"
    assert "标签：BTCUSDT M15" in result.output
    assert "数据文件名" in result.output


def test_mine_explicit_labels_beat_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI 显式 ``--timeframe/--symbol`` 压过文件名（显式意图 > 自动推断）。"""
    recorder = _capture_run_mine(monkeypatch)
    data = _csv(tmp_path, name="BTCUSDT_M15.csv")
    result = runner.invoke(
        app,
        ["mine", "--data", str(data), "--out", str(tmp_path / "s.json"),
         "--timeframe", "H4", "--symbol", "ETHUSDT"],
    )
    assert recorder.kwargs is not None, result.output
    config = recorder.kwargs["config"]
    assert config.timeframe == "H4"
    assert config.symbol == "ETHUSDT"
    assert "CLI 显式" in result.output


def test_mine_env_label_beats_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """env 显式设置（值≠配置默认）压过文件名，且不违反「env 只在 config 读」铁律。"""
    recorder = _capture_run_mine(monkeypatch)
    monkeypatch.setenv("MIAOSUAN_TIMEFRAME", "M30")
    monkeypatch.setenv("MIAOSUAN_SYMBOL", "GBPUSD")
    data = _csv(tmp_path, name="BTCUSDT_M15.csv")
    result = runner.invoke(
        app, ["mine", "--data", str(data), "--out", str(tmp_path / "s.json")]
    )
    assert recorder.kwargs is not None, result.output
    config = recorder.kwargs["config"]
    assert config.timeframe == "M30"
    assert config.symbol == "GBPUSD"
    assert "env 显式" in result.output
