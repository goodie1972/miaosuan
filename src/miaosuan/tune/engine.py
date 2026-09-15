# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""参数寻优引擎（Optuna TPE 采样）。

职责
----
* 接收 :class:`TuneConfig` + 因子 token + 行情数据路径；
* 用 Optuna TPE 在 ``ParamSpace`` 定义的范围内采样；
* 每次试验调用 :func:`evaluate_trial` 执行全样本回测；
* 按综合评分排序，返回最优参数组合。

依赖方向
--------
``tune.engine`` ← ``ir``, ``adapters.base``, ``report.equity``, ``core.signal``, ``data.loader``, ``market.profiles``
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.base import ParamSpace
from ..data.loader import load
from ..ir.schema import ParamPayload, StrategySpec
from ..market.profiles import get_profile
from ..report.equity import BacktestRun, run_full_backtest

__all__ = [
    "TrialResult",
    "TuneConfig",
    "TuneEngine",
    "TuneResult",
    "evaluate_trial",
    "load_param_space_from_spec",
    "save_tune_result",
    "score_trial",
]


# ── 数据类 ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrialResult:
    """一次参数组合的回测结果。

    Attributes:
        trial_id: 本次试验的唯一标识（UUID 短串）。
        params: 本次使用的参数值（``{参数名: 数值}``）。
        score: 综合评分（由 :func:`score_trial` 计算）。
        backtest: 回测运行结果（含资金曲线 / 指标 / 交易统计）。
    """

    trial_id: str
    params: dict[str, float]
    score: float
    backtest: BacktestRun | None = None


@dataclass
class TuneConfig:
    """参数寻优任务配置。

    Attributes:
        spec_path: 源策略 spec JSON 路径。
        param_spaces: 可调参数空间列表。
        n_trials: 最大试验次数（优化预算）。
        metric: 优化目标指标名（``"sharpe"`` / ``"sortino"`` / ``"calmar"`` / ``"composite"``）。
        composite_weights: 综合评分权重。
        out_dir: 结果落盘目录（默认 ``artifacts/tune/``）。
        seed: 随机种子（保证可复现）。
        paper_trading_path: 实盘模拟持仓记录路径（可选）。
    """

    spec_path: str
    param_spaces: list[ParamSpace]
    n_trials: int = 50
    metric: str = "sharpe"
    composite_weights: dict[str, float] = field(default_factory=dict)
    out_dir: str = "artifacts/tune"
    seed: int = 42
    paper_trading_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "spec_path": self.spec_path,
            "param_spaces": [p.to_dict() for p in self.param_spaces],
            "n_trials": self.n_trials,
            "metric": self.metric,
            "composite_weights": dict(self.composite_weights),
            "out_dir": self.out_dir,
            "seed": self.seed,
            "paper_trading_path": self.paper_trading_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TuneConfig:
        """从字典还原。"""
        raw_spaces = data.get("param_spaces") or []
        return cls(
            spec_path=str(data["spec_path"]),
            param_spaces=[ParamSpace.from_dict(s) for s in raw_spaces],
            n_trials=int(data.get("n_trials", 50)),
            metric=str(data.get("metric", "sharpe")),
            composite_weights={
                str(k): float(v) for k, v in (data.get("composite_weights") or {}).items()
            },
            out_dir=str(data.get("out_dir", "artifacts/tune")),
            seed=int(data.get("seed", 42)),
            paper_trading_path=str(data.get("paper_trading_path", "")),
        )


@dataclass
class TuneResult:
    """参数寻优最终结果。

    Attributes:
        result_id: 本次寻优任务唯一标识。
        config: 使用的配置快照。
        trials: 所有试验结果（按评分降序排列）。
        best: 最优参数组合。
        elapsed_sec: 总耗时（秒）。
    """

    result_id: str
    config: TuneConfig
    trials: list[TrialResult]
    best: TrialResult
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "result_id": self.result_id,
            "config": self.config.to_dict(),
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "params": dict(t.params),
                    "score": float(t.score),
                    "sharpe": float(t.backtest.metrics.sharpe) if t.backtest and t.backtest.metrics else None,
                    "sortino": float(t.backtest.metrics.sortino) if t.backtest and t.backtest.metrics else None,
                    "max_drawdown": float(t.backtest.metrics.max_drawdown) if t.backtest and t.backtest.metrics else None,
                }
                for t in self.trials[:20]
            ],
            "best": {
                "trial_id": self.best.trial_id,
                "params": dict(self.best.params),
                "score": float(self.best.score),
            },
            "elapsed_sec": round(self.elapsed_sec, 2),
            "n_trials": len(self.trials),
        }


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_param_space_from_spec(spec_path: str) -> list[ParamSpace]:
    """从已导出策略 spec 中提取 ``ParamSpace`` 定义（若有）。

    Args:
        spec_path: StrategySpec JSON 文件路径。

    Returns:
        参数空间列表（当前为空，预留扩展点）。
    """
    try:
        spec = StrategySpec.from_dict(json.loads(Path(spec_path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    return []


def score_trial(
    backtest: BacktestRun,
    metric: str = "sharpe",
    weights: dict[str, float] | None = None,
) -> float:
    """计算单次试验的综合评分。

    Args:
        backtest: 回测结果。
        metric: 指标名（``"sharpe"`` / ``"sortino"`` / ``"calmar"`` / ``"composite"``）。
        weights: 综合权重（仅 ``"composite"`` 时使用）。

    Returns:
        标量评分（越高越好）。
    """
    if backtest is None or backtest.metrics is None:
        return -1e9

    metrics = backtest.metrics
    w = weights or {"sharpe": 0.6, "drawdown_penalty": 0.4}

    if metric == "sharpe":
        return float(metrics.sharpe)
    if metric == "sortino":
        return float(metrics.sortino)
    if metric == "calmar":
        return float(metrics.calmar)
    if metric == "composite":
        sharpe = float(metrics.sharpe)
        mdd = float(metrics.max_drawdown) if metrics.max_drawdown else 0.0
        dd_pen = min(mdd / 100.0, 1.0)
        return float(w.get("sharpe", 0.6) * sharpe - w.get("drawdown_penalty", 0.4) * dd_pen)

    raise ValueError(f"未知指标 {metric!r}，可选：sharpe / sortino / calmar / composite")


def evaluate_trial(
    tokens: tuple[int, ...],
    params: dict[str, float],
    spec: StrategySpec,
    data_path: str,
) -> BacktestRun:
    """执行一次回测试验。

    Args:
        tokens: 因子 token 序列。
        params: 当前试验的参数值。
        spec: 源策略 spec。
        data_path: 行情数据文件路径。

    Returns:
        回测结果。
    """
    panel = load(data_path)
    profile_name = spec.provenance.market or panel.market_profile_name or "forex"
    profile = get_profile(profile_name)

    semantics = spec.semantics
    if "neutral_band" in params:
        semantics = type(semantics)(
            **{**semantics.to_dict(), "neutral_band": params["neutral_band"]}
        )

    return run_full_backtest(
        tokens, panel, profile,
        semantics=semantics,
    )


def save_tune_result(result: TuneResult, path: str | Path) -> None:
    """将寻优结果持久化到 JSON 文件。

    Args:
        result: 寻优结果对象。
        path: 输出路径。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ── 引擎 ─────────────────────────────────────────────────────────────────────

class TuneEngine:
    """参数寻优引擎（Optuna TPE + 回测评估）。

    使用 Optuna 的 Tree-structured Parzen Estimator (TPE) 采样，
    在给定参数空间中搜索最优参数组合。
    """

    def __init__(self, config: TuneConfig) -> None:
        self.config = config
        self.study: Any = None
        self.trials: list[TrialResult] = []

    def _define_param_space(self, trial: Any) -> dict[str, Any]:
        """为 Optuna trial 定义参数空间。"""
        params: dict[str, Any] = {}
        for space in self.config.param_spaces:
            if space.kind == "float":
                low = space.low or (space.default * 0.5)
                high = space.high or (space.default * 2.0)
                params[space.name] = trial.suggest_float(space.name, low, high)
            elif space.kind == "int":
                low = int(space.low or (space.default * 0.5))
                high = int(space.high or (space.default * 2.0))
                params[space.name] = trial.suggest_int(space.name, low, high)
            elif space.kind == "choice":
                params[space.name] = trial.suggest_categorical(space.name, list(space.choices))
            elif space.kind == "bool":
                params[space.name] = trial.suggest_categorical(space.name, [False, True])
            else:
                params[space.name] = space.default
        return params

    def optimize(self, tokens: tuple[int, ...], data_path: str, spec: StrategySpec) -> TuneResult:
        """执行参数寻优。

        Args:
            tokens: 因子 token 序列。
            data_path: 行情数据文件路径。
            spec: 源策略 spec。

        Returns:
            寻优结果。
        """
        import optuna

        result_id = uuid.uuid4().hex[:8]
        start_time = time.time()

        direction = "MAXIMIZE"
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
            study_name=f"miaosuan-tune-{result_id}",
        )

        def objective(trial: Any) -> float:
            params = self._define_param_space(trial)
            try:
                bt = evaluate_trial(tokens, params, spec, data_path)
                score = score_trial(bt, self.config.metric, self.config.composite_weights)
                return score
            except Exception as exc:  # noqa: BLE001
                return -1e9

        self.study.optimize(objective, n_trials=self.config.n_trials)

        elapsed = time.time() - start_time
        trials = []
        for i, trial in enumerate(self.study.trials):
            if trial.state.is_finished():
                trials.append(TrialResult(
                    trial_id=f"t{i:04d}",
                    params=dict(trial.params),
                    score=trial.value or 0.0,
                    backtest=None,
                ))

        trials.sort(key=lambda t: t.score, reverse=True)
        best = trials[0] if trials else TrialResult(
            trial_id="none",
            params={},
            score=-1e9,
            backtest=None,
        )

        result = TuneResult(
            result_id=result_id,
            config=self.config,
            trials=trials,
            best=best,
            elapsed_sec=elapsed,
        )

        out_path = Path(self.config.out_dir) / f"tune_{result_id}.json"
        save_tune_result(result, out_path)

        return result
