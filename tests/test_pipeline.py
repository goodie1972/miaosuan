# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M14 支撑：编排层的市场画像解析与 verdict 快照解析。"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.config import ConfigError
from miaosuan.data.panel import Panel
from miaosuan.pipeline import _gate_reasons, resolve_market


def _panel(profile_name: str = "") -> Panel:
    arr = np.zeros((1, 5), dtype=np.float32)
    return Panel(
        open=arr.copy(),
        high=arr.copy(),
        low=arr.copy(),
        close=arr.copy(),
        volume=arr.copy(),
        time=np.arange(5, dtype=np.int64),
        symbols=("XAUUSD",),
        timeframe="H1",
        market_profile_name=profile_name,
    )


def test_explicit_market_wins() -> None:
    assert resolve_market(_panel("FOREX_XAUUSD"), "CRYPTO_BTC", "XAUUSD") == "CRYPTO_BTC"


def test_panel_profile_used_when_no_explicit() -> None:
    assert resolve_market(_panel("CRYPTO_BTC"), "", "XAUUSD") == "CRYPTO_BTC"


def test_fallback_by_symbol_when_panel_has_none() -> None:
    assert resolve_market(_panel(""), "", "XAUUSD") == "FOREX_XAUUSD"
    assert resolve_market(_panel(""), "", "xauusd") == "FOREX_XAUUSD"
    assert resolve_market(_panel(""), "", "BTCUSDT") == "CRYPTO_BTC"


def test_unknown_symbol_raises_actionable_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        resolve_market(_panel(""), "", "UNKNOWN_PAIR")
    message = str(excinfo.value)
    assert "--market" in message
    assert "FOREX_XAUUSD" in message


def test_gate_reasons_parses_real_verdict_snapshot() -> None:
    """真实快照的键是 hard_failures / soft_failures（GateVerdict.snapshot）。"""
    snapshot = (
        '{"dsr": 0.08, "hard_failures": ["DSR 0.082 < 0.5"],'
        ' "soft_failures": ["WF 胜率 0.40 < 0.6"], "status": "BLOCKED"}'
    )
    assert _gate_reasons(snapshot) == ("DSR 0.082 < 0.5", "WF 胜率 0.40 < 0.6")


def test_gate_reasons_accepts_reasons_alias() -> None:
    assert _gate_reasons('{"reasons": ["a", "b"]}') == ("a", "b")


def test_gate_reasons_tolerates_garbage() -> None:
    assert _gate_reasons("") == ()
    assert _gate_reasons("not json") == ()
    assert _gate_reasons("[1, 2]") == ()
    assert _gate_reasons('{"reasons": null}') == ()


def test_gate_reasons_coerces_to_str() -> None:
    assert _gate_reasons('{"reasons": [1, 2]}') == ("1", "2")
