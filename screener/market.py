"""
Market regime header — same rules as compute_macro_regime() in DB/build_simulation_db.py:

  bull  VIX < 18 and SPY > SMA200 and HYG > SMA20
  bear  at least 2 of: VIX > 25, SPY < SMA200, HYG < SMA20   (bear wins over bull)
  else  sideways
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Regime:
    label: str                      # bull | bear | sideways | unknown
    reasons: list[str] = field(default_factory=list)


def compute_regime(m: pd.DataFrame) -> Regime:
    try:
        vix = m.at["VIX", "close"]
        spy, spy200 = m.at["SPY", "close"], m.at["SPY", "sma200"]
        hyg, hyg20 = m.at["HYG", "close"], m.at["HYG", "sma20"]
    except KeyError:
        return Regime("unknown", ["missing market data"])
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (vix, spy, spy200, hyg, hyg20)):
        return Regime("unknown", ["missing market data"])

    reasons = [
        f"VIX {vix:.1f} ({'<18 calm' if vix < 18 else '>25 fear' if vix > 25 else '18-25 elevated'})",
        f"SPY {'above' if spy > spy200 else 'below'} SMA200",
        f"HYG {'above' if hyg > hyg20 else 'below'} SMA20 (credit {'risk-on' if hyg > hyg20 else 'risk-off'})",
    ]
    bear_count = int(vix > 25) + int(spy < spy200) + int(hyg < hyg20)
    if bear_count >= 2:
        return Regime("bear", reasons)
    if vix < 18 and spy > spy200 and hyg > hyg20:
        return Regime("bull", reasons)
    return Regime("sideways", reasons)
