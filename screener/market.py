"""
Market regime header — same rules as compute_macro_regime() in the HalgOM backtester:

  bull  VIX < 18 and SPY > SMA200 and HYG > SMA20
  bear  at least 2 of: VIX > 25, SPY < SMA200, HYG < SMA20   (bear wins over bull)
  else  sideways

trend_strength() adds SPY's ADX(14) as a separate indication of how strong the trend is
(ADX >= 25 strong, < 20 weak); it does not change the regime rules above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Regime:
    label: str                      # bull | bear | sideways | unknown
    reasons: list[str] = field(default_factory=list)


VIX_CALM = "<18 calm"
VIX_FEAR = ">25 fear"
VIX_ELEVATED = "18-25 elevated"
SPY_ABOVE = "above"
SPY_BELOW = "below"
HYG_RISK_ON = "credit risk-on"
HYG_RISK_OFF = "credit risk-off"
MISSING_DATA = "missing market data"


def compute_regime(m: pd.DataFrame) -> Regime:
    try:
        vix = m.at["VIX", "close"]
        spy, spy200 = m.at["SPY", "close"], m.at["SPY", "sma200"]
        hyg, hyg20 = m.at["HYG", "close"], m.at["HYG", "sma20"]
    except KeyError:
        return Regime("unknown", [MISSING_DATA])
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (vix, spy, spy200, hyg, hyg20)):
        return Regime("unknown", [MISSING_DATA])

    vix_part = VIX_CALM if vix < 18 else VIX_FEAR if vix > 25 else VIX_ELEVATED
    spy_part = SPY_ABOVE if spy > spy200 else SPY_BELOW
    hyg_part = HYG_RISK_ON if hyg > hyg20 else HYG_RISK_OFF

    reasons = [
        f"VIX {vix:.1f} ({vix_part})",
        f"SPY {spy_part} SMA200",
        f"HYG {SPY_ABOVE if hyg > hyg20 else SPY_BELOW} SMA20 ({hyg_part})",
    ]
    bear_count = int(vix > 25) + int(spy < spy200) + int(hyg < hyg20)
    if bear_count >= 2:
        return Regime("bear", reasons)
    if vix < 18 and spy > spy200 and hyg > hyg20:
        return Regime("bull", reasons)
    return Regime("sideways", reasons)


@dataclass
class Trend:
    label: str                      # "Trend strong ▲" | "Trend building ▼" | "Trend weak" | "Trend n/a"
    color: str
    help: str


def trend_strength(adx: float, di_plus: float = float("nan"), di_minus: float = float("nan")) -> Trend:
    if adx is None or (isinstance(adx, float) and math.isnan(adx)):
        return Trend("Trend n/a", "gray", "אין נתון ADX כרגע")
    direction = ""
    if not (math.isnan(di_plus) or math.isnan(di_minus)):
        direction = " ▲" if di_plus > di_minus else " ▼"
    side = {" ▲": "כלפי מעלה (DI+ מעל DI-)", " ▼": "כלפי מטה (DI- מעל DI+)", "": ""}[direction]
    base = (f"ADX(14) של SPY = {adx:.1f}. מודד את עוצמת המגמה, לא את הכיוון: "
            "מעל 25 = מגמה חזקה, מתחת ל-20 = שוק בלי מגמה ברורה (דשדוש). ")
    if adx >= 25:
        return Trend("Trend strong" + direction, "blue", base + f"כיוון המגמה: {side}")
    if adx >= 20:
        return Trend("Trend building" + direction, "violet", base + f"מגמה מתפתחת, כיוון: {side}")
    return Trend("Trend weak", "gray", base + "כרגע אין מגמה ברורה — פריצות נכשלות יותר")
