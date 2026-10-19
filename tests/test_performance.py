import datetime as dt
import numpy as np
import pandas as pd
import pytest

from screener.fields import provider_columns
from screener.derive import add_derived, _pct_from
from screener.fields import COLUMNS
from screener.scores import add_scores
from screener.signals import summarize

def generate_synthetic_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = provider_columns()
    data = {}
    for c in cols:
        if c in ("ticker", "company", "sector", "industry", "exchange", "country", "sec_type", "indexes"):
            data[c] = [f"STR_{i}" for i in range(n_rows)]
        elif c == "earnings_date":
            base = pd.Timestamp('2024-01-01')
            s = pd.Series(base + pd.to_timedelta(rng.integers(-30, 30, size=n_rows), unit='D'))
            s.loc[rng.random(n_rows) < 0.1] = pd.NaT
            data[c] = s
        else:
            s = pd.Series(rng.uniform(0.1, 100.0, size=n_rows))
            s.loc[rng.random(n_rows) < 0.1] = np.nan
            data[c] = s
    return pd.DataFrame(data)

def old_add_derived(raw: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    df = raw.copy()
    from screener.fields import provider_columns
    missing = [k for k in provider_columns() if k not in df.columns]
    for k in missing:
        df[k] = None if COLUMNS[k].kind == "text" else np.nan
    if "earnings_date" in missing:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    today = today or dt.date.today()
    p = df["price"]

    df["adj_price"] = p
    df["dollar_volume"] = df["avg_volume"] * p
    df["fwd_pe"] = p / df["eps_next_fy"].where(df["eps_next_fy"] > 0)
    for n in (20, 50, 150, 200):
        df[f"sma{n}_pct"] = _pct_from(p, df[f"sma{n}"])
    df["sma50w_pct"] = _pct_from(df["close_w"], df["sma50_w"])
    df["high52_pct"] = _pct_from(p, df["high52"])
    df["low52_pct"] = _pct_from(p, df["low52"])
    span = (df["high52"] - df["low52"]).where(lambda s: s > 0)
    df["range52"] = ((p - df["low52"]) / span).clip(0, 1)
    df["range_1m_pct"] = _pct_from(df["high_1m"], df["low_1m"])
    df["range_5d_pct"] = _pct_from(df["high_5d"], df["low_5d"])
    df["target_upside"] = _pct_from(df["target"], p)

    days = (df["earnings_date"] - pd.Timestamp(today)).dt.days
    df["days_to_earnings"] = days.where(days >= 0).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    df["signals"], df["n_signals"] = summarize(df)
    df["new"] = ""
    return add_scores(df)

def test_performance_equality():
    df = generate_synthetic_data()
    today = dt.date(2024, 1, 1)

    import warnings
    warnings.simplefilter("ignore")

    baseline = old_add_derived(df, today)
    derived = add_derived(df, today)

    # ensure that it's equal
    pd.testing.assert_frame_equal(baseline, derived)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
