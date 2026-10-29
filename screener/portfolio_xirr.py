"""
Pure XIRR (money-weighted annualised return) calculator for cash flows.
"""

import datetime
import math
import pandas as pd


def _npv_and_deriv(r: float, t_cf_pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Calculate Net Present Value and its derivative with respect to r."""
    base = 1.0 + r
    if base <= 0:
        return float("nan"), float("nan")
    npv = 0.0
    deriv = 0.0
    for t_i, cf_i in t_cf_pairs:
        try:
            factor = base**t_i
            npv += cf_i / factor
            if t_i != 0.0:
                deriv += (-t_i * cf_i) / (factor * base)
        except (OverflowError, ZeroDivisionError):
            return float("nan"), float("nan")
    return npv, deriv


def xirr(flows: list[tuple[datetime.date, float]]) -> float | None:
    """
    Compute internal rate of return (XIRR) for a series of dated cash flows.

    Cash flow convention:
    - Negative value = money invested
    - Positive value = money received (final entry is current portfolio value)

    Solves sum(cf / (1+r)^((d-d0)/365)) = 0 for r in (-0.99, 10.0) using a
    bounded bisection/Newton hybrid. Max 200 iterations.

    Returns None when:
    - fewer than 2 flows
    - no sign change (all positive or all negative)
    - all flows on one date
    - no root in range (-0.99, 10.0)
    Never raises on bad input.
    """
    try:
        if not flows or not isinstance(flows, (list, tuple)):
            return None

        clean_dict: dict[datetime.date, float] = {}
        for item in flows:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            d, cf = item[0], item[1]
            if d is None or cf is None:
                continue
            if hasattr(d, "date") and callable(d.date):
                d = d.date()
            if not isinstance(d, datetime.date):
                try:
                    parsed = pd.to_datetime(d, errors="coerce")
                    if pd.isna(parsed):
                        continue
                    d = parsed.date()
                except Exception:
                    continue
            try:
                cf_val = float(cf)
                if math.isnan(cf_val) or math.isinf(cf_val):
                    continue
            except (ValueError, TypeError):
                continue

            clean_dict[d] = clean_dict.get(d, 0.0) + cf_val

        # Require at least 2 distinct dates
        if len(clean_dict) < 2:
            return None

        # Check for sign change
        has_pos = any(cf > 0 for cf in clean_dict.values())
        has_neg = any(cf < 0 for cf in clean_dict.values())
        if not (has_pos and has_neg):
            return None

        sorted_dates = sorted(clean_dict.keys())
        d0 = sorted_dates[0]
        t_cf_pairs = [((d - d0).days / 365.0, clean_dict[d]) for d in sorted_dates]

        # Bracketing search across r in (-0.99, 10.0)
        grid_steps = 100
        r_min, r_max = -0.99, 10.0
        step_size = (r_max - r_min) / grid_steps

        grid_r = [r_min + i * step_size for i in range(grid_steps + 1)]
        grid_f = []
        for r in grid_r:
            f_val, _ = _npv_and_deriv(r, t_cf_pairs)
            grid_f.append(f_val)

        bracket = None
        for i in range(grid_steps):
            f1, f2 = grid_f[i], grid_f[i + 1]
            if math.isnan(f1) or math.isnan(f2):
                continue
            if abs(f1) < 1e-8:
                return grid_r[i]
            if abs(f2) < 1e-8:
                return grid_r[i + 1]
            if f1 * f2 < 0:
                bracket = (grid_r[i], grid_r[i + 1])
                break

        if bracket is None:
            # Fallback check ends
            f_min, _ = _npv_and_deriv(r_min, t_cf_pairs)
            f_max, _ = _npv_and_deriv(r_max, t_cf_pairs)
            if not (math.isnan(f_min) or math.isnan(f_max)) and f_min * f_max <= 0:
                bracket = (r_min, r_max)

        if bracket is None:
            return None

        a, b = bracket
        fa, _ = _npv_and_deriv(a, t_cf_pairs)
        fb, _ = _npv_and_deriv(b, t_cf_pairs)

        r_curr = (a + b) / 2.0
        for _ in range(200):
            if abs(b - a) < 1e-8:
                return r_curr

            f_curr, df_curr = _npv_and_deriv(r_curr, t_cf_pairs)
            if math.isnan(f_curr):
                r_curr = (a + b) / 2.0
                f_curr, df_curr = _npv_and_deriv(r_curr, t_cf_pairs)

            if not math.isnan(f_curr) and abs(f_curr) < 1e-8:
                return r_curr

            # Try Newton step
            r_new = None
            if not math.isnan(df_curr) and abs(df_curr) > 1e-12:
                cand = r_curr - f_curr / df_curr
                if a < cand < b:
                    r_new = cand

            if r_new is None:
                r_new = (a + b) / 2.0

            f_new, _ = _npv_and_deriv(r_new, t_cf_pairs)
            if not math.isnan(f_new) and abs(f_new) < 1e-8:
                return r_new

            # Update bracket
            if not math.isnan(f_new) and not math.isnan(fa) and fa * f_new <= 0:
                b = r_new
                fb = f_new
            else:
                a = r_new
                fa = f_new

            r_curr = r_new if r_new is not None else (a + b) / 2.0

        return r_curr
    except Exception:
        return None


def flows_from_trades(
    trades: pd.DataFrame, end_value: float, end_date: datetime.date
) -> list[tuple[datetime.date, float]]:
    """
    Construct dated cash flow tuples from a trades DataFrame and final portfolio valuation.

    `trades` has columns: date, side ('buy'/'sell'), shares, price.
    - buys are negative flows (-1 * shares * price)
    - sells are positive flows (+1 * shares * price)
    - final entry is `end_value` on `end_date`

    Ignores rows with missing/non-positive shares or price or invalid date/side.
    """
    flows: list[tuple[datetime.date, float]] = []

    clean_end_date = None
    if end_date is not None:
        try:
            if hasattr(end_date, "date") and callable(end_date.date):
                clean_end_date = end_date.date()
            elif isinstance(end_date, datetime.date):
                clean_end_date = end_date
            else:
                parsed = pd.to_datetime(end_date, errors="coerce")
                if not pd.isna(parsed):
                    clean_end_date = parsed.date()
        except Exception:
            clean_end_date = None

    if isinstance(trades, pd.DataFrame) and not trades.empty:
        required_cols = {"date", "side", "shares", "price"}
        if required_cols.issubset(set(trades.columns)):
            for _, row in trades.iterrows():
                try:
                    shares = float(row["shares"])
                    price = float(row["price"])
                    if math.isnan(shares) or math.isnan(price) or shares <= 0 or price <= 0:
                        continue

                    side = str(row["side"]).strip().lower() if pd.notna(row["side"]) else ""
                    if side == "buy":
                        mult = -1.0
                    elif side == "sell":
                        mult = 1.0
                    else:
                        continue

                    raw_date = row["date"]
                    if pd.isna(raw_date):
                        continue

                    if hasattr(raw_date, "date") and callable(raw_date.date):
                        t_date = raw_date.date()
                    elif isinstance(raw_date, datetime.date):
                        t_date = raw_date
                    else:
                        parsed = pd.to_datetime(raw_date, errors="coerce")
                        if pd.isna(parsed):
                            continue
                        t_date = parsed.date()

                    flows.append((t_date, mult * shares * price))
                except Exception:
                    continue

    if clean_end_date is not None:
        try:
            end_val_flt = float(end_value)
            if not math.isnan(end_val_flt) and not math.isinf(end_val_flt):
                flows.append((clean_end_date, end_val_flt))
        except Exception:
            pass

    return flows


def annualised_to_period(rate: float, days: int) -> float:
    """
    Convert an annualised return rate to the rate over the given number of days.

    Formula: (1 + rate)^(days / 365) - 1
    Returns float('nan') on invalid input.
    """
    try:
        if rate is None or math.isnan(rate) or (1.0 + rate) < 0:
            return float("nan")
        days_flt = float(days)
        return (1.0 + rate) ** (days_flt / 365.0) - 1.0
    except Exception:
        return float("nan")
