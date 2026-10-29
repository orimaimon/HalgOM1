"""Portfolio Rebalance View Component.

Provides interactive portfolio rebalancing analysis, drift tracking against
target weights, actionable buy/sell recommendations, and target weight adjustments.
"""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from screener.ui.common import col_label, hebrew_note, L, ss, t

log = logging.getLogger("screener.app")

REBALANCE_COLUMNS = [
    "מניה",
    "משקל נוכחי %",
    "משקל יעד %",
    "סטייה %",
    "פעולה מומלצת",
    "שווי לשינוי $",
]


def calculate_rebalance(
    holdings_df: pd.DataFrame,
    target_weights: dict[str, float] | None = None,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """Calculate rebalance metrics, drift, action recommendations and dollar amounts.

    Returns a DataFrame with columns:
      - מניה
      - משקל נוכחי %
      - משקל יעד %
      - סטייה %
      - פעולה מומלצת
      - שווי לשינוי $
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(columns=REBALANCE_COLUMNS)

    df = holdings_df.copy()

    # Determine ticker column
    ticker_col = None
    for c in ("ticker", "Ticker", "מניה", "symbol", "Symbol"):
        if c in df.columns:
            ticker_col = c
            break

    if ticker_col is not None:
        tickers = df[ticker_col].astype(str).tolist()
    else:
        tickers = [str(idx) for idx in df.index]

    n_stocks = len(tickers)
    if n_stocks == 0:
        return pd.DataFrame(columns=REBALANCE_COLUMNS)

    # Determine position values
    value_col = None
    for c in ("value", "Value", "שווי", "val"):
        if c in df.columns:
            value_col = c
            break

    if value_col is not None:
        values = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).values
    elif "quantity" in df.columns and "price" in df.columns:
        q = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
        p = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        values = (q * p).values
    elif "כמות" in df.columns and "מחיר" in df.columns:
        q = pd.to_numeric(df["כמות"], errors="coerce").fillna(0.0)
        p = pd.to_numeric(df["מחיר"], errors="coerce").fillna(0.0)
        values = (q * p).values
    else:
        values = None

    total_value = float(values.sum()) if values is not None else 0.0

    # Determine current weights
    weight_col = None
    for c in ("weight", "Weight", "משקל", "weight_pct", "משקל %", "משקל נוכחי %"):
        if c in df.columns:
            weight_col = c
            break

    if total_value > 0 and values is not None:
        current_weights = (values / total_value) * 100.0
    elif weight_col is not None:
        current_weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).values
    else:
        current_weights = [100.0 / n_stocks] * n_stocks

    # Determine target weights
    target_col = None
    for c in ("target_weight", "target_pct", "target", "משקל יעד %", "משקל יעד"):
        if c in df.columns:
            target_col = c
            break

    targets = []
    for i, tkr in enumerate(tickers):
        if target_weights is not None and tkr in target_weights:
            targets.append(float(target_weights[tkr]))
        elif target_col is not None and pd.notna(df[target_col].iloc[i]):
            targets.append(float(df[target_col].iloc[i]))
        else:
            targets.append(100.0 / n_stocks)

    out_rows = []
    for i in range(n_stocks):
        tkr = tickers[i]
        curr_w = float(current_weights[i])
        tgt_w = float(targets[i])
        drift = curr_w - tgt_w

        if drift > threshold_pct:
            action = "מכירה"
        elif drift < -threshold_pct:
            action = "קנייה"
        else:
            action = "ללא שינוי"

        if total_value > 0 and action != "ללא שינוי":
            amt_usd = abs(drift) / 100.0 * total_value
        else:
            amt_usd = 0.0

        out_rows.append({
            "מניה": tkr,
            "משקל נוכחי %": round(curr_w, 2),
            "משקל יעד %": round(tgt_w, 2),
            "סטייה %": round(drift, 2),
            "פעולה מומלצת": action,
            "שווי לשינוי $": round(amt_usd, 2),
        })

    return pd.DataFrame(out_rows, columns=REBALANCE_COLUMNS)


def _render_rebalance_body(
    portfolio_name: str,
    holdings_df: pd.DataFrame,
    on_save_targets=None,
) -> None:
    if holdings_df is None or holdings_df.empty:
        hebrew_note("אין פוזיציות פתוחות לאיזון בתיק זה")
        return

    st.subheader(f"איזון מחדש של התיק — {portfolio_name}")

    # Session state for target weights
    ss_targets_key = f"rb_targets_{portfolio_name}"
    if ss_targets_key not in ss:
        ss[ss_targets_key] = {}

    current_targets = ss[ss_targets_key]

    c_top1, c_top2 = st.columns([2, 2])
    with c_top1:
        threshold_val = st.number_input(
            "סף סטייה להתראה (%)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key=f"rb_thresh_{portfolio_name}",
        )

    reb_df = calculate_rebalance(
        holdings_df=holdings_df,
        target_weights=current_targets if current_targets else None,
        threshold_pct=float(threshold_val),
    )

    # b) Visual highlight for drifts exceeding threshold
    drifted_rows = reb_df[reb_df["פעולה מומלצת"] != "ללא שינוי"]
    if not drifted_rows.empty:
        count = len(drifted_rows)
        st.warning(
            f"⚠️ נמצאו {count} מניות בסטייה מהיעד שעלתה על הסף ({threshold_val:.1f}%). "
            f"מומלץ לבצע התאמות לאיזון התיק."
        )
    else:
        st.success(f"✓ כל המניות בתיק נמצאות בטווח היעד (סף סטייה {threshold_val:.1f}%)")

    # a) Comparison table
    st.dataframe(
        reb_df,
        hide_index=True,
        use_container_width=True,
        key=f"rb_table_{portfolio_name}",
    )

    # c) Expander to adjust target weights with validation (sum = 100%)
    with st.expander("התאמת משקלי יעד", expanded=False):
        st.caption("קבע את משקל היעד באחוזים עבור כל מניה בתיק:")

        tickers = reb_df["מניה"].tolist()
        n_tickers = len(tickers)
        default_equal = round(100.0 / n_tickers, 2) if n_tickers > 0 else 0.0

        new_targets = {}
        cols = st.columns(min(3, max(1, n_tickers)))

        for idx, tkr in enumerate(tickers):
            col = cols[idx % len(cols)]
            init_val = float(current_targets.get(tkr, default_equal))
            w_val = col.number_input(
                f"{tkr} (%)",
                min_value=0.0,
                max_value=100.0,
                value=init_val,
                step=1.0,
                key=f"rb_tw_{portfolio_name}_{tkr}",
            )
            new_targets[tkr] = float(w_val)

        total_target_sum = sum(new_targets.values())
        sum_diff = abs(total_target_sum - 100.0)

        if sum_diff > 0.05:
            st.error(
                f"❌ סך משקלי היעד חייב להיות 100% (סך נוכחי: {total_target_sum:.1f}%). "
                f"נא לתקן את המשקלים."
            )
        else:
            st.caption(f"סך משקלי היעד: {total_target_sum:.1f}% ✓")
            c_btn1, c_btn2 = st.columns([2, 2])
            with c_btn1:
                if st.button("שמירת משקלי יעד", key=f"rb_save_btn_{portfolio_name}", type="primary"):
                    ss[ss_targets_key] = new_targets
                    if on_save_targets is not None:
                        on_save_targets(portfolio_name, new_targets)
                    st.success("משקלי היעד נשמרו בהצלחה")
                    st.rerun()
            with c_btn2:
                if st.button("איפוס למשקל שווה", key=f"rb_reset_btn_{portfolio_name}"):
                    eq_w = round(100.0 / n_tickers, 2)
                    reset_targets = {tkr: eq_w for tkr in tickers}
                    ss[ss_targets_key] = reset_targets
                    if on_save_targets is not None:
                        on_save_targets(portfolio_name, reset_targets)
                    st.rerun()


def render_rebalance_view(
    portfolio_name: str,
    holdings_df: pd.DataFrame,
    on_save_targets=None,
) -> None:
    """Render interactive portfolio rebalance UI component."""
    try:
        _render_rebalance_body(portfolio_name, holdings_df, on_save_targets)
    except Exception:
        log.exception("Error rendering rebalance view for %s", portfolio_name)
        hebrew_note("שגיאה בהצגת איזון התיק")
