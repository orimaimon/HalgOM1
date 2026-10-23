"""
Stock explanation & automatic Hebrew description engine.

Pure functions that explain why a stock is in the screen (or why it fails) given
filter selections, and generate a concise Hebrew narrative description for a stock.
"""
from __future__ import annotations

import pandas as pd

from screener import fields, filters, i18n


def format_value(key: str, v, lang: str = i18n.HE) -> str:
    """Format a value according to its column kind in fields.COLUMNS or dynamic rules.

    NaN or missing -> '—'.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NaT or v == "":
        return "—"

    if key in ("sector", "country"):
        return str(i18n.value(key, v, lang))
    if key == "industry":
        return str(v)

    col = fields.COLUMNS.get(key)
    kind = col.kind if col else None

    if kind == "pct":
        return f"{v:+.1f}%" if key in ("change", "sma200_pct", "high52_pct", "low52_pct", "target_upside") else f"{v:.1f}%"
    if kind == "money":
        prefix = "$"
        abs_v = abs(v)
        sign = "-" if v < 0 else ""
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs_v >= div:
                return f"{sign}{prefix}{abs_v / div:.1f}{suf}"
        return f"{sign}{prefix}{abs_v:,.0f}"
    if kind == "price":
        return f"${v:,.2f}"
    if kind == "ratio":
        return f"{v:.2f}"
    if kind == "count":
        abs_v = abs(v)
        sign = "-" if v < 0 else ""
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs_v >= div:
                return f"{sign}{abs_v / div:.1f}{suf}"
        return f"{sign}{abs_v:,.0f}"
    if kind == "date":
        return pd.Timestamp(v).strftime("%d/%m/%Y")
    if kind in ("days", "score", "int"):
        return f"{v:,.0f}"

    return str(i18n.value(key, v, lang))


def explain(row: pd.Series, selections: dict[str, str], lang: str = i18n.HE) -> list[dict]:
    """Return explanation dicts for each active filter on the stock row.

    Order matches filters.FILTERS.
    Each dict contains:
      - key: filter key
      - filter: translated filter label
      - option: translated option label
      - column: column key from filter
      - value: formatted stock value (or "—")
      - passes: bool indicating if row satisfies the active selection
    """
    active_sel = filters.active(selections)
    if not active_sel:
        return []

    one_row_df = row.to_frame().T
    res = []

    for f in filters.FILTERS:
        if f.key not in active_sel:
            continue
        opt_label = active_sel[f.key]

        # Evaluate filter pass/fail using existing apply_filters pipeline
        filtered_df = filters.apply_filters(one_row_df, {f.key: opt_label})
        passes = not filtered_df.empty

        # Dynamic filter value handling (sector, industry, country) vs row column value
        raw_val = row.get(f.column)
        formatted_val = format_value(f.column, raw_val, lang=lang)

        # Labels translated
        flabel = i18n.filter_label(f.key, f.label, lang)
        olabel = i18n.option(opt_label, lang)

        res.append({
            "key": f.key,
            "filter": flabel,
            "option": olabel,
            "column": f.column,
            "value": formatted_val,
            "passes": bool(passes),
        })

    return res


def describe(row: pd.Series) -> str:
    """Generate 1-4 short Hebrew sentences summarizing the stock's profile based on rules.

    Skips sentences where required values are NaN or missing.
    """
    sentences = []

    # 1. Size by market cap + sector
    # ענק ≥ $200B, גדולה $10B–$200B, בינונית $2B–$10B, קטנה $300M–$2B, מיקרו below
    cap = row.get("market_cap")
    if cap is not None and not pd.isna(cap) and cap > 0:
        if cap >= 200e9:
            size_label = "חברה ענקית"
        elif cap >= 10e9:
            size_label = "חברה גדולה"
        elif cap >= 2e9:
            size_label = "חברה בינונית"
        elif cap >= 300e6:
            size_label = "חברה קטנה"
        else:
            size_label = "חברת מיקרו"

        cap_fmt = format_value("market_cap", cap, i18n.HE)
        sector_val = row.get("sector")
        if sector_val is not None and not pd.isna(sector_val) and str(sector_val).strip() != "":
            sec_he = i18n.value("sector", sector_val, i18n.HE)
            sentences.append(f"{size_label} (שווי {cap_fmt}) בסקטור {sec_he}.")
        else:
            sentences.append(f"{size_label} (שווי {cap_fmt}).")

    # 2. Trend: SMA200 + near 52W high
    # Check above/below SMA200 using sma200_pct or price vs sma200
    sma200_pct = row.get("sma200_pct")
    price = row.get("price")
    sma200 = row.get("sma200")

    above_sma200 = None
    if sma200_pct is not None and not pd.isna(sma200_pct):
        above_sma200 = sma200_pct > 0
    elif price is not None and not pd.isna(price) and sma200 is not None and not pd.isna(sma200):
        above_sma200 = price > sma200

    high52_pct = row.get("high52_pct")
    near_high = high52_pct is not None and not pd.isna(high52_pct) and high52_pct >= -5

    if above_sma200 is True:
        if near_high:
            sentences.append("נסחרת במגמת עלייה מעל ממוצע 200 וקרובה לשיא השנתי.")
        else:
            sentences.append("נסחרת במגמת עלייה מעל ממוצע 200.")
    elif above_sma200 is False:
        if near_high:
            sentences.append("נסחרת מתחת לממוצע 200 אך קרובה לשיא השנתי.")
        else:
            sentences.append("נסחרת במגמת ירידה מתחת לממוצע 200.")
    elif near_high:
        sentences.append("קרובה לשיא השנתי.")

    # 3. Growth & Profit
    sales_growth = row.get("sales_growth")
    has_growth = sales_growth is not None and not pd.isna(sales_growth) and sales_growth >= 20

    net_margin = row.get("net_margin")
    profit_status = None
    if net_margin is not None and not pd.isna(net_margin):
        if net_margin > 0:
            profit_status = "רווחית"
        elif net_margin < 0:
            profit_status = "מפסידה"

    if has_growth and profit_status:
        sentences.append(f"צמיחה חזקה בהכנסות ({sales_growth:g}%) ו{profit_status}.")
    elif has_growth:
        sentences.append(f"צמיחה חזקה בהכנסות ({sales_growth:g}%).")
    elif profit_status:
        sentences.append(f"החברה {profit_status}.")

    # 4. Dividend
    div_yield = row.get("div_yield")
    if div_yield is not None and not pd.isna(div_yield) and div_yield >= 3:
        sentences.append(f"מציעה תשואת דיבידנד של {div_yield:g}%.")

    # 5. pe_vs_industry
    pe_vs_ind = row.get("pe_vs_industry")
    if pe_vs_ind is not None and not pd.isna(pe_vs_ind) and pe_vs_ind <= -20:
        sentences.append(f"זולה ביחס לתעשייה (P/E נמוך ב-{abs(pe_vs_ind):g}% מהחציון).")

    # 6. Score
    score = row.get("score")
    if score is not None and not pd.isna(score) and score >= 80:
        sentences.append(f"ציון HalgOM גבוה ({score:g}).")

    return " ".join(sentences)
