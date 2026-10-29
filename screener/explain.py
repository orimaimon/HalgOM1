"""
Stock explanation & automatic Hebrew description engine.

Pure functions that explain why a stock is in the screen (or why it fails) given
filter selections, and generate a concise Hebrew narrative description for a stock.
"""
from __future__ import annotations

from typing import Any

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
        sentences.append(f"צמיחה חזקה בהכנסות ({sales_growth:.0f}%) ו{profit_status}.")
    elif has_growth:
        sentences.append(f"צמיחה חזקה בהכנסות ({sales_growth:.0f}%).")
    elif profit_status:
        sentences.append(f"החברה {profit_status}.")

    # 4. Dividend
    div_yield = row.get("div_yield")
    if div_yield is not None and not pd.isna(div_yield) and div_yield >= 3:
        sentences.append(f"מציעה תשואת דיבידנד של {div_yield:.1f}%.")

    # 5. pe_vs_industry
    pe_vs_ind = row.get("pe_vs_industry")
    if pe_vs_ind is not None and not pd.isna(pe_vs_ind) and pe_vs_ind <= -20:
        sentences.append(f"זולה ביחס לתעשייה (P/E נמוך ב-{abs(pe_vs_ind):.0f}% מהחציון).")

    # 6. Score
    score = row.get("score")
    if score is not None and not pd.isna(score) and score >= 80:
        sentences.append(f"ציון HalgOM גבוה ({score:.0f}).")

    return " ".join(sentences)


def explain_stock_match(ticker: str, row: pd.Series, active_filters: dict[str, Any]) -> list[str]:
    """Generate friendly, human-readable Hebrew bullet points explaining why a stock satisfies active filters.

    Analyzes actual stock metrics against satisfied filter predicates and formats margin/context.
    """
    active_sel = filters.active(active_filters or {})
    if not active_sel:
        return []

    one_row_df = row.to_frame().T
    bullets = []

    for f in filters.FILTERS:
        if f.key not in active_sel:
            continue
        opt_label = active_sel[f.key]

        # Check if row passes this filter
        filtered_df = filters.apply_filters(one_row_df, {f.key: opt_label})
        if filtered_df.empty:
            continue

        raw_val = row.get(f.column)
        formatted_val = format_value(f.column, raw_val, lang=i18n.HE)
        olabel = i18n.option(opt_label, lang=i18n.HE)

        val_str = f"{raw_val:g}" if isinstance(raw_val, (int, float)) and not pd.isna(raw_val) else formatted_val

        # 1. Moving average filters (sma20, sma50, sma150, sma200)
        if f.key in ("sma20", "sma50", "sma150", "sma200"):
            n = f.key.replace("sma", "")
            pct_col = f"sma{n}_pct"
            pct_val = row.get(pct_col)
            if pct_val is not None and not pd.isna(pct_val):
                if pct_val >= 0:
                    bullets.append(f"מעל ממוצע {n} ב-{pct_val:.1f}%")
                else:
                    bullets.append(f"מתחת לממוצע {n} ב-{abs(pct_val):.1f}%")
            else:
                bullets.append(f"ממוצע {n}: {formatted_val} ({olabel})")

        # 2. Valuation multiples (pe, fwd_pe, ps, pb, pfcf, peg, ev_ebitda)
        elif f.key in ("pe", "fwd_pe", "ps", "pb", "pfcf", "peg", "ev_ebitda"):
            col_obj = fields.COLUMNS.get(f.column)
            short_name = col_obj.label if col_obj else f.label
            if olabel.startswith("מתחת ל-"):
                thresh = olabel.replace("מתחת ל-", "")
                bullets.append(f"{short_name} {val_str} נמוך מ-{thresh}")
            elif olabel.startswith("מעל "):
                thresh = olabel.replace("מעל ", "")
                bullets.append(f"{short_name} {val_str} מעל {thresh}")
            else:
                bullets.append(f"{short_name} {val_str} ({olabel})")

        # 3. Growth filters
        elif f.key == "sales_growth":
            if raw_val is not None and not pd.isna(raw_val) and raw_val > 0:
                bullets.append(f"צמיחת הכנסות חיובית {raw_val:g}%")
            else:
                bullets.append(f"צמיחת הכנסות {formatted_val} ({olabel})")

        elif f.key == "eps_growth":
            if raw_val is not None and not pd.isna(raw_val) and raw_val > 0:
                bullets.append(f"צמיחת EPS חיובית {raw_val:g}%")
            else:
                bullets.append(f"צמיחת EPS {formatted_val} ({olabel})")

        # 4. Relative metrics (pe_vs_industry, ps_vs_industry, sales_growth_vs_sector)
        elif f.key == "pe_vs_industry":
            if raw_val is not None and not pd.isna(raw_val):
                if raw_val < 0:
                    bullets.append(f"P/E נמוך ב-{abs(raw_val):.0f}% מהחציון בתעשייה")
                else:
                    bullets.append(f"P/E גבוה ב-{raw_val:.0f}% מהחציון בתעשייה")
            else:
                bullets.append(f"P/E מול תעשייה {formatted_val} ({olabel})")

        elif f.key == "ps_vs_industry":
            if raw_val is not None and not pd.isna(raw_val):
                if raw_val < 0:
                    bullets.append(f"P/S נמוך ב-{abs(raw_val):.0f}% מהחציון בתעשייה")
                else:
                    bullets.append(f"P/S גבוה ב-{raw_val:.0f}% מהחציון בתעשייה")
            else:
                bullets.append(f"P/S מול תעשייה {formatted_val} ({olabel})")

        elif f.key == "sales_growth_vs_sector":
            if raw_val is not None and not pd.isna(raw_val) and raw_val > 0:
                bullets.append(f"צמיחת הכנסות גבוהה ב-{raw_val:.0f}% מחציון הסקטור")
            else:
                bullets.append(f"צמיחת הכנסות מול סקטור {formatted_val} ({olabel})")

        # 5. Volume spike
        elif f.key == "vol_spike":
            rel_vol = row.get("rel_volume")
            if rel_vol is not None and not pd.isna(rel_vol) and rel_vol > 0:
                bullets.append(f"מחזור חריג פי {rel_vol:.1f} מהממוצע")
            else:
                bullets.append(f"קפיצת מחזור ({olabel})")

        # 6. Dynamic filters (sector, country, industry, exchange)
        elif f.key == "sector":
            sec_he = i18n.value("sector", raw_val, i18n.HE) if raw_val else formatted_val
            bullets.append(f"סקטור {sec_he}")

        elif f.key == "country":
            ctry_he = i18n.value("country", raw_val, i18n.HE) if raw_val else formatted_val
            bullets.append(f"מדינה {ctry_he}")

        # 7. Fallback for other filters
        else:
            flabel = i18n.filter_label(f.key, f.label, i18n.HE)
            bullets.append(f"{flabel} {formatted_val} ({olabel})")

    return bullets


def stock_summary_card_text(row: pd.Series) -> str:
    """Generate a 1-2 sentence executive summary in Hebrew highlighting the stock's strongest factors (Value, Momentum, Growth)."""
    if row is None or row.empty:
        return ""

    score_val = row.get("score_value")
    score_mom = row.get("score_momentum")
    pe_vs_ind = row.get("pe_vs_industry")
    pe = row.get("pe")
    sma200_pct = row.get("sma200_pct")
    high52_pct = row.get("high52_pct")
    perf_y = row.get("perf_y")
    sales_growth = row.get("sales_growth")
    eps_growth = row.get("eps_growth")

    # Assess factor strength
    has_val = (
        (score_val is not None and not pd.isna(score_val) and score_val >= 70)
        or (pe_vs_ind is not None and not pd.isna(pe_vs_ind) and pe_vs_ind <= -20)
        or (pe is not None and not pd.isna(pe) and 0 < pe <= 15)
    )

    has_mom = (
        (score_mom is not None and not pd.isna(score_mom) and score_mom >= 70)
        or (
            sma200_pct is not None and not pd.isna(sma200_pct) and sma200_pct > 0
            and high52_pct is not None and not pd.isna(high52_pct) and high52_pct >= -10
        )
        or (perf_y is not None and not pd.isna(perf_y) and perf_y >= 25)
    )

    has_growth = (
        (sales_growth is not None and not pd.isna(sales_growth) and sales_growth >= 15)
        or (eps_growth is not None and not pd.isna(eps_growth) and eps_growth >= 15)
    )

    sentences = []

    # Combination of strong factors
    if has_val and has_mom and has_growth:
        sentences.append("המניה מציגה שילוב חזק של ערך (תמחור אטרקטיבי), צמיחה מהירה ומומנטום חיובי במגמה עולה.")
    elif has_val and has_growth:
        sentences.append("המניה בולטת בגורמי ערך וצמיחה: תמחור נוח לצד צמיחה חזקה בהכנסות.")
    elif has_mom and has_growth:
        sentences.append("המניה מציגה מומנטום חזק במגמת עלייה לצד צמיחה מהירה במכירות.")
    elif has_val and has_mom:
        sentences.append("המניה משלבת תמחור נוח (ערך) עם מומנטום חיובי במגמת עלייה.")
    elif has_val:
        sentences.append("המניה בולטת בעיקר בתמחור נוח וערך אטרקטיבי ביחס לשוק ולתעשייה.")
    elif has_mom:
        sentences.append("המניה מציגה מומנטום חיובי ומגמת עלייה חזקה.")
    elif has_growth:
        sentences.append("המניה מתאפיינת בצמיחה חזקה בהכנסות וברווחיות.")
    else:
        tot_score = row.get("score")
        if tot_score is not None and not pd.isna(tot_score) and tot_score >= 75:
            sentences.append(f"המניה מציגה פרופיל מאוזן עם ציון HalgOM גבוה ({tot_score:.0f}).")
        else:
            desc = describe(row)
            if desc:
                parts = [s.strip() for s in desc.split(".") if s.strip()]
                return ". ".join(parts[:2]) + "." if parts else ""
            return ""

    return " ".join(sentences)
