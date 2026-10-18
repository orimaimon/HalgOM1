"""
Column registry — the single source of truth for every column the screener knows.

Each column defines:
  label  English display label (Finviz-style terminology)
  kind   format kind; drives the on-screen table, the Excel number format and filter labels
  tv     TradingView scanner field, or None for columns computed in derive.py / scores.py
  help   Hebrew tooltip shown to the end user

Percent columns (kind="pct") hold percent numbers (5.0 == 5%), as TradingView returns them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Col:
    label: str
    kind: str  # text | price | pct | ratio | int | money | count | date | days | score | range
    tv: str | None = None
    help: str = ""


COLUMNS: dict[str, Col] = {
    # ── Descriptive ─────────────────────────────────────────────────────────
    "ticker":       Col("Ticker", "text", "name", "סימול המניה בבורסה"),
    "company":      Col("Company", "text", "description", "שם החברה"),
    "exchange":     Col("Exchange", "text", "exchange", "הבורסה שבה המניה נסחרת"),
    "sector":       Col("Sector", "text", "sector", "הסקטור של החברה (סיווג TradingView)"),
    "industry":     Col("Industry", "text", "industry", "תעשייה — סיווג מפורט יותר מהסקטור"),
    "country":      Col("Country", "text", "country", "מדינת המטה של החברה"),
    "sec_type":     Col("Type", "text", "type", "stock = מניה רגילה, dr = ADR (תעודת מניה זרה)"),
    "employees":    Col("Employees", "int", "number_of_employees", "מספר עובדים"),

    # ── Price & volume ──────────────────────────────────────────────────────
    "price":        Col("Price", "price", "close", "מחיר אחרון (בעיכוב של כ-15 דקות)"),
    "change":       Col("Change", "pct", "change", "שינוי יומי באחוזים"),
    "change_abs":   Col("Change $", "price", "change_abs", "שינוי יומי בדולרים"),
    "gap":          Col("Gap", "pct", "gap", "פער פתיחה: הפתיחה היום מול הסגירה אתמול"),
    "premarket":    Col("Pre-market", "pct", "premarket_change", "שינוי במסחר שלפני הפתיחה"),
    "postmarket":   Col("After-hours", "pct", "postmarket_change", "שינוי במסחר שאחרי הסגירה"),
    "volume":       Col("Volume", "count", "volume", "מחזור מניות היום"),
    "avg_volume":   Col("Avg Volume", "count", "average_volume_30d_calc", "מחזור ממוצע ב-30 הימים האחרונים"),
    "rel_volume":   Col("Rel Volume", "ratio", "relative_volume_10d_calc",
                        "מחזור יחסי: מחזור היום חלקי הממוצע. 2 = פי שניים מהרגיל"),
    "dollar_volume": Col("Dollar Vol", "money", None,
                         "נזילות: מחזור ממוצע כפול מחיר, כלומר כמה דולרים נסחרים ביום"),
    "market_cap":   Col("Market Cap", "money", "market_cap_basic", "שווי שוק: מחיר כפול מספר המניות"),
    "float_shares": Col("Float", "count", "float_shares_outstanding", "מניות שנסחרות בפועל בציבור"),
    "shares_out":   Col("Shares Out", "count", "total_shares_outstanding_current", "סך המניות המונפקות"),

    # ── Valuation ───────────────────────────────────────────────────────────
    "pe":           Col("P/E", "ratio", "price_earnings_ttm",
                        "מכפיל רווח: כמה שנות רווח משלמים על המניה. נמוך = זול יותר"),
    "eps_next_fy":  Col("EPS next Y", "price", "earnings_per_share_forecast_next_fy",
                        "תחזית אנליסטים לרווח למניה בשנה הבאה"),
    "fwd_pe":       Col("Fwd P/E", "ratio", None, "מכפיל רווח עתידי: מחיר חלקי תחזית הרווח לשנה הבאה"),
    "peg":          Col("PEG", "ratio", "price_earnings_growth_ttm",
                        "מכפיל רווח חלקי קצב הצמיחה. מתחת ל-1 נחשב זול ביחס לצמיחה"),
    "ps":           Col("P/S", "ratio", "price_sales_current", "מכפיל מכירות: שווי שוק חלקי ההכנסות"),
    "pb":           Col("P/B", "ratio", "price_book_fq", "מכפיל הון: שווי שוק חלקי ההון העצמי"),
    "pocf":         Col("P/OCF", "ratio", "price_to_cash_f_operating_activities_ttm",
                        "מחיר חלקי תזרים המזומנים מפעילות שוטפת"),
    "pfcf":         Col("P/FCF", "ratio", "price_free_cash_flow_ttm", "מחיר חלקי תזרים המזומנים החופשי"),
    "ev_ebitda":    Col("EV/EBITDA", "ratio", "enterprise_value_ebitda_ttm",
                        "שווי פעילות חלקי EBITDA. משווה חברות בלי השפעת החוב"),
    "eps":          Col("EPS", "price", "earnings_per_share_diluted_ttm", "רווח למניה ב-12 החודשים האחרונים"),
    "eps_growth":   Col("EPS growth", "pct", "earnings_per_share_diluted_yoy_growth_ttm",
                        "צמיחת הרווח למניה ב-12 החודשים האחרונים מול השנה הקודמת"),
    "eps_growth_q": Col("EPS Q/Q", "pct", "earnings_per_share_diluted_yoy_growth_fq",
                        "צמיחת הרווח למניה ברבעון האחרון מול אותו רבעון אשתקד"),
    "revenue":      Col("Sales", "money", "total_revenue_ttm", "הכנסות ב-12 החודשים האחרונים"),
    "sales_growth": Col("Sales growth", "pct", "total_revenue_yoy_growth_ttm",
                        "צמיחת ההכנסות ב-12 החודשים האחרונים מול השנה הקודמת"),
    "sales_growth_q": Col("Sales Q/Q", "pct", "total_revenue_yoy_growth_fq",
                          "צמיחת ההכנסות ברבעון האחרון מול אותו רבעון אשתקד"),

    # ── Financial ───────────────────────────────────────────────────────────
    "roa":          Col("ROA", "pct", "return_on_assets", "תשואה על הנכסים: רווח חלקי סך הנכסים"),
    "roe":          Col("ROE", "pct", "return_on_equity", "תשואה על ההון: רווח חלקי ההון העצמי. מעל 15% נחשב טוב"),
    "roic":         Col("ROIC", "pct", "return_on_invested_capital", "תשואה על ההון המושקע (הון + חוב)"),
    "curr_ratio":   Col("Curr R", "ratio", "current_ratio_fq",
                        "יחס שוטף: נכסים שוטפים חלקי התחייבויות שוטפות. מעל 1 = יכולת לשלם התחייבויות קצרות"),
    "quick_ratio":  Col("Quick R", "ratio", "quick_ratio_fq", "יחס מהיר: כמו יחס שוטף, בלי מלאי"),
    "lt_debt_eq":   Col("LTDebt/Eq", "ratio", "long_term_debt_to_equity_fq", "חוב לטווח ארוך חלקי ההון העצמי"),
    "debt_eq":      Col("Debt/Eq", "ratio", "debt_to_equity_fq", "סך החוב חלקי ההון העצמי. נמוך = פחות ממונף"),
    "gross_margin": Col("Gross M", "pct", "gross_margin_ttm", "שיעור רווח גולמי"),
    "oper_margin":  Col("Oper M", "pct", "operating_margin_ttm", "שיעור רווח תפעולי"),
    "net_margin":   Col("Profit M", "pct", "net_margin_ttm", "שיעור רווח נקי"),
    "fcf_margin":   Col("FCF M", "pct", "free_cash_flow_margin_ttm", "תזרים מזומנים חופשי כאחוז מההכנסות"),
    "div_yield":    Col("Dividend", "pct", "dividends_yield_current", "תשואת דיבידנד שנתית"),
    "payout":       Col("Payout", "pct", "dividend_payout_ratio_ttm",
                        "איזה חלק מהרווח מחולק כדיבידנד. מעל 80% עלול להיות לא בר-קיימא"),

    # ── Performance ─────────────────────────────────────────────────────────
    "perf_w":       Col("Perf Week", "pct", "Perf.W", "תשואה בשבוע האחרון"),
    "perf_m":       Col("Perf Month", "pct", "Perf.1M", "תשואה בחודש האחרון"),
    "perf_q":       Col("Perf Quart", "pct", "Perf.3M", "תשואה ב-3 החודשים האחרונים"),
    "perf_h":       Col("Perf Half", "pct", "Perf.6M", "תשואה בחצי השנה האחרונה"),
    "perf_ytd":     Col("Perf YTD", "pct", "Perf.YTD", "תשואה מתחילת השנה"),
    "perf_y":       Col("Perf Year", "pct", "Perf.Y", "תשואה בשנה האחרונה"),
    "perf_5y":      Col("Perf 5Y", "pct", "Perf.5Y", "תשואה ב-5 השנים האחרונות"),
    "vol_w":        Col("Volatility W", "pct", "Volatility.W", "תנודתיות יומית ממוצעת בשבוע האחרון"),
    "vol_m":        Col("Volatility M", "pct", "Volatility.M", "תנודתיות יומית ממוצעת בחודש האחרון"),

    # ── Technical ───────────────────────────────────────────────────────────
    "rsi":          Col("RSI", "ratio", "RSI",
                        "RSI(14): מעל 70 = קנוי יתר (עלה מהר מדי), מתחת ל-30 = מכור יתר (ירד מהר מדי)"),
    "sma20":        Col("SMA20", "price", "SMA20", "ממוצע נע פשוט של 20 יום"),
    "sma50":        Col("SMA50", "price", "SMA50", "ממוצע נע פשוט של 50 יום"),
    "sma200":       Col("SMA200", "price", "SMA200", "ממוצע נע פשוט של 200 יום"),
    "sma20_pct":    Col("SMA20 %", "pct", None, "מרחק המחיר מהממוצע הנע של 20 יום, באחוזים"),
    "sma50_pct":    Col("SMA50 %", "pct", None, "מרחק המחיר מהממוצע הנע של 50 יום, באחוזים"),
    "sma200_pct":   Col("SMA200 %", "pct", None, "מרחק המחיר מהממוצע הנע של 200 יום, באחוזים"),
    "high52":       Col("52W High", "price", "price_52_week_high", "המחיר הגבוה ביותר ב-52 השבועות האחרונים"),
    "low52":        Col("52W Low", "price", "price_52_week_low", "המחיר הנמוך ביותר ב-52 השבועות האחרונים"),
    "high52_pct":   Col("52W High %", "pct", None, "כמה אחוזים המחיר מתחת לשיא השנתי (0 = בשיא)"),
    "low52_pct":    Col("52W Low %", "pct", None, "כמה אחוזים המחיר מעל השפל השנתי"),
    "range52":      Col("52W Range", "range", None, "איפה המחיר נמצא בין השפל (0%) לשיא (100%) של השנה"),
    "atr":          Col("ATR", "price", "ATR", "טווח תנועה יומי ממוצע (14 יום), בדולרים"),
    "atr_pct":      Col("ATR %", "pct", "ATRP", "טווח תנועה יומי ממוצע כאחוז מהמחיר"),
    "beta":         Col("Beta", "ratio", "beta_1_year",
                        "בטא: כמה המניה זזה ביחס לשוק. 1 = כמו השוק, 2 = כפול מהשוק"),

    # ── Events / analysts ───────────────────────────────────────────────────
    "earnings_date": Col("Earnings", "date", "earnings_release_next_date", "תאריך הדוח הכספי הבא"),
    "days_to_earnings": Col("Earn Days", "days", None, "כמה ימים נשארו עד הדוח הבא"),
    "recom":        Col("Recom", "ratio", "recommendation_mark",
                        "המלצת אנליסטים ממוצעת: 1 = קנייה חזקה, 3 = החזק, 5 = מכירה"),
    "target":       Col("Target Price", "price", "price_target_average", "מחיר יעד ממוצע של אנליסטים"),
    "target_upside": Col("Target %", "pct", None, "כמה אחוזים מחיר היעד של האנליסטים מעל המחיר הנוכחי"),

    # ── HalgOM scores (scores.py) ───────────────────────────────────────────
    "score":          Col("Score", "score", None,
                          "ציון משוקלל 0–100: ממוצע ציוני Value, Quality ו-Momentum. 100 = הכי טוב ביקום"),
    "score_value":    Col("Value", "score", None, "ציון זול/יקר 0–100 לפי P/E, P/S, P/B, EV/EBITDA ו-P/FCF"),
    "score_quality":  Col("Quality", "score", None, "ציון איכות 0–100 לפי ROE, ROIC, מרווחים וחוב"),
    "score_momentum": Col("Momentum", "score", None,
                          "ציון מומנטום 0–100 לפי תשואות 3/6/12 חודשים וקרבה לשיא השנתי"),

    # ── Internal ────────────────────────────────────────────────────────────
    "tv_symbol":    Col("Symbol", "text", None, "סימול מלא ב-TradingView (בורסה:סימול)"),
}

# TradingView fields fetched in addition to COLUMNS — used for cleaning only.
EXTRA_TV_FIELDS = ("is_primary", "typespecs", "currency")


def tv_fields() -> dict[str, str]:
    """Map TradingView field -> column key, for every column fetched from TradingView."""
    out = {c.tv: k for k, c in COLUMNS.items() if c.tv}
    assert len(out) == sum(1 for c in COLUMNS.values() if c.tv), "duplicate TradingView field"
    return out


# Finviz-style table views. "ticker" is always first and pinned by the UI.
_TAIL = ["price", "change", "volume"]
VIEWS: dict[str, list[str]] = {
    "Overview":    ["ticker", "company", "sector", "industry", "country", "market_cap", "pe", *_TAIL],
    "Valuation":   ["ticker", "market_cap", "pe", "fwd_pe", "peg", "ps", "pb", "pocf", "pfcf", "ev_ebitda",
                    "eps_growth", "sales_growth", *_TAIL],
    "Financial":   ["ticker", "market_cap", "div_yield", "payout", "roa", "roe", "roic", "curr_ratio",
                    "quick_ratio", "lt_debt_eq", "debt_eq", "gross_margin", "oper_margin", "net_margin",
                    "earnings_date", *_TAIL],
    "Performance": ["ticker", "perf_w", "perf_m", "perf_q", "perf_h", "perf_ytd", "perf_y", "vol_w", "vol_m",
                    "rel_volume", "avg_volume", *_TAIL],
    "Technical":   ["ticker", "beta", "atr_pct", "sma20_pct", "sma50_pct", "sma200_pct", "high52_pct",
                    "low52_pct", "range52", "rsi", "gap", "rel_volume", *_TAIL],
    "Scores":      ["ticker", "company", "sector", "score", "score_value", "score_quality", "score_momentum",
                    "market_cap", "days_to_earnings", *_TAIL],
}

DEFAULT_SORT = ("market_cap", False)
