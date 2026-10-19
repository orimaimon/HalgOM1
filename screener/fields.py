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
    kind: str  # text | price | pct | ratio | int | money | count | date | days | score | range | flag
    source: str = "derived"
    help: str = ""


# Candlestick patterns TradingView detects on the latest daily bar:
#   key -> (field, label, bias, short tag for the Signals column, Hebrew help)
# bias: "+" bullish, "-" bearish, "" neutral (the Signals column shows tag + bias, e.g. "Engulfing+").
# Values are 1 (pattern present) or 0.
_CANDLE_NOTE = " (זוהה על הנר היומי האחרון; במהלך המסחר הנר עוד לא נסגר ויכול להשתנות)"
CANDLES: dict[str, tuple[str, str, str, str]] = {
    "cdl_engulfing_bull": ("Bullish Engulfing", "+", "Engulfing",
                           "נר עולה גדול שבולע את הנר היורד של אתמול — סימן אפשרי להיפוך כלפי מעלה"),
    "cdl_engulfing_bear": ("Bearish Engulfing", "-", "Engulfing",
                           "נר יורד גדול שבולע את הנר העולה של אתמול — סימן אפשרי להיפוך כלפי מטה"),
    "cdl_hammer": ("Hammer", "+", "Hammer",
                           "פטיש: גוף קטן למעלה וצל תחתון ארוך אחרי ירידה — הקונים החזירו את המחיר למעלה"),
    "cdl_inv_hammer": ("Inverted Hammer", "+", "Inv Hammer",
                           "פטיש הפוך: גוף קטן למטה וצל עליון ארוך אחרי ירידה — ניסיון ראשון של הקונים"),
    "cdl_hanging_man": ("Hanging Man", "-", "Hanging Man",
                           "איש תלוי: צורת פטיש אחרי עלייה — סימן אזהרה להיחלשות"),
    "cdl_shooting_star": ("Shooting Star", "-", "Shooting Star",
                           "כוכב נופל: צל עליון ארוך אחרי עלייה — המוכרים דחפו את המחיר חזרה למטה"),
    "cdl_morning_star": ("Morning Star", "+", "Morning Star",
                           "כוכב בוקר: תבנית של 3 נרות בתחתית ירידה — היפוך כלפי מעלה"),
    "cdl_evening_star": ("Evening Star", "-", "Evening Star",
                           "כוכב ערב: תבנית של 3 נרות בראש עלייה — היפוך כלפי מטה"),
    "cdl_3_soldiers": ("Three White Soldiers", "+", "3 Soldiers",
                           "שלושה חיילים לבנים: 3 נרות עולים חזקים ברצף — מומנטום קנייה"),
    "cdl_3_crows": ("Three Black Crows", "-", "3 Crows",
                           "שלושה עורבים שחורים: 3 נרות יורדים חזקים ברצף — מומנטום מכירה"),
    "cdl_harami_bull": ("Bullish Harami", "+", "Harami",
                           "הראמי עולה: נר קטן בתוך הגוף של נר יורד גדול — הירידה מאבדת כוח"),
    "cdl_harami_bear": ("Bearish Harami", "-", "Harami",
                           "הראמי יורד: נר קטן בתוך הגוף של נר עולה גדול — העלייה מאבדת כוח"),
    "cdl_kicking_bull": ("Bullish Kicking", "+", "Kicking",
                           "בעיטה עולה: פער פתיחה חד כלפי מעלה אחרי נר יורד מלא — היפוך חזק"),
    "cdl_kicking_bear": ("Bearish Kicking", "-", "Kicking",
                           "בעיטה יורדת: פער פתיחה חד כלפי מטה אחרי נר עולה מלא — היפוך חזק"),
    "cdl_baby_bull": ("Bullish Abandoned Baby", "+", "Aband. Baby",
                           "תינוק נטוש עולה: דוג'י מבודד בפערים בתחתית — היפוך נדיר וחזק"),
    "cdl_baby_bear": ("Bearish Abandoned Baby", "-", "Aband. Baby",
                           "תינוק נטוש יורד: דוג'י מבודד בפערים בראש — היפוך נדיר וחזק"),
    "cdl_marubozu_white": ("White Marubozu", "+", "Marubozu",
                           "מרובוזו לבן: נר עולה מלא כמעט בלי צללים — הקונים שלטו כל היום"),
    "cdl_marubozu_black": ("Black Marubozu", "-", "Marubozu",
                           "מרובוזו שחור: נר יורד מלא כמעט בלי צללים — המוכרים שלטו כל היום"),
    "cdl_doji": ("Doji", "", "Doji",
                           "דוג'י: פתיחה וסגירה כמעט באותו מחיר — חוסר הכרעה בין קונים למוכרים"),
    "cdl_doji_dragonfly": ("Dragonfly Doji", "+", "Dragonfly",
                           "דוג'י שפירית: צל תחתון ארוך וסגירה בגבוה — דחייה של מחירים נמוכים"),
    "cdl_doji_gravestone": ("Gravestone Doji", "-", "Gravestone",
                           "דוג'י מצבה: צל עליון ארוך וסגירה בנמוך — דחייה של מחירים גבוהים"),
}

# Major indexes, from TradingView's `indexes` field (list of {name, proname}): proname -> label.
INDEXES: dict[str, str] = {
    "SP:SPX": "S&P 500",
    "NASDAQ:NDX": "Nasdaq 100",
    "DJ:DJI": "DJIA",
    "TVC:RUI": "Russell 1000",
    "TVC:RUT": "Russell 2000",
}

COLUMNS: dict[str, Col] = {
    # ── Descriptive ─────────────────────────────────────────────────────────
    "ticker":       Col("Ticker", "text", "provider", "סימול המניה בבורסה"),
    "company":      Col("Company", "text", "provider", "שם החברה"),
    "exchange":     Col("Exchange", "text", "provider", "הבורסה שבה המניה נסחרת"),
    "sector":       Col("Sector", "text", "provider", "הסקטור של החברה (סיווג TradingView)"),
    "industry":     Col("Industry", "text", "provider", "תעשייה — סיווג מפורט יותר מהסקטור"),
    "country":      Col("Country", "text", "provider", "מדינת המטה של החברה"),
    "sec_type":     Col("Type", "text", "provider", "stock = מניה רגילה, dr = ADR (תעודת מניה זרה)"),
    "indexes":      Col("Index", "text", "provider", "מדדים מרכזיים שהמניה חברה בהם: S&P 500, Nasdaq 100, DJIA (דאו ג'ונס), Russell 1000/2000"),
    "employees":    Col("Employees", "int", "provider", "מספר עובדים"),

    # ── Price & volume ──────────────────────────────────────────────────────
    "price":        Col("Price", "price", "provider", "מחיר אחרון (בעיכוב של כ-15 דקות)"),
    "adj_price":    Col("Adj Price", "price", "derived", "מחיר מתואם (לפיצולים ודיבידנדים). למחיר של היום הוא זהה ל-Price — "
                        "תיאום משנה רק מחירי עבר, למשל בגרף או בחישוב תשואה"),
    "change":       Col("Change", "pct", "provider", "שינוי יומי באחוזים"),
    "change_abs":   Col("Change $", "price", "provider", "שינוי יומי בדולרים"),
    "gap":          Col("Gap", "pct", "provider", "פער פתיחה: הפתיחה היום מול הסגירה אתמול"),
    "premarket":    Col("Pre-market", "pct", "provider", "שינוי במסחר שלפני הפתיחה"),
    "postmarket":   Col("After-hours", "pct", "provider", "שינוי במסחר שאחרי הסגירה"),
    "volume":       Col("Volume", "count", "provider", "מחזור מניות היום"),
    "avg_volume":   Col("Avg Volume", "count", "provider", "מחזור ממוצע ב-30 הימים האחרונים"),
    "rel_volume":   Col("Rel Volume", "ratio", "provider", "מחזור יחסי: מחזור היום חלקי הממוצע. 2 = פי שניים מהרגיל"),
    "dollar_volume": Col("Dollar Vol", "money", "derived", "נזילות: מחזור ממוצע כפול מחיר, כלומר כמה דולרים נסחרים ביום"),
    "market_cap":   Col("Market Cap", "money", "provider", "שווי שוק: מחיר כפול מספר המניות"),
    "float_shares": Col("Float", "count", "provider", "מניות שנסחרות בפועל בציבור"),
    "shares_out":   Col("Shares Out", "count", "provider", "סך המניות המונפקות"),

    # ── Valuation ───────────────────────────────────────────────────────────
    "pe":           Col("P/E", "ratio", "provider", "מכפיל רווח: כמה שנות רווח משלמים על המניה. נמוך = זול יותר"),
    "eps_next_fy":  Col("EPS next Y", "price", "provider", "תחזית אנליסטים לרווח למניה בשנה הבאה"),
    "fwd_pe":       Col("Fwd P/E", "ratio", "derived", "מכפיל רווח עתידי: מחיר חלקי תחזית הרווח לשנה הבאה"),
    "peg":          Col("PEG", "ratio", "provider", "מכפיל רווח חלקי קצב הצמיחה. מתחת ל-1 נחשב זול ביחס לצמיחה"),
    "ps":           Col("P/S", "ratio", "provider", "מכפיל מכירות: שווי שוק חלקי ההכנסות"),
    "pb":           Col("P/B", "ratio", "provider", "מכפיל הון: שווי שוק חלקי ההון העצמי"),
    "pocf":         Col("P/OCF", "ratio", "provider", "מחיר חלקי תזרים המזומנים מפעילות שוטפת"),
    "pfcf":         Col("P/FCF", "ratio", "provider", "מחיר חלקי תזרים המזומנים החופשי"),
    "ev_ebitda":    Col("EV/EBITDA", "ratio", "provider", "שווי פעילות חלקי EBITDA. משווה חברות בלי השפעת החוב"),
    "eps":          Col("EPS", "price", "provider", "רווח למניה ב-12 החודשים האחרונים"),
    "eps_growth":   Col("EPS growth", "pct", "provider", "צמיחת הרווח למניה ב-12 החודשים האחרונים מול השנה הקודמת"),
    "eps_growth_q": Col("EPS Q/Q", "pct", "provider", "צמיחת הרווח למניה ברבעון האחרון מול אותו רבעון אשתקד"),
    "revenue":      Col("Sales", "money", "provider", "הכנסות ב-12 החודשים האחרונים"),
    "sales_growth": Col("Sales growth", "pct", "provider", "צמיחת ההכנסות ב-12 החודשים האחרונים מול השנה הקודמת"),
    "sales_growth_q": Col("Sales Q/Q", "pct", "provider", "צמיחת ההכנסות ברבעון האחרון מול אותו רבעון אשתקד"),

    # ── Financial ───────────────────────────────────────────────────────────
    "roa":          Col("ROA", "pct", "provider", "תשואה על הנכסים: רווח חלקי סך הנכסים"),
    "roe":          Col("ROE", "pct", "provider", "תשואה על ההון: רווח חלקי ההון העצמי. מעל 15% נחשב טוב"),
    "roic":         Col("ROIC", "pct", "provider", "תשואה על ההון המושקע (הון + חוב)"),
    "curr_ratio":   Col("Curr R", "ratio", "provider", "יחס שוטף: נכסים שוטפים חלקי התחייבויות שוטפות. מעל 1 = יכולת לשלם התחייבויות קצרות"),
    "quick_ratio":  Col("Quick R", "ratio", "provider", "יחס מהיר: כמו יחס שוטף, בלי מלאי"),
    "lt_debt_eq":   Col("LTDebt/Eq", "ratio", "provider", "חוב לטווח ארוך חלקי ההון העצמי"),
    "debt_eq":      Col("Debt/Eq", "ratio", "provider", "סך החוב חלקי ההון העצמי. נמוך = פחות ממונף"),
    "gross_margin": Col("Gross M", "pct", "provider", "שיעור רווח גולמי"),
    "oper_margin":  Col("Oper M", "pct", "provider", "שיעור רווח תפעולי"),
    "net_margin":   Col("Profit M", "pct", "provider", "שיעור רווח נקי"),
    "fcf_margin":   Col("FCF M", "pct", "provider", "תזרים מזומנים חופשי כאחוז מההכנסות"),
    "div_yield":    Col("Dividend", "pct", "provider", "תשואת דיבידנד שנתית"),
    "payout":       Col("Payout", "pct", "provider", "איזה חלק מהרווח מחולק כדיבידנד. מעל 80% עלול להיות לא בר-קיימא"),

    # ── Performance ─────────────────────────────────────────────────────────
    "perf_w":       Col("Perf Week", "pct", "provider", "תשואה בשבוע האחרון"),
    "perf_m":       Col("Perf Month", "pct", "provider", "תשואה בחודש האחרון"),
    "perf_q":       Col("Perf Quart", "pct", "provider", "תשואה ב-3 החודשים האחרונים"),
    "perf_h":       Col("Perf Half", "pct", "provider", "תשואה בחצי השנה האחרונה"),
    "perf_ytd":     Col("Perf YTD", "pct", "provider", "תשואה מתחילת השנה"),
    "perf_y":       Col("Perf Year", "pct", "provider", "תשואה בשנה האחרונה"),
    "perf_5y":      Col("Perf 5Y", "pct", "provider", "תשואה ב-5 השנים האחרונות"),
    "vol_w":        Col("Volatility W", "pct", "provider", "תנודתיות יומית ממוצעת בשבוע האחרון"),
    "vol_m":        Col("Volatility M", "pct", "provider", "תנודתיות יומית ממוצעת בחודש האחרון"),

    # ── Technical ───────────────────────────────────────────────────────────
    "rsi":          Col("RSI", "ratio", "provider", "RSI(14): מעל 70 = קנוי יתר (עלה מהר מדי), מתחת ל-30 = מכור יתר (ירד מהר מדי)"),
    "sma20":        Col("SMA20", "price", "provider", "ממוצע נע פשוט של 20 יום"),
    "sma50":        Col("SMA50", "price", "provider", "ממוצע נע פשוט של 50 יום"),
    "sma150":       Col("SMA150", "price", "provider", "ממוצע נע פשוט של 150 יום (כ-30 שבועות, בשימוש אצל Minervini)"),
    "sma200":       Col("SMA200", "price", "provider", "ממוצע נע פשוט של 200 יום"),
    "sma20_pct":    Col("SMA20 %", "pct", "derived", "מרחק המחיר מהממוצע הנע של 20 יום, באחוזים"),
    "sma50_pct":    Col("SMA50 %", "pct", "derived", "מרחק המחיר מהממוצע הנע של 50 יום, באחוזים"),
    "sma150_pct":   Col("SMA150 %", "pct", "derived", "מרחק המחיר מהממוצע הנע של 150 יום, באחוזים"),
    "sma200_pct":   Col("SMA200 %", "pct", "derived", "מרחק המחיר מהממוצע הנע של 200 יום, באחוזים"),
    "close_prev":   Col("Prev Close", "price", "provider", "מחיר הסגירה של יום המסחר הקודם"),
    "sma20_prev":   Col("SMA20 (prev)", "price", "provider", "הממוצע הנע של 20 יום כפי שהיה אתמול — לזיהוי חציות"),
    "sma50_prev":   Col("SMA50 (prev)", "price", "provider", "הממוצע הנע של 50 יום כפי שהיה אתמול — לזיהוי חציות"),
    "sma200_prev":  Col("SMA200 (prev)", "price", "provider", "הממוצע הנע של 200 יום כפי שהיה אתמול — לזיהוי חציות"),

    # ── Weekly timeframe (TradingView "|1W" fields) ─────────────────────────
    "close_w":      Col("Close (W)", "price", "provider", "מחיר הסגירה בגרף השבועי (בשבוע פתוח = המחיר האחרון)"),
    "rsi_w":        Col("RSI (W)", "ratio", "provider", "RSI(14) בגרף השבועי — מגמה ארוכה יותר ופחות רעש מה-RSI היומי"),
    "sma50_w":      Col("SMA50 (W)", "price", "provider", "ממוצע נע של 50 שבועות (כשנה) בגרף השבועי"),
    "sma50w_pct":   Col("SMA50W %", "pct", "derived", "מרחק הסגירה השבועית מהממוצע הנע של 50 שבועות, באחוזים"),

    # ── Ranges & breakouts (TradingView 1M / 3M / 6M windows include today's bar; "[1]" = as of yesterday) ─
    "high_5d":      Col("5D High", "price", "provider", "המחיר הגבוה ביותר ב-5 ימי המסחר האחרונים"),
    "low_5d":       Col("5D Low", "price", "provider", "המחיר הנמוך ביותר ב-5 ימי המסחר האחרונים"),
    "high_1m":      Col("1M High", "price", "provider", "המחיר הגבוה ביותר בחודש האחרון"),
    "low_1m":       Col("1M Low", "price", "provider", "המחיר הנמוך ביותר בחודש האחרון"),
    "high_1m_prev": Col("1M High (prev)", "price", "provider", "השיא החודשי כפי שהיה אתמול — סגירה מעליו היום = פריצה"),
    "high_3m_prev": Col("3M High (prev)", "price", "provider", "שיא 3 החודשים כפי שהיה אתמול"),
    "high_6m_prev": Col("6M High (prev)", "price", "provider", "שיא חצי השנה כפי שהיה אתמול"),
    "high52_prev":  Col("52W High (prev)", "price", "provider", "השיא השנתי כפי שהיה אתמול"),
    "low_1m_prev":  Col("1M Low (prev)", "price", "provider", "השפל החודשי כפי שהיה אתמול — סגירה מתחתיו היום = שבירה כלפי מטה"),
    "low_3m_prev":  Col("3M Low (prev)", "price", "provider", "שפל 3 החודשים כפי שהיה אתמול"),
    "low_6m_prev":  Col("6M Low (prev)", "price", "provider", "שפל חצי השנה כפי שהיה אתמול"),
    "low52_prev":   Col("52W Low (prev)", "price", "provider", "השפל השנתי כפי שהיה אתמול"),
    "range_5d_pct": Col("5D Range", "pct", "derived", "טווח המסחר של 5 הימים האחרונים: כמה אחוזים הגבוה מעל הנמוך. נמוך = התכנסות"),
    "range_1m_pct": Col("1M Range", "pct", "derived", "טווח המסחר של החודש האחרון: כמה אחוזים הגבוה מעל הנמוך. נמוך = התכנסות לפני תנועה"),
    "high52":       Col("52W High", "price", "provider", "המחיר הגבוה ביותר ב-52 השבועות האחרונים"),
    "low52":        Col("52W Low", "price", "provider", "המחיר הנמוך ביותר ב-52 השבועות האחרונים"),
    "high52_pct":   Col("52W High %", "pct", "derived", "כמה אחוזים המחיר מתחת לשיא השנתי (0 = בשיא)"),
    "low52_pct":    Col("52W Low %", "pct", "derived", "כמה אחוזים המחיר מעל השפל השנתי"),
    "range52":      Col("52W Range", "range", "derived", "איפה המחיר נמצא בין השפל (0%) לשיא (100%) של השנה"),
    "atr":          Col("ATR", "price", "provider", "טווח תנועה יומי ממוצע (14 יום), בדולרים"),
    "atr_pct":      Col("ATR %", "pct", "provider", "טווח תנועה יומי ממוצע כאחוז מהמחיר"),
    "beta":         Col("Beta", "ratio", "provider", "בטא: כמה המניה זזה ביחס לשוק. 1 = כמו השוק, 2 = כפול מהשוק"),

    # ── Events / analysts ───────────────────────────────────────────────────
    "earnings_date": Col("Earnings", "date", "provider", "תאריך הדוח הכספי הבא"),
    "days_to_earnings": Col("Earn Days", "days", "derived", "כמה ימים נשארו עד הדוח הבא"),
    "recom":        Col("Recom", "ratio", "provider", "המלצת אנליסטים ממוצעת: 1 = קנייה חזקה, 3 = החזק, 5 = מכירה"),
    "target":       Col("Target Price", "price", "provider", "מחיר יעד ממוצע של אנליסטים"),
    "target_upside": Col("Target %", "pct", "derived", "כמה אחוזים מחיר היעד של האנליסטים מעל המחיר הנוכחי"),

    # ── HalgOM scores (scores.py) ───────────────────────────────────────────
    "score":          Col("Score", "score", "derived", "ציון משוקלל 0–100: ממוצע ציוני Value, Quality ו-Momentum. 100 = הכי טוב ביקום"),
    "score_value":    Col("Value", "score", "derived", "ציון זול/יקר 0–100 לפי P/E, P/S, P/B, EV/EBITDA ו-P/FCF"),
    "score_quality":  Col("Quality", "score", "derived", "ציון איכות 0–100 לפי ROE, ROIC, מרווחים וחוב"),
    "score_momentum": Col("Momentum", "score", "derived", "ציון מומנטום 0–100 לפי תשואות 3/6/12 חודשים וקרבה לשיא השנתי"),

    # ── Signals (signals.py) ────────────────────────────────────────────────
    "signals":      Col("Signals", "text", "derived",
                        "כל האותות שהמניה עברה היום, בקיצור: תבניות נרות (+ עולה, - יורד), מחזור ×2.8 (Vol 2.8x) = מחזור "
                        "פי 2.8 מהרגיל, פריצה חודשית (BO 1M) = פריצה לשיא חודשי, שבירה (BD) = שבירה לשפל, "
                        "Golden/Death Cross, התכנסות חודשית (Tight 1M) = התכנסות"),
    "n_signals":    Col("# Signals", "int", "derived", "כמה אותות המניה עברה היום. מיון לפי העמודה מציג קודם את הבולטות"),
    "new":          Col("New", "text", "derived", "NEW = המניה עומדת בפילטרים היום ולא עמדה בהם ב-snapshot של יום המסחר הקודם"),

    # ── Internal ────────────────────────────────────────────────────────────
    "tv_symbol":    Col("Symbol", "text", "derived", "סימול מלא ב-TradingView (בורסה:סימול)"),
}
COLUMNS.update({k: Col(label, "flag", "provider", help_ + _CANDLE_NOTE) for k, (label, _, _, help_) in CANDLES.items()})

# Columns of the Groups page (one row per sector / industry). Performance columns reuse COLUMNS.
GROUP_COLUMNS: dict[str, Col] = {
    "group":      Col("Name", "text", "derived", "שם הסקטור או התעשייה"),
    "stocks":     Col("Stocks", "int", "derived", "מספר המניות בקבוצה"),
    "market_cap": Col("Market Cap", "money", "derived", "שווי השוק הכולל של כל המניות בקבוצה"),
    "change":     Col("Change", "pct", "derived", "שינוי יומי ממוצע של המניות בקבוצה"),
    **{k: Col(COLUMNS[k].label, "pct", "derived", COLUMNS[k].help + " — ממוצע של המניות בקבוצה")
       for k in ("perf_w", "perf_m", "perf_q", "perf_h", "perf_ytd", "perf_y")},
    "rel_volume": Col("Rel Volume", "ratio", "derived", "מחזור יחסי ממוצע בקבוצה. מעל 1 = פעילות גבוהה מהרגיל"),
    "pct_above_sma50": Col("% > SMA50", "pct", "derived", "איזה אחוז מהמניות בקבוצה נסחרות מעל הממוצע הנע של 50 יום"),
}



# Finviz-style table views. "ticker" is always first and pinned by the UI.
_TAIL = ["price", "change", "volume"]
VIEWS: dict[str, list[str]] = {
    "Overview":    ["ticker", "company", "sector", "industry", "country", "indexes", "market_cap", "pe", "adj_price",
                    *_TAIL],
    "Valuation":   ["ticker", "market_cap", "pe", "fwd_pe", "peg", "ps", "pb", "pocf", "pfcf", "ev_ebitda",
                    "eps_growth", "sales_growth", *_TAIL],
    "Financial":   ["ticker", "market_cap", "div_yield", "payout", "roa", "roe", "roic", "curr_ratio",
                    "quick_ratio", "lt_debt_eq", "debt_eq", "gross_margin", "oper_margin", "net_margin",
                    "earnings_date", *_TAIL],
    "Performance": ["ticker", "perf_w", "perf_m", "perf_q", "perf_h", "perf_ytd", "perf_y", "vol_w", "vol_m",
                    "rel_volume", "avg_volume", *_TAIL],
    "Technical":   ["ticker", "beta", "atr_pct", "sma20_pct", "sma50_pct", "sma150_pct", "sma200_pct", "high52_pct",
                    "low52_pct", "range52", "rsi", "rsi_w", "sma50w_pct", "gap", "rel_volume", *_TAIL],
    "Signals":     ["ticker", "company", "n_signals", "signals", "rel_volume", "range_1m_pct", "high52_pct",
                    "sma50_pct", "sma200_pct", "rsi", *_TAIL],
    "Scores":      ["ticker", "company", "sector", "score", "score_value", "score_quality", "score_momentum",
                    "market_cap", "days_to_earnings", *_TAIL],
}


def provider_columns() -> list[str]:
    """Return keys of columns that come from the data provider."""
    return [k for k, c in COLUMNS.items() if c.source == "provider"]

DEFAULT_SORT = ("market_cap", False)
