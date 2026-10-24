"""
Display language (Hebrew by default, English switch).

Only what is shown is translated. Internal values stay English: filter option labels, preset names,
view and page names, column keys and data values (sector, country). So saved screens
(_data/presets.json), the built-in presets and the tests do not depend on the language.

Hebrew style (decided 2026-10-18): Hebrew names with the familiar English abbreviation where it
helps ("מכפיל רווח (P/E)"); short table headers keep the abbreviation alone (P/E, RSI, SMA50).
Sectors and countries are translated; the 128 industries stay English (TradingView terms with no
accepted Hebrew names). Anything without a translation falls back to the English text.
"""
from __future__ import annotations

import re

HE, EN = "he", "en"
LANGS = {HE: "עברית", EN: "English"}

# ── app texts (buttons, captions, page names) ──────────────────────────────
UI: dict[str, str] = {
    "No price in screener for:": "אין בסקרינר מחיר ל:",
    "(e.g. ETFs). They will be saved, but without current value.": "(למשל קרנות סל). הן יישמרו, אבל בלי שווי נוכחי.",
    "(e.g. ETFs). They are not included in the value and summaries.": "(למשל קרנות סל). הן לא נכללות בשווי ובסיכומים.",

    "Cannot connect to TradingView right now - showing saved data from": "לא ניתן להתחבר ל-TradingView כרגע — מוצגים נתונים שמורים מ-",
    "at": "בשעה",
    "Will try again automatically in a few minutes.": "ננסה שוב אוטומטית בעוד כמה דקות.",

    "Cannot read file. Save it as CSV or xlsx in Excel and try again.": "לא ניתן לקרוא את הקובץ. שמור אותו באקסל כ-CSV או כ-xlsx ונסה שוב.",
    "Fix errors in the file and load again:": "יש לתקן בקובץ ולטעון שוב:",
    "File is empty": "הקובץ ריק",
    "Update price history": "עדכון היסטוריית מחירים",
    "No history yet — click update price history": "עדיין אין היסטוריה — לחצו על עדכון היסטוריית מחירים",
    "Updated": "עודכנו",
    "Not updated:": "לא עודכנו:",
    "No history for:": "אין היסטוריה עבור:",
    "Dividends received": "דיבידנדים שהתקבלו",
    "Portfolio:": "התיק:",
    "Portfolio value": "שווי התיק",
    "Portfolio + dividends": "תיק + דיבידנדים",
    "Net invested": "השקעה נטו",
    "No stocks to show - filters or search yielded no results": "אין מניות להצגה — הפילטרים או החיפוש לא החזירו תוצאות",
    "No stocks to show": "אין מניות להצגה",

    "Cannot save right now, please try again later": "לא ניתן לשמור כרגע, נסו שוב בעוד רגע",
    "Data was updated": "הנתונים עודכנו לפני",
    "seconds ago": "שניות",
    "List saved": "הרשימה נשמרה",
    "Added to": "נוספו ל",
    "Not found in screener": "לא נמצאו בסקרינר",
    "The list was deleted in another window": "הרשימה נמחקה בחלון אחר",
    "is a portfolio, not a watchlist": "הוא תיק, לא רשימת מעקב",
    "Trade added": "העסקה נוספה",
    "A copy of every downloaded file is also saved in:": "עותק של כל קובץ שמורד נשמר גם בתיקייה:",
    "Average cost includes fees. Splits and dividends after purchase are not yet calculated.":
        "עלות ממוצעת כוללת עמלות. ספליטים ודיבידנדים אחרי הקנייה עוד לא מחושבים.",
    "Closes the screener and the black window. Next time, open again from the desktop shortcut.":
        "סוגר את הסקרינר ואת החלון השחור. בפעם הבאה פותחים שוב מהקיצור בשולחן העבודה.",
    "Data: TradingView (delayed ~15 min). For research purposes only, not investment advice.":
        "נתונים: TradingView (בעיכוב של כ-15 דקות). הכלי לצורכי מחקר בלבד ואינו ייעוץ השקעות.",
    "Filters have been changed since the screen was selected.": "הפילטרים שונו מאז שנבחרה הסריקה.",
    "Lists are saved on this computer only. Prices are from the screener (delayed ~15 min).":
        "הרשימות נשמרות במחשב הזה בלבד. מחירים מהסקרינר (בעיכוב של כ-15 דקות).",
    "There is already a list named": "כבר יש רשימה בשם",
    "Saving with a file will replace it.": "שמירה עם קובץ תחליף אותה.",
    "Showing list": "מוצגת הרשימה", "from Portfolios page": "מדף התיקים",
    "The list is empty. Add tickers in the box below, or from the \"Add to watchlist\" button in the stock details in the screener.":
        "הרשימה ריקה. מוסיפים סימולים בתיבה למטה, או מכפתור “הוספה לרשימת מעקב” בפרטי מניה בסורק.",
    "There are currently no open positions in the portfolio.": "אין כרגע פוזיציות פתוחות בתיק.",
    "Delete the list": "למחוק את",
    "This cannot be undone. Consider exporting first.": "אי אפשר לבטל. כדאי לייצא קודם.",
    "👆 Click a row in the table to open chart and stock details": "👆 לחץ על שורה בטבלה כדי לפתוח גרף ונתונים של המניה",
    "Cannot download data, and there is no saved data from the last time. Check your internet connection and try again.": "לא ניתן להוריד נתונים, ואין נתונים שמורים מהפעם הקודמת. בדוק את החיבור לאינטרנט ונסה שוב.",

    # pages, views, filter tabs
    "Screener": "סורק", "Groups": "קבוצות", "Heatmap": "מפת חום", "Guide": "מדריך",
    "Overview": "סקירה", "Valuation": "תמחור", "Financial": "פיננסי", "Performance": "ביצועים",
    "Technical": "טכני", "Signals": "אותות", "Scores": "ציונים",
    "Descriptive": "כללי", "Fundamental": "פונדמנטלי",
    # header
    "US stocks": "מניות ארה\"ב", "in universe": "מניות ביקום", "Market data unavailable": "אין נתוני שוק כרגע",
    "Market regime · SPY ADX": "מצב השוק · ADX של SPY",
    "BULL": "שוק עולה", "BEAR": "שוק יורד", "SIDEWAYS": "דשדוש", "UNKNOWN": "לא ידוע",
    "Trend strong": "מגמה חזקה", "Trend building": "מגמה מתפתחת", "Trend weak": "מגמה חלשה",
    "Trend n/a": "אין נתון מגמה",
    "Data as of": "נתונים נכונים ל-", "Data time": "שעת הנתונים", "live": "חי", "cached": "שמור", "OFFLINE": "לא מקוון",
    "delayed": "בעיכוב", "min": "דק'", "realtime": "בזמן אמת", "end of day": "סוף יום",
    "update mode": "מצב עדכון", "delayed ~15 min": "בעיכוב של כ-15 דק'",
    "Refresh data": "רענון נתונים", "Exit": "יציאה", "Close the screener": "סגירת הסקרינר",
    "Large text": "טקסט גדול", "Active filters": "פילטרים פעילים",
    "Loading market data from TradingView… (≈15 sec)": "טוען נתוני שוק מ-TradingView… (כ-15 שניות)",
    "Try again": "נסה שוב", "Details": "פרטים",
    # toolbar
    "Screen": "סריקה", "— Custom —": "— מותאם אישית —", "Search": "חיפוש",
    "Ticker / company, or list: AAPL, MSFT": "סימול / חברה, או רשימה: AAPL, MSFT",
    "Order by": "מיון לפי", "Ascending": "סדר עולה", "Hide junk": "הסתר זבל", "Reset": "איפוס",
    "Clear all filters": "איפוס כל הפילטרים והחיפוש",
    "Save current screen as": "שמירת הסריקה הנוכחית בשם", "Save": "שמירה",
    "Delete saved screen": "מחיקת סריקה שמורה", "Delete": "מחיקה",
    "Why is this stock in the screen?": "למה המניה בסריקה?",
    "No active filters — all stocks shown": "אין פילטרים פעילים — כל המניות מוצגות",
    "Saved screen": "הסריקה נשמרה", "Deleted": "נמחקה",
    "A screen with this name already exists. Click save again to overwrite.":
        "סריקה בשם זה כבר קיימת. לחיצה נוספת על שמירה תעדכן אותה.",
    # Hide junk rules (filters.JUNK_RULES)
    "Price under $1": "מחיר מתחת ל-$1", "Avg Volume under 50K": "מחזור ממוצע מתחת ל-50K",
    "Market Cap under $50M": "שווי שוק מתחת ל-$50M",
    "SPAC (blank-check company)": "חברת רכישה ייעודית (SPAC)",
    # filters / results
    "Filters": "פילטרים", "active": "פעילים", "results": "תוצאות", "Only new": "רק חדשות", "NEW": "חדש",
    "Export all columns": "ייצוא כל העמודות", "TradingView list": "רשימה ל-TradingView", "Tickers": "סימולים",
    # detail panel
    "Price": "מחיר", "Valuation & size": "תמחור", "Growth & quality": "צמיחה ואיכות",
    "52-Week Range": "טווח 52 שבועות",
    "Events & scores": "אירועים וציונים", "Open in TradingView": "פתיחה ב-TradingView",
    # groups / heatmap
    "Group by": "קיבוץ לפי", "Sector": "סקטור", "Industry": "תעשייה", "Averages": "ממוצעים",
    "Equal": "שווה", "Market cap": "שווי שוק", "Median": "חציון", "All sectors": "כל הסקטורים",
    "Only screener results": "רק תוצאות הסריקה", "sectors": "סקטורים", "industries": "תעשיות",
    "stocks": "מניות", "Stocks": "מניות", "Color by": "צבע לפי",
    "S&P 500": "S&P 500", "Nasdaq 100": "Nasdaq 100", "Screener results": "תוצאות הסריקה",
    "All stocks (top 1,000)": "כל המניות (1,000 הגדולות)",
    # portfolios / watchlists
    "Portfolios": "תיקים ומעקב", "List": "רשימה", "Watchlist": "רשימת מעקב", "Portfolio": "תיק",
    "New list / import a file": "רשימה חדשה / ייבוא קובץ", "List name": "שם הרשימה",
    "CSV or Excel file": "קובץ CSV או אקסל", "Sample file": "קובץ לדוגמה", "Save list": "שמירת הרשימה",
    "Empty watchlist": "רשימת מעקב ריקה", "— New watchlist —": "— רשימת מעקב חדשה —",
    "Add to watchlist": "הוספה לרשימת מעקב", "Add tickers": "הוספת סימולים", "Add": "הוספה",
    "Remove ticker": "הסרת סימול", "Remove": "הסרה", "Add trade": "הוספת עסקה", "Trades": "עסקאות",
    "Restore screen": "החזרת הסריקה", "Back to all stocks": "חזרה לכל המניות", "(changed)": "(שונה)",
    "Show in screener": "הצגה בסורק", "Export list": "ייצוא הרשימה", "Delete list": "מחיקת הרשימה",
    "Ticker": "סימול", "Date": "תאריך", "Quantity": "כמות", "Side": "פעולה", "buy": "קנייה", "sell": "מכירה",
    "Fees": "עמלות", "Market value": "שווי נוכחי", "Cost": "עלות", "Unrealized P&L": "רווח/הפסד פתוח",
    "Realized P&L": "רווח/הפסד ממומש", "today": "היום", "Avg cost": "עלות ממוצעת", "P&L": "רווח/הפסד",
    "P&L %": "רווח/הפסד %", "Weight": "משקל", "Annual return": "תשואה שנתית", "First buy": "קנייה ראשונה",
    "Added": "תאריך הוספה", "Price then": "מחיר בהוספה", "Since added": "מאז ההוספה",
    # export sheet
    "Results": "תוצאות", "Status / Action": "סטטוס / פעולה", "Target Entry": "מחיר כניסה",
    "Notes": "הערות", "Filter": "פילטר", "Value": "ערך", "HalgOM Screener export": "ייצוא HalgOM Screener",
"Exported": "יוצא בתאריך", "(custom)": "(מותאם אישית)",
    "(none)": "(אין)", "Source: TradingView scanner (delayed ~15 min). Not investment advice.":
        "מקור: סורק TradingView (בעיכוב של כ-15 דקות). אין לראות בזה ייעוץ השקעות.",
    "Track record": "תעודת ביצועים",
    "Calculating track record…": "מחשב תעודת ביצועים…",
    "Past performance only. Does not guarantee future results and is not investment advice.":
        "ביצועי עבר בלבד. אינם מבטיחים תוצאות בעתיד ואינם ייעוץ השקעות.",
    "Still few data points — numbers will stabilize over time":
        "מעט נתונים עדיין — המספרים יתייצבו עם הזמן",
    "Over SPY": "מעל SPY",
    "Over market median": "מעל חציון השוק",
    "Pending": "ממתין",
    "Entries": "כניסות",
    "20-day median": "חציון 20 יום",
    "% positive": "% חיוביות",
    "20-day excess": "עודף 20 יום",
}

# ── table headers (short) ──────────────────────────────────────────────────
COLUMNS_HE: dict[str, str] = {
    "ticker": "סימול", "company": "חברה", "exchange": "בורסה", "sector": "סקטור", "industry": "תעשייה",
    "country": "מדינה", "sec_type": "סוג", "indexes": "מדדים", "employees": "עובדים",
    "price": "מחיר", "adj_price": "מחיר מתואם", "change": "שינוי", "change_abs": "שינוי $", "gap": "פער פתיחה",
    "premarket": "טרום מסחר", "postmarket": "אחרי המסחר", "volume": "מחזור", "avg_volume": "מחזור ממוצע",
    "rel_volume": "מחזור יחסי", "dollar_volume": "מחזור $", "market_cap": "שווי שוק",
    "float_shares": "מניות צפות", "shares_out": "מניות מונפקות",
    "pe": "P/E", "pe_vs_industry": "P/E מול תעשייה", "eps_next_fy": "EPS שנה הבאה", "fwd_pe": "P/E עתידי", "peg": "PEG", "ps": "P/S", "ps_vs_industry": "P/S מול תעשייה", "pb": "P/B",
    "pocf": "P/OCF", "pfcf": "P/FCF", "ev_ebitda": "EV/EBITDA", "eps": "EPS", "eps_growth": "צמיחת EPS",
    "eps_growth_q": "EPS רבעוני", "revenue": "הכנסות", "sales_growth": "צמיחת הכנסות",
    "sales_growth_vs_sector": "צמיחת הכנסות מול סקטור", "sales_growth_q": "הכנסות רבעוני",
    "roa": "ROA", "roe": "ROE", "roic": "ROIC", "curr_ratio": "יחס שוטף", "quick_ratio": "יחס מהיר",
    "lt_debt_eq": "חוב ארוך/הון", "debt_eq": "חוב/הון", "gross_margin": "רווח גולמי %",
    "oper_margin": "רווח תפעולי %", "net_margin": "רווח נקי %", "fcf_margin": "FCF %",
    "div_yield": "דיבידנד", "payout": "שיעור חלוקה",
    "perf_w": "תשואה שבוע", "perf_m": "תשואה חודש", "perf_q": "תשואה רבעון", "perf_h": "תשואה חצי שנה",
    "perf_ytd": "תשואה מתחילת שנה", "perf_y": "תשואה שנה", "perf_5y": "תשואה 5 שנים",
    "vol_w": "תנודתיות שבוע", "vol_m": "תנודתיות חודש",
    "rsi": "RSI", "sma20": "SMA20", "sma50": "SMA50", "sma150": "SMA150", "sma200": "SMA200",
    "sma20_pct": "SMA20 %", "sma50_pct": "SMA50 %", "sma150_pct": "SMA150 %", "sma200_pct": "SMA200 %",
    "close_prev": "סגירה קודמת", "sma20_prev": "SMA20 (אתמול)", "sma50_prev": "SMA50 (אתמול)",
    "sma200_prev": "SMA200 (אתמול)", "close_w": "סגירה (שבועי)", "rsi_w": "RSI (שבועי)",
    "sma50_w": "SMA50 (שבועי)", "sma50w_pct": "SMA50 שבועי %",
    "high_5d": "שיא 5 ימים", "low_5d": "שפל 5 ימים", "high_1m": "שיא חודשי", "low_1m": "שפל חודשי",
    "high_1m_prev": "שיא חודשי (אתמול)", "high_3m_prev": "שיא 3 חודשים (אתמול)",
    "high_6m_prev": "שיא חצי שנה (אתמול)", "high52_prev": "שיא שנתי (אתמול)",
    "low_1m_prev": "שפל חודשי (אתמול)", "low_3m_prev": "שפל 3 חודשים (אתמול)",
    "low_6m_prev": "שפל חצי שנה (אתמול)", "low52_prev": "שפל שנתי (אתמול)",
    "range_5d_pct": "טווח 5 ימים", "range_1m_pct": "טווח חודשי",
    "high52": "שיא שנתי", "low52": "שפל שנתי", "high52_pct": "מהשיא %", "low52_pct": "מהשפל %",
    "range52": "טווח שנתי", "atr": "ATR", "atr_pct": "ATR %", "beta": "בטא",
    "earnings_date": "דוח הבא", "days_to_earnings": "ימים לדוח", "recom": "המלצה", "target": "מחיר יעד",
    "target_upside": "יעד %", "score": "ציון", "score_value": "ערך", "score_quality": "איכות",
    "score_momentum": "מומנטום", "signals": "אותות", "n_signals": "# אותות", "new": "חדש",
    "tv_symbol": "סימול מלא",
}

GROUP_COLUMNS_HE: dict[str, str] = {
    "group": "שם", "stocks": "מניות", "market_cap": "שווי שוק", "change": "שינוי",
    "perf_w": "תשואה שבוע", "perf_m": "תשואה חודש", "perf_q": "תשואה רבעון", "perf_h": "תשואה חצי שנה",
    "perf_ytd": "תשואה מתחילת שנה", "perf_y": "תשואה שנה", "rel_volume": "מחזור יחסי",
    "pct_above_sma50": "% מעל SMA50",
}

# ── filter names ───────────────────────────────────────────────────────────
FILTERS_HE: dict[str, str] = {
    "exchange": "בורסה", "sector": "סקטור", "industry": "תעשייה", "country": "מדינה", "index": "מדד",
    "market_cap": "שווי שוק", "div_yield": "תשואת דיבידנד", "price": "מחיר", "avg_volume": "מחזור ממוצע",
    "rel_volume": "מחזור יחסי", "volume": "מחזור היום", "dollar_volume": "מחזור בדולרים",
    "float_shares": "מניות צפות (Float)", "shares_out": "מניות מונפקות", "earnings": "תאריך דוח", "recom": "המלצת אנליסטים",
    "target_upside": "מחיר יעד",
    "pe": "מכפיל רווח (P/E)", "pe_vs_industry": "P/E מול תעשייה", "fwd_pe": "מכפיל רווח עתידי (Fwd P/E)", "peg": "מכפיל רווח לצמיחה (PEG)",
    "ps": "מכפיל מכירות (P/S)", "ps_vs_industry": "P/S מול תעשייה", "pb": "מכפיל הון (P/B)", "pfcf": "מחיר לתזרים חופשי (P/FCF)",
    "ev_ebitda": "EV/EBITDA", "eps_growth": "צמיחת רווח למניה (EPS) שנתי",
    "eps_growth_q": "צמיחת רווח למניה (EPS) רבעוני", "sales_growth": "צמיחת הכנסות שנתי", "sales_growth_vs_sector": "צמיחת הכנסות מול סקטור",
    "sales_growth_q": "צמיחת הכנסות רבעוני", "roa": "תשואה על הנכסים (ROA)", "roe": "תשואה על ההון (ROE)",
    "roic": "תשואה על ההון המושקע (ROIC)", "curr_ratio": "יחס שוטף", "quick_ratio": "יחס מהיר",
    "lt_debt_eq": "חוב לטווח ארוך להון", "debt_eq": "חוב להון", "gross_margin": "שיעור רווח גולמי",
    "oper_margin": "שיעור רווח תפעולי", "net_margin": "שיעור רווח נקי", "payout": "שיעור חלוקה (Payout)",
    "change": "שינוי יומי", "perf_w": "תשואה (שבוע)", "perf_m": "תשואה (חודש)", "perf_q": "תשואה (רבעון)",
    "perf_h": "תשואה (חצי שנה)", "perf_ytd": "תשואה (מתחילת השנה)", "perf_y": "תשואה (שנה)",
    "rsi": "RSI (14)", "sma20": "ממוצע נע 20 יום (SMA20)", "sma50": "ממוצע נע 50 יום (SMA50)",
    "sma150": "ממוצע נע 150 יום (SMA150)", "sma200": "ממוצע נע 200 יום (SMA200)",
    "high52": "שיא 52 שבועות", "low52": "שפל 52 שבועות", "gap": "פער פתיחה (Gap)", "beta": "בטא (Beta)",
    "atr_pct": "תנועה יומית ממוצעת (ATR %)", "vol_w": "תנודתיות (שבוע)", "vol_m": "תנודתיות (חודש)",
    "rsi_w": "RSI (14) שבועי", "sma50_w": "ממוצע נע 50 שבועות",
    "candle": "תבנית נרות", "breakout": "פריצה / שבירה", "consolidation": "התכנסות",
    "vol_spike": "קפיצת מחזור", "ma_cross": "חציית ממוצעים", "ma_stack": "סדר הממוצעים",
    "n_signals": "מספר אותות", "score": "ציון HalgOM", "score_value": "ציון ערך (Value)",
    "score_quality": "ציון איכות (Quality)", "score_momentum": "ציון מומנטום (Momentum)",
}

# ── built-in presets ───────────────────────────────────────────────────────
PRESETS_HE: dict[str, str] = {
    "Value — cheap & profitable": "ערך — זולות ורווחיות", "Quality growth": "צמיחה איכותית",
    "Momentum leaders": "מובילות מומנטום", "Oversold in uptrend": "מכירת יתר במגמת עלייה",
    "52-week high breakout": "פריצה לשיא שנתי", "Dividend — stable": "דיבידנד יציב",
    "Small-cap growth": "צמיחה במניות קטנות", "Unusual volume": "מחזור חריג",
    "Stage 2 (Minervini)": "שלב 2 (Minervini)", "HalgOM top score": "ציון HalgOM גבוה",
}

# ── data values ────────────────────────────────────────────────────────────
SECTORS_HE: dict[str, str] = {
    "Commercial Services": "שירותים מסחריים", "Communications": "תקשורת",
    "Consumer Durables": "מוצרי צריכה בני-קיימא", "Consumer Non-Durables": "מוצרי צריכה שוטפת",
    "Consumer Services": "שירותי צרכנות", "Distribution Services": "הפצה",
    "Electronic Technology": "טכנולוגיה אלקטרונית", "Energy Minerals": "אנרגיה",
    "Finance": "פיננסים", "Health Services": "שירותי בריאות", "Health Technology": "טכנולוגיה רפואית",
    "Industrial Services": "שירותים תעשייתיים", "Miscellaneous": "שונות",
    "Non-Energy Minerals": "מתכות וכרייה", "Process Industries": "תעשיות תהליך",
    "Producer Manufacturing": "ייצור תעשייתי", "Retail Trade": "קמעונאות",
    "Technology Services": "שירותי טכנולוגיה", "Transportation": "תחבורה", "Utilities": "תשתיות",
    "Government": "ממשלה", "Other": "אחר",
}
COUNTRIES_HE: dict[str, str] = {
    "Argentina": "ארגנטינה", "Australia": "אוסטרליה", "Bahamas": "איי בהאמה", "Belgium": "בלגיה",
    "Bermuda": "ברמודה", "Brazil": "ברזיל", "British Virgin Islands": "איי הבתולה הבריטיים",
    "Cambodia": "קמבודיה", "Canada": "קנדה", "Cayman Islands": "איי קיימן", "Chile": "צ'ילה", "China": "סין",
    "Colombia": "קולומביה", "Cyprus": "קפריסין", "Denmark": "דנמרק", "Finland": "פינלנד", "France": "צרפת",
    "Germany": "גרמניה", "Gibraltar": "גיברלטר", "Greece": "יוון", "Hong Kong": "הונג קונג", "India": "הודו",
    "Indonesia": "אינדונזיה", "Ireland": "אירלנד", "Israel": "ישראל", "Italy": "איטליה", "Japan": "יפן",
    "Jordan": "ירדן", "Kazakhstan": "קזחסטן", "Luxembourg": "לוקסמבורג", "Macau": "מקאו", "Malaysia": "מלזיה",
    "Mauritius": "מאוריציוס", "Mexico": "מקסיקו", "Monaco": "מונקו", "Netherlands": "הולנד",
    "New Zealand": "ניו זילנד", "Norway": "נורווגיה", "Panama": "פנמה", "Peru": "פרו",
    "Philippines": "הפיליפינים", "Puerto Rico": "פוארטו ריקו", "Singapore": "סינגפור",
    "South Africa": "דרום אפריקה", "South Korea": "דרום קוריאה", "Spain": "ספרד", "Sweden": "שוודיה",
    "Switzerland": "שווייץ", "Taiwan": "טייוואן", "Thailand": "תאילנד", "Turkey": "טורקיה",
    "United Arab Emirates": "איחוד האמירויות", "United Kingdom": "בריטניה", "United States": "ארה\"ב",
    "Uruguay": "אורוגוואי", "Vietnam": "וייטנאם",
}
VALUES_HE = {"sector": SECTORS_HE, "country": COUNTRIES_HE}

SIGNALS_HE: dict[str, str] = {
    "Engulfing+": "בליעה+", "Engulfing-": "בליעה-", "Hammer+": "פטיש+", "Inv Hammer+": "פטיש הפוך+",
    "Hanging Man-": "איש תלוי-", "Shooting Star-": "כוכב נופל-", "Morning Star+": "כוכב בוקר+",
    "Evening Star-": "כוכב ערב-", "3 Soldiers+": "3 חיילים+", "3 Crows-": "3 עורבים-",
    "Harami+": "הראמי+", "Harami-": "הראמי-", "Kicking+": "בעיטה+", "Kicking-": "בעיטה-",
    "Aband. Baby+": "תינוק נטוש+", "Aband. Baby-": "תינוק נטוש-", "Marubozu+": "מרובוזו+",
    "Marubozu-": "מרובוזו-", "Doji": "דוג'י", "Dragonfly+": "שפירית+", "Gravestone-": "מצבה-",
    "BO 52W": "פריצה שנתית", "BO 6M": "פריצה חצי שנה", "BO 3M": "פריצה 3 חודשים", "BO 1M": "פריצה חודשית",
    "BD 52W": "שבירה שנתית", "BD 6M": "שבירה חצי שנה", "BD 3M": "שבירה 3 חודשים", "BD 1M": "שבירה חודשית",
    "Tight 1M": "התכנסות חודשית",
    "Golden Cross": "Golden Cross", "Death Cross": "Death Cross",
    "SMA20↑SMA50": "SMA20↑SMA50", "SMA20↓SMA50": "SMA20↓SMA50",
    "Px↑SMA50": "מחיר↑SMA50", "Px↓SMA50": "מחיר↓SMA50",
    "Px↑SMA200": "מחיר↑SMA200", "Px↓SMA200": "מחיר↓SMA200",
}

# ── filter options ─────────────────────────────────────────────────────────
CANDLES_HE: dict[str, str] = {
    "Bullish Engulfing": "בליעה עולה", "Bearish Engulfing": "בליעה יורדת", "Hammer": "פטיש",
    "Inverted Hammer": "פטיש הפוך", "Hanging Man": "איש תלוי", "Shooting Star": "כוכב נופל",
    "Morning Star": "כוכב בוקר", "Evening Star": "כוכב ערב", "Three White Soldiers": "שלושה חיילים לבנים",
    "Three Black Crows": "שלושה עורבים שחורים", "Bullish Harami": "הראמי עולה", "Bearish Harami": "הראמי יורד",
    "Bullish Kicking": "בעיטה עולה", "Bearish Kicking": "בעיטה יורדת",
    "Bullish Abandoned Baby": "תינוק נטוש עולה", "Bearish Abandoned Baby": "תינוק נטוש יורד",
    "White Marubozu": "מרובוזו לבן", "Black Marubozu": "מרובוזו שחור", "Doji": "דוג'י",
    "Dragonfly Doji": "דוג'י שפירית", "Gravestone Doji": "דוג'י מצבה",
}

_MONTHS = {"1-month": "חודשי", "3-month": "3 חודשים", "6-month": "חצי שנה", "52-week": "52 שבועות"}

OPTIONS_HE: dict[str, str] = {
    "Any": "הכל", "USA": "ארה\"ב", "Foreign (ex-USA)": "זרות (לא ארה\"ב)",
    # index
    "Not in S&P 500": "לא ב-S&P 500", "Any major index": "כל מדד מרכזי",
    # market cap
    "Mega ($200B+)": "ענק ($200B+)", "Large ($10B to $200B)": "גדולה ($10B עד $200B)",
    "Mid ($2B to $10B)": "בינונית ($2B עד $10B)", "Small ($300M to $2B)": "קטנה ($300M עד $2B)",
    "Micro ($50M to $300M)": "מיקרו ($50M עד $300M)", "Nano (under $50M)": "ננו (מתחת ל-$50M)",
    "+Large (over $10B)": "גדולה ומעלה (מעל $10B)", "+Mid (over $2B)": "בינונית ומעלה (מעל $2B)",
    "+Small (over $300M)": "קטנה ומעלה (מעל $300M)", "+Micro (over $50M)": "מיקרו ומעלה (מעל $50M)",
    "-Large (under $200B)": "גדולה ומטה (מתחת ל-$200B)", "-Mid (under $10B)": "בינונית ומטה (מתחת ל-$10B)",
    "-Small (under $2B)": "קטנה ומטה (מתחת ל-$2B)",
    # qualitative
    "None (0%)": "אין (0%)", "Positive (>0%)": "חיובי (>0%)", "Negative (<0%)": "שלילי (<0%)",
    "Very High (>10%)": "גבוה מאוד (>10%)", "Very positive (>30%)": "חיובי מאוד (>30%)",
    "Positive, low (0-10%)": "חיובי, נמוך (0-10%)", "Profitable (>0)": "רווחי (>0)",
    "Negative equity": "הון עצמי שלילי", "Up": "עלייה", "Down": "ירידה",
    # earnings / analysts / target
    "Today": "היום", "Tomorrow": "מחר", "Next 7 days": "ב-7 הימים הקרובים", "Next 14 days": "ב-14 הימים הקרובים",
    "Next 30 days": "ב-30 הימים הקרובים", "Not in next 7 days": "לא ב-7 הימים הקרובים",
    "Not in next 14 days": "לא ב-14 הימים הקרובים", "Strong Buy (1)": "קנייה חזקה (1)",
    "Buy or better": "קנייה ומעלה", "Hold or better": "החזק ומעלה", "Hold": "החזק", "Sell or worse": "מכירה ומטה",
    "Above price": "מעל המחיר", "Below price": "מתחת למחיר",
    # RSI
    "Above 50 (uptrend)": "מעל 50 (מגמת עלייה)", "Below 50 (downtrend)": "מתחת ל-50 (מגמת ירידה)",
    # 52 weeks
    "New High (at high)": "בשיא", "New Low (at low)": "בשפל",
    # signals
    "Any bullish pattern": "כל תבנית עולה", "Any bearish pattern": "כל תבנית יורדת",
    "Volatility contraction (1M range < 3x ATR)": "התכווצות תנודתיות (טווח חודשי קטן מ-3 ATR)",
    "Tight base near 1-month high (<15%, within 3%)": "בסיס צר ליד השיא החודשי (<15%, עד 3% מהשיא)",
    "Up on 2x volume": "עלייה במחזור כפול", "Down on 2x volume": "ירידה במחזור כפול",
    "Golden Cross today (SMA50 crossed above SMA200)": "Golden Cross היום (SMA50 חצה מעל SMA200)",
    "Death Cross today (SMA50 crossed below SMA200)": "Death Cross היום (SMA50 חצה מתחת ל-SMA200)",
    "Price > SMA50 > SMA150 > SMA200 (Stage 2)": "מחיר מעל SMA50 מעל SMA150 מעל SMA200 (שלב 2)",
    "Price > SMA20 > SMA50 > SMA200": "מחיר מעל SMA20 מעל SMA50 מעל SMA200",
    "Price < SMA50 < SMA200": "מחיר מתחת ל-SMA50 מתחת ל-SMA200",
    "Price < SMA20 < SMA50 < SMA200": "מחיר מתחת ל-SMA20 מתחת ל-SMA50 מתחת ל-SMA200",
    "Below industry median": "מתחת לחציון התעשייה",
    "20%+ below industry": "זולה ב-20%+ מהתעשייה",
    "50%+ below industry": "זולה ב-50%+ מהתעשייה",
    "Above industry median": "מעל חציון התעשייה",
    "Above sector median": "מעל חציון הסקטור",
    "10+ pts above sector": "10+ נקודות אחוז מעל הסקטור",
    "Below sector median": "מתחת לחציון הסקטור",
    "None": "אין",
    **{en: f"{he} ({en})" for en, he in CANDLES_HE.items()},
}


def _under(v: str) -> str:
    return f"מתחת ל {v}" if v.startswith("-") else f"מתחת ל-{v}"


# (regex on the English label, Hebrew template) — tried in order after OPTIONS_HE
_RULES: list[tuple[re.Pattern, object]] = [(re.compile(p), r) for p, r in [
    (r"^Over (\S+)$", lambda m: f"מעל {m[1]}"),
    (r"^Under (\S+)$", lambda m: _under(m[1])),
    (r"^(\S+) to (\S+)$", lambda m: f"{m[1]} עד {m[2]}"),
    (r"^Up (\d+%\+)$", lambda m: f"עלייה {m[1]}"),
    (r"^Down (\d+%\+)$", lambda m: f"ירידה {m[1]}"),
    (r"^Low \((.+)\)$", lambda m: f"נמוך ({m[1]})"),
    (r"^High \((.+)\)$", lambda m: f"גבוה ({m[1]})"),
    (r"^Overbought \((\d+)\)$", lambda m: f"קניית יתר ({m[1]})"),
    (r"^Oversold \((\d+)\)$", lambda m: f"מכירת יתר ({m[1]})"),
    (r"^Not Overbought \((.+)\)$", lambda m: f"לא בקניית יתר ({m[1]})"),
    (r"^Not Oversold \((.+)\)$", lambda m: f"לא במכירת יתר ({m[1]})"),
    (r"^Price above (SMA\d+)( \(weekly\))?$", lambda m: f"מחיר מעל {m[1]}" + (" (שבועי)" if m[2] else "")),
    (r"^Price below (SMA\d+)( \(weekly\))?$", lambda m: f"מחיר מתחת ל-{m[1]}" + (" (שבועי)" if m[2] else "")),
    (r"^Price (\d+)% above (SMA\d+)( \(weekly\))?$",
     lambda m: f"מחיר {m[1]}% מעל {m[2]}" + (" (שבועי)" if m[3] else "")),
    (r"^Price (\d+)% below (SMA\d+)( \(weekly\))?$",
     lambda m: f"מחיר {m[1]}% מתחת ל-{m[2]}" + (" (שבועי)" if m[3] else "")),
    (r"^Price within (\d+)% of (SMA\d+)( \(weekly\))?$",
     lambda m: f"מחיר בטווח {m[1]}% מ-{m[2]}" + (" (שבועי)" if m[3] else "")),
    (r"^(SMA\d+) rising$", lambda m: f"{m[1]} בעלייה"),
    (r"^(SMA\d+) above (SMA\d+)$", lambda m: f"{m[1]} מעל {m[2]}"),
    (r"^(SMA\d+) below (SMA\d+)$", lambda m: f"{m[1]} מתחת ל-{m[2]}"),
    (r"^(SMA\d+) crossed above (SMA\d+) today$", lambda m: f"{m[1]} חצה מעל {m[2]} היום"),
    (r"^(SMA\d+) crossed below (SMA\d+) today$", lambda m: f"{m[1]} חצה מתחת ל-{m[2]} היום"),
    (r"^Price crossed above (SMA\d+) today$", lambda m: f"המחיר חצה מעל {m[1]} היום"),
    (r"^Price crossed below (SMA\d+) today$", lambda m: f"המחיר חצה מתחת ל-{m[1]} היום"),
    (r"^(\d+-\d+%) below High$", lambda m: f"{m[1]} מתחת לשיא"),
    (r"^(\d+%) or more below High$", lambda m: f"{m[1]} או יותר מתחת לשיא"),
    (r"^(\d+-\d+%) above Low$", lambda m: f"{m[1]} מעל השפל"),
    (r"^(\d+%) or more above Low$", lambda m: f"{m[1]} או יותר מעל השפל"),
    (r"^(\d+%) above price$", lambda m: f"{m[1]} מעל המחיר"),
    (r"^(\d+%) below price$", lambda m: f"{m[1]} מתחת למחיר"),
    (r"^New (1-month|3-month|6-month|52-week) high \(breakout\)$",
     lambda m: f"שיא {_MONTHS[m[1]]} חדש (פריצה)"),
    (r"^New (1-month|3-month|6-month|52-week) low \(breakdown\)$",
     lambda m: f"שפל {_MONTHS[m[1]]} חדש (שבירה)"),
    (r"^Tight 1-month range \((.+)\)$", lambda m: f"טווח חודשי צר ({m[1]})"),
    (r"^Tight 5-day range \((.+)\)$", lambda m: f"טווח 5 ימים צר ({m[1]})"),
    (r"^Volume (\d+)x average$", lambda m: f"מחזור פי {m[1]} מהממוצע"),
    (r"^(\d+) or more$", lambda m: f"{m[1]} או יותר"),
]]


# ── lookups ────────────────────────────────────────────────────────────────

def ui(text: str, lang: str) -> str:
    return UI.get(text, text) if lang == HE else text


def column(key: str, english: str, lang: str) -> str:
    return COLUMNS_HE.get(key, CANDLES_HE.get(english, english)) if lang == HE else english


def group_column(key: str, english: str, lang: str) -> str:
    return GROUP_COLUMNS_HE.get(key, english) if lang == HE else english


def filter_label(key: str, english: str, lang: str) -> str:
    return FILTERS_HE.get(key, english) if lang == HE else english


def option(text: str, lang: str) -> str:
    """A filter option (or a dynamic value such as a sector) for display."""
    if lang != HE:
        return text
    if text in OPTIONS_HE:
        return OPTIONS_HE[text]
    if text in SECTORS_HE or text in COUNTRIES_HE:
        return SECTORS_HE.get(text) or COUNTRIES_HE[text]
    for pattern, render in _RULES:
        m = pattern.match(text)
        if m:
            return render(m)
    return text


def preset(name: str, lang: str) -> str:
    return PRESETS_HE.get(name, name) if lang == HE else name


def value(column_key: str, v, lang: str):
    """Data value for display (sector / country names); anything else is returned unchanged."""
    if lang != HE or not isinstance(v, str):
        return v
    return VALUES_HE.get(column_key, {}).get(v, v)


PROBLEMS_HE = {
    "no_ticker_column": "לא נמצאה עמודת סימולים. בשורה הראשונה צריכה להיות כותרת ticker או סימול",
    "bad_ticker": "סימול לא תקין: {value}",
    "unknown_ticker": "הסימול {value} לא נמצא בסקרינר",
    "bad_date": "תאריך לא מובן: {value} (למשל 14/03/2025)",
    "future_date": "תאריך עתידי: {value}",
    "bad_number": "מספר לא תקין ב{value}",
    "missing_value": "חסר {value}",
    "bad_side": "פעולה לא מוכרת: {value} (קנייה או מכירה)",
    "oversold": "ב-{value} נמכרו יותר מניות ממה שנקנו עד אותו תאריך",
}
_FIELDS_HE = {"ticker": "סימול", "date": "תאריך", "price": "מחיר", "quantity": "כמות", "fees": "עמלות"}


def problem(p) -> str:
    """A portfolio.Problem in Hebrew: 'שורה 4: חסר מחיר'."""
    v = p.value
    if p.code == "missing_value":
        v = _FIELDS_HE.get(v, v)
    elif p.code == "bad_number":
        col, _, raw = v.partition(": ")
        v = f"{_FIELDS_HE.get(col, col)}: {raw}"
    text = PROBLEMS_HE.get(p.code, p.code).format(value=v)
    return f"שורה {p.row}: {text}" if p.row else text


def rtl_cell(text):
    """Wrap Hebrew text for the LTR results grid (RLE ... PDF), so a cell that mixes Hebrew and Latin tags
    ("בליעה+, SMA20↑SMA50") keeps the tag order and the +/- next to the right word."""
    return f"\u202b{text}\u202c" if isinstance(text, str) and text else text


def signal_tags(text: str, lang: str) -> str:
    """Translate signals tag list 'Engulfing+, Vol 2.8x' -> 'בליעה+, מחזור ×2.8'."""
    if lang != HE or not text:
        return text
    parts = []
    for tag in text.split(", "):
        if tag in SIGNALS_HE:
            parts.append(SIGNALS_HE[tag])
        elif tag.startswith("Vol ") and tag.endswith("x"):
            parts.append(f"מחזור ×{tag[4:-1]}")
        else:
            parts.append(tag)
    return ", ".join(parts)


REGIME_REASONS_HE = {
    "missing market data": "חסרים נתוני שוק",
    "<18 calm": "<18 רגוע",
    ">25 fear": ">25 פחד",
    "18-25 elevated": "18-25 מתוח",
    "credit risk-on": "אשראי risk-on",
    "credit risk-off": "אשראי risk-off",
}


def regime_reason(text: str, lang: str) -> str:
    """Translate a market regime reason string for display in tooltips."""
    if lang != HE or not text:
        return text
    if text in REGIME_REASONS_HE:
        return REGIME_REASONS_HE[text]
    m_vix = re.match(r"^VIX ([\d\.]+) \((<18 calm|>25 fear|18-25 elevated)\)$", text)
    if m_vix:
        v, part = m_vix.group(1), REGIME_REASONS_HE.get(m_vix.group(2), m_vix.group(2))
        return f"VIX {v} ({part})"
    m_spy = re.match(r"^SPY (above|below) SMA200$", text)
    if m_spy:
        return "SPY מעל SMA200" if m_spy.group(1) == "above" else "SPY מתחת ל-SMA200"
    m_hyg = re.match(r"^HYG (above|below) SMA20 \((credit risk-on|credit risk-off)\)$", text)
    if m_hyg:
        credit = REGIME_REASONS_HE.get(m_hyg.group(2), m_hyg.group(2))
        where = "מעל SMA20" if m_hyg.group(1) == "above" else "מתחת ל-SMA20"
        return f"HYG {where} ({credit})"
    return text
