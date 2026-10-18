# HalgOM Screener

סורק מניות מקומי בסגנון Finviz למניות ארה"ב, בעברית. מותאם למשתמש רגיל על Windows.

## מה זה
הסורק מוריד נתונים על כ-5,000 מניות ארה"ב (מניות רגילות ו-ADRs בבורסות NASDAQ, NYSE, AMEX). הנתונים מגיעים מ-TradingView ומוצגים בעיכוב של כ-15 דקות.
השימוש הוא למטרות מחקר בלבד, ואינו מהווה ייעוץ השקעות.

## התקנה והפעלה ראשונה (Windows)
כדי לקבל את התיקייה, מומלץ להריץ בשורת הפקודה (cmd) את הפקודה הבאה:
```cmd
git clone https://github.com/orimaimon/HalgOM1.git
```
היתרון בכך הוא שהסורק יתעדכן אוטומטית בכל הפעלה. אם מורידים כקובץ ZIP, עדכון אוטומטי לא יעבוד (כי המנגנון מריץ `git pull` בתוך התיקייה).

להפעלה, יש ללחוץ לחיצה כפולה על `run_screener.bat`.
בהפעלה הראשונה הוא יתקין בעצמו את `uv`, פייתון ואת החבילות הנדרשות (פעולה שלוקחת דקה או שתיים, ללא צורך בהרשאות מנהל מערכת). לאחר מכן הדפדפן ייפתח עם הסורק.
כמו כן, בהפעלה הראשונה ייווצר קיצור דרך בשולחן העבודה בשם "HalgOM Screener".

## שימוש יומיומי

### עמודים
במערכת יש שלושה עמודים מרכזיים (מופיעים בצד תחת "סורק", "קבוצות", "מפת חום"). העמוד המרכזי הוא "סורק". בעמוד זה יש לשוניות פילטרים ("פילטרים"), שורת חיפוש, אפשרויות מיון, ותצוגות טבלה שונות (כמו "סקירה", "תמחור", "טכני" ועוד).

### סריקות מובנות
המערכת כוללת סריקות מוכנות מראש (תחת "סריקה"):
* **ערך — זולות ורווחיות** - מניות זולות (P/E מתחת ל-15) של חברות רווחיות (ROE מעל 10%) עם חוב סביר
* **צמיחה איכותית** - חברות רווחיות שהרווח וההכנסות שלהן צומחים מהר
* **מובילות מומנטום** - מניות במגמת עלייה חזקה: מעל הממוצעים הנעים וקרובות לשיא השנתי
* **מכירת יתר במגמת עלייה** - ירידה זמנית (RSI נמוך) של מניות גדולות שעדיין במגמת עלייה ארוכה
* **פריצה לשיא שנתי** - מניות שנוגעות בשיא השנתי עם מחזור גבוה מהרגיל
* **דיבידנד יציב** - דיבידנד מעל 3% שהחברה יכולה להרשות לעצמה (מחלקת פחות מ-80% מהרווח)
* **צמיחה במניות קטנות** - חברות קטנות שההכנסות שלהן צומחות מעל 30% ברבעון
* **מחזור חריג** - מחזור פי 3 ויותר מהרגיל — משהו קורה במניה
* **שלב 2 (Minervini)** - שלב 2 לפי Minervini: מחיר מעל SMA50 שמעל SMA150 שמעל SMA200, לפחות 30% מעל השפל השנתי ובטווח של 25% מהשיא.
* **ציון HalgOM גבוה** - הציון המשוקלל הגבוה ביותר, בלי מניות שיוצא להן דוח השבוע

### מסננים וחיפוש
* **הסתר זבל:** מתג שמופעל כברירת מחדל כדי להסתיר מניות לא רלוונטיות (מחיר מתחת ל-$1, מחזור ממוצע מתחת ל-50K, או שווי שוק מתחת ל-$50M).
* **רק חדשות:** מאפשר לראות אילו מניות נוספו לסריקה הנוכחית מאז צילום הנתונים של יום המסחר הקודם (מופיע כ-"חדש" בעמודה ייעודית).
* לחיצה על שורה בטבלה פותחת את הגרף ופרטי המניה תחתיו.

### עמודת האותות (Signals)
הטבלה מציגה בעמודת האותות סימונים קצרים לאירועים טכניים שקרו היום.

| קוד קצר | פירוש |
|:---|:---|
| `[Pattern]+ / [Pattern]-` | תבנית נרות יפנים (למשל `Engulfing+`) |
| `Vol [X]x` | קפיצת מחזור פי X מהממוצע (מעל 2.0x) |
| `BO 52W / 6M / 3M / 1M` | פריצה: סגירה מעל השיא של התקופה (משנה ועד חודש) |
| `BD 52W / 6M / 3M / 1M` | שבירה: סגירה מתחת לשפל של התקופה (משנה ועד חודש) |
| `Tight 1M` | התכווצות תנודתיות (טווח חודשי צר ביחס ל-ATR) |
| `Golden Cross` | ממוצע 50 יום חצה היום מעל ממוצע 200 יום |
| `Death Cross` | ממוצע 50 יום חצה היום מתחת לממוצע 200 יום |
| `SMA20↑SMA50` / `SMA20↓SMA50` | חציית ממוצעים 20 ו-50 |
| `Px↑SMA50 / Px↓SMA50` | המחיר חצה היום את ממוצע 50 |
| `Px↑SMA200 / Px↓SMA200` | המחיר חצה היום את ממוצע 200 |

### שורת מצב השוק
בראש המסך מוצג מצב השוק לפי כללים:
* **שוק עולה (BULL):** VIX מתחת ל-18, SPY מעל ממוצע 200, ו-HYG מעל ממוצע 20.
* **שוק יורד (BEAR):** לפחות 2 מתוך התנאים (VIX מעל 25, SPY מתחת ל-200, HYG מתחת ל-20). זה גובר על שוק עולה.
* **דשדוש (SIDEWAYS):** כל מצב אחר.

ישנו גם תג של עוצמת המגמה (מבוסס על SPY ADX ל-14 יום): חזקה (מעל 25), מתפתחת (20-25), או חלשה (מתחת ל-20).

### שמירה ומחיקה של סריקות
ניתן לשמור סריקות עם פילטרים מותאמים אישית או למחוק סריקות שנשמרו, באמצעות הלחצנים בתפריט הסריקות.

### ייצוא
בסרגל העליון יש אפשרויות ייצוא שונות:
* אקסל מתקדם עם צבעים, CSV, רשימה ל-TradingView (לייבוא מהיר שם) או רשימת סימולים פשוטה.
* כל קובץ שאתם מייצאים גם נשמר אוטומטית בתיקייה פנימית (`screener/_data/exports`).

### שפה
ניתן להחליף את שפת הממשק בין עברית לאנגלית באמצעות כפתור השפה בסרגל. כל הפילטרים והכללים מתורגמים. השמות המקוריים נשמרים במערכת כדי שהסריקות יעבדו בכל שפה.

### סגירה
* כדי לסגור מסודר, השתמשו בכפתור "יציאה".
* ניתן גם פשוט לסגור את החלון השחור של שורת הפקודה.
* אם תסגרו את כל חלונות הדפדפן, המערכת תכבה את עצמה אוטומטית לאחר 15 דקות של חוסר פעילות.
* אם תפעילו את `run_screener.bat` בזמן שהמערכת כבר פועלת, היא פשוט תפתח שוב את הדפדפן לעמוד הקיים, ולא תפעיל כפילות.

### עדכונים
בכל הפעלה, המערכת מושכת עדכונים אוטומטית (דרך הפקודה `git pull` ולכן דרוש Git).
אם רוצים לדלג על העדכון, אפשר להגדיר משתנה סביבה `SCREENER_NO_UPDATE=1`.

### בעיות נפוצות
* **אין אינטרנט או ש-TradingView חסום:** המערכת תציג את הנתונים השמורים האחרונים עם אזהרה.
* **התראת "Something went wrong" בחלון השחור:** צלמו מסך של החלון השחור ושלחו את הקובץ `screener\_data\logs\screener.log` (אם קיים) ל-Ori.
* **הנתונים לא מעודכנים ב-100%:** ישנה השהיה קבועה של כ-15 דקות מ-TradingView.

### הבהרה
הסורק הוא למטרות מחקר בלבד ולא מהווה בשום אופן ייעוץ השקעות.

---

## למפתחים (For Developers)

### Architecture Overview

| Module | Description |
|:---|:---|
| `fields.py` | Column registry (TradingView field, label, format kind, Hebrew tooltip) + table views. |
| `filters.py` | Finviz-style dropdown filters; each option is a predicate on the whole DataFrame. |
| `data/tradingview.py` | Direct `requests` client for `scanner.tradingview.com` (unofficial, ~15 min delayed). |
| `data/store.py` | Latest-cache (15 min), one snapshot per day, offline fallback ("stale"). |
| `derive.py`, `scores.py` | Derived columns and Value/Quality/Momentum percentile scores. |
| `signals.py` | Signal definitions shared by the Signals filters and the "Signals" / "# Signals" columns. |
| `history.py` | "New since yesterday": same pipeline on the previous snapshot day. |
| `groups.py` | Groups page (sector/industry averages) and the plotly treemap heatmap. |
| `i18n.py` | Display language. Internal values stay English; only display is translated. |
| `lifecycle.py` | "Exit" button and an idle watchdog that stops the server after 15 minutes. |
| `market.py` | Regime header and SPY ADX trend strength. |
| `export.py` | Excel, CSV, TradingView watchlist exports. |
| `app.py` | Streamlit UI (pages: Screener / Groups / Heatmap). |

### Running and Testing

Setup and run tests:
```bash
uv venv --python 3.12 screener/.venv
uv pip install --python screener/.venv -r screener/requirements.lock pytest
screener/.venv/bin/python -m pytest tests/ -q
```
*(On Windows cmd, use `screener\.venv\Scripts\python`)*

Run the app locally:
```bash
screener/.venv/bin/python -m streamlit run screener/app.py
```

### Key Rules from CLAUDE.md
* **Data fetch:** Exactly ONE TradingView request per 15 minutes, never parallel.
* **Providers:** Must return canonical column keys from `fields.COLUMNS`.
* **Launcher (`run_screener.bat`):** Must stay ASCII with CRLF line endings. Bytes 529–785 are frozen for the auto-update mechanism; tests check this.
* **Translations:** Internal values stay English and only the display is translated. Tests require a Hebrew text for every filter, option, column, view and preset.
* **Roadmap:** See `CLAUDE.md` for phase 3 (strategy presets, yfinance integration).
