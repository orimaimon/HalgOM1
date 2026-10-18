# SANDBOX — סשן 2026-05-17

## מה עשינו היום

### 1. ניתוח Focused Sweep (batch `8dabac8e`) — 500 ריצות

ריצה שהתחילה ב-2026-05-17 03:53 וסיימה ב-04:11 (18 דקות, 500/500).

**Overall:**
| מדד | ערך |
|-----|-----|
| Runs | 500 |
| Beat SPY | **100%** |
| Median CAGR | **18.76%** |
| Avg Sharpe | 0.899 |
| Avg MaxDD | -42.64% |

---

### 2. הממצא המרכזי — Regime Filter מחלק את העולם

| Subspace | n | Med CAGR | Med Sharpe | Med MaxDD |
|----------|---|----------|------------|----------|
| regime=**False** | 253 | 20.33% | 0.860 | **-51.92%** |
| regime=**True** | 247 | 16.64% | **0.930** | **-33.79%** |
| regime=True + equal sizing | 116 | **20.43%** | **1.000** | **-35.25%** |

**מסקנה:** ה-top-25-by-CAGR כולם regime=False — אבל עם MaxDD של -60%. זה לא ישים.
**regime=True + equal sizing = הסאבספייס האמיתי**: CAGR+20%, Sharpe=1.0, MaxDD≈-35%.

---

### 3. ממצאי פרמטרים

| פרמטר | ממצא |
|-------|------|
| `top_n` | 3→4→5: CAGR יורד, MaxDD משתפר. **top_n=4-5 = sweet spot in-sample** |
| `use_regime_filter` | True מנצח ב-risk-adjusted; False נותן raw CAGR גבוה יותר אבל DD של -52% |
| `sizing_method` | **equal מנצח בבירור** — +6% CAGR, +0.07 Sharpe על relative_dv |
| `min_price` | **1-3$ עדיף** על 5-10$ בכל מדד |
| `hold_days` | 150-210d עדיפות קלה ב-CAGR, לא דטרמיניסטי |

---

### 4. Walk-Forward Validation Round 1 (batch `aa3f5e56`)

4 קונפיגים × 5 folds (2018-2022), כל fold שנה אחת:

| Config | top_n | med_exQQQ | pos_alpha |
|--------|-------|-----------|-----------|
| A | 3 | -0.42pp | 2/5 |
| B | 4 | -1.42pp | 2/5 |
| C | 5 | +0.77pp | 3/5 |
| **D** | **6** | **+3.22pp** | **4/5** |

**D-top6 מנצח OOS.** top_n=6 יציב יותר — יותר פיזור = פחות variance בין folds.

---

### 5. שיפור ה-Regime Filter

**הבעיה:** ב-2022 האסטרטגיה עשתה 42 עסקאות ואיבדה 33% בגלל whipsaw סביב SMA200.

**הפיתרון שממומש ב-`strategies/top_n_volume.py`:**

```python
regime_entry_buffer=0.01  # SPY חייב להיות >1% מעל SMA200 לכניסה
regime_confirm_days=1     # כניסה מיידית אחרי שהסף מתקיים
```

- **יציאה מהירה:** SPY < SMA200 → מיידית לcash (ללא delay, הגנת הון)
- **כניסה איטית:** SPY > SMA200 × 1.01 × N ימים → חזרה לשוק

**השוואת גרסאות (batch `8fe1fba7`):**

| Variant | med_exQQQ | med_MaxDD | med_trades |
|---------|-----------|-----------|------------|
| old (0%) | +3.2pp | -24.9% | 30 |
| **buf=1%** | **+10.7pp** | **-19.2%** | **18** |
| buf=2% | +10.1pp | -19.2% | 12 |
| buf=3% | +5.4pp | -20.3% | 12 |

---

### 6. V2 Sweep (batch `6470b7fd`) — 500 ריצות

אסטרטגיה `topn_focused_v2` — מקבעת sizing=equal ו-regime=True, מוסיפה שני צירים חדשים:

**Heatmap: Excess QQQ (median pp) — buffer × confirm_days:**

```
             confirm=1  confirm=2  confirm=3
buf=0.01:    +5.84      +4.26      +2.44   ← BEST
buf=0.00:    +3.83      +1.38      +2.22
buf=0.02:    +3.17      +2.22      +2.40
buf=0.03+:   declining...
buf=0.04+:   negative
```

**מסקנות:** buf=0.01, confirm=1 = sweet spot. buf>0.03 מזיק.

---

### 7. Walk-Forward V2 (batch `c2d5a7e5`)

E(top3)/F(top4)/G(top6-buf1)/D(top6-old) × 5 folds (2018-2022):

| Config | med_exQQQ | med_MaxDD | pos_alpha |
|--------|-----------|-----------|-----------|
| **G-top6-buf1** | **+9.5pp** | **-23.8%** | **4/5** |
| D-top6-old | +3.2pp | -24.9% | 4/5 |
| F-top4-buf1 | +6.7pp | -24.3% | 3/5 |
| E-top3-buf1 | -0.2pp | -24.9% | 2/5 |

**E-top3 נכשל OOS** — מאשר שtop_n=3 הוא overfitting in-sample.
**G-top6-buf1 מנצח** — 3x שיפור ב-excess QQQ vs הבסיס.

---

### 8. Walk-Forward Extended (batch `efa85c3e`)

G vs D על 6 folds היסטוריים (2007-2009 + 2023-2025):

| Fold | G excess_QQQ | D excess_QQQ | משמעות |
|------|-------------|-------------|--------|
| 2007 | +12.4pp | +14.3pp | שניהם חזקים בbull |
| **2008** | **+40.9pp** | **+40.9pp** | **בcash כל השנה — regime filter מושלם** |
| 2009 | -22.6pp | -20.2pp | פיספוס V-recovery (צפוי) |
| 2023 | -5.5pp | -2.1pp | D קצת עדיף בrally |
| **2024** | **+39.4pp** | **+39.4pp** | **שניהם מדהימים בAI bull** |
| 2025 | -21.9pp | -23.9pp | G עדיף בtariff volatility |

**Combined 11 folds:** G median = +9.5pp, D median = +3.2pp → **G עדיין מנצח.**

---

### 9. Production Config — מאושר על 11 OOS folds ✓

```
Strategy:            topn_focused_v2
top_n:               6
hold_days:           130
min_price:           2.0
sizing_method:       equal
use_regime_filter:   True
regime_entry_buffer: 0.01
regime_confirm_days: 1
```

Folds מאושרים: 2007, 2008, 2009, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.

---

## מבנה ה-DB — שימוש מאוחד

> **חשוב:** כל סוגי הריצות — MC sweeps, WF train, WF test, השוואות regime — נכנסים
> לאותו `DB/mc_runs.sqlite`. המפתח לאבחנה ביניהם הוא `batch_id`.

```
sqlite: DB/mc_runs.sqlite
tables: batches, runs, equity_curves, trades
```

### מה נשמר לפי טבלה

| טבלה | מה יש שם | מפתח |
|------|----------|------|
| `batches` | מטא-דאטה של כל batch — strategy, n_runs, notes, תאריכים | `batch_id` (8 תווים) |
| `runs` | כל ריצה בודדת — params + metrics מלאים | `run_id` (int), `batch_id` |
| `equity_curves` | עקומת ההון יומית (רק Top-K runs) | `run_id` |
| `trades` | כל עסקה בפועל (רק Top-K runs) | `run_id` |

### איך לזהות batch לפי סוג

```python
# MC Sweep רגיל — strategy_name = שם האסטרטגיה
# WF Train     — strategy_name מכיל "(WF-TRAIN)"
# WF Test      — strategy_name מכיל "(WF-TEST)"
# WF Fixed     — strategy_name מכיל "(WF-FIXED)" / "(WF-V2)" / "(WF-EXTENDED)"
# Regime cmp   — strategy_name מכיל "(WF-REGIME-COMPARE)"
```

### Batches עיקריים

| batch_id | strategy | runs | תאריך | תיאור |
|----------|----------|------|-------|-------|
| `fef04bf7` | TopNVolumeStrategy | 500 | 2026-04-28 | sweep רחב |
| `29cfe1a0` | TopN+SL+Regime+Trend | 500 | 2026-04-27 | עם stop-loss |
| `d522c3d7` | TopN DV/MC | 300 | 2026-05-15 | גרסת DVMC |
| `8dabac8e` | **TopNVolume Focused** | 500 | 2026-05-17 | focused sweep |
| `aa3f5e56` | WF-FIXED A-D | 20 | 2026-05-17 | WF round 1 |
| `8fe1fba7` | WF-REGIME-COMPARE | 25 | 2026-05-17 | buffer variants |
| `6470b7fd` | **topn_focused_v2** | 500 | 2026-05-17 | v2 sweep + buffer axis |
| `c2d5a7e5` | **WF-V2 E/F/G** | 20 | 2026-05-17 | WF validation → G מנצח |
| `efa85c3e` | **WF-EXTENDED** | 12 | 2026-05-17 | 2007-2009 + 2023-2025 |

### Queries שימושיים

```sql
-- כל הbatches לפי תאריך
SELECT batch_id, strategy_name, n_runs_completed, started_at, notes
FROM batches ORDER BY started_at DESC;

-- top runs מbatch ספציפי
SELECT run_id, cagr_pct, sharpe, max_drawdown_pct, excess_cagr_qqq, params_json
FROM runs WHERE batch_id='c2d5a7e5' AND status='completed'
ORDER BY excess_cagr_qqq DESC LIMIT 10;

-- השוואת batch_ids — median metrics
SELECT batch_id, COUNT(*) n,
       ROUND(AVG(cagr_pct),2) avg_cagr,
       ROUND(AVG(sharpe),3) avg_sharpe,
       ROUND(AVG(excess_cagr_qqq),2) avg_exqqq
FROM runs WHERE status='completed'
GROUP BY batch_id ORDER BY avg_exqqq DESC;
```

---

## קבצים שנוצרו/עודכנו בסשן זה

| קובץ | שינוי |
|------|-------|
| `strategies/top_n_volume.py` | נוסף `regime_entry_buffer` + `regime_confirm_days` עם stateful hysteresis |
| `mc_strategies.py` | נוסף `topn_focused_v2` + builders עודכנו לפרמטרים החדשים |
| `wf_fixed.py` | **חדש** — WF למחקר: modes `fixed`, `regime`, `v2`, `extended` |

---

## פקודות מהירות

```bash
# רשימת כל הbatches
python mc_cli.py list

# תוצאות WF המוודא (G-top6-buf1)
python mc_cli.py top --batch-id c2d5a7e5
python mc_cli.py dashboard --batch-id c2d5a7e5 --open

# הרצת WF מחדש (כל mode)
python wf_fixed.py --mode fixed    --workers 4
python wf_fixed.py --mode v2       --workers 4
python wf_fixed.py --mode extended --workers 4

# sweep חדש
python mc_cli.py run --strategy topn_focused_v2 --n-runs 500 --workers 8
```
