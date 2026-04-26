"""
top_n_volume_sector.py — TopN Volume + Sector Diversification (Block)

מטרה:
    האסטרטגיה הבסיסית TopN (block) קונה את ה-N המניות עם DV הגבוה ביותר.
    הבעיה: בתקופות מסוימות כל ה-N המניות יכולות להיות מאותו סקטור (טכנולוגיה,
    ביוטק) ואז התיק חשוף לקריסה מערכתית של אותו סקטור.

    כאן מוגבלת הבחירה: לא יותר מ-`max_per_sector` מניות מאותו סקטור.

מבנה:
    Block — כל ה-N המניות נקנות יחד, מוחזקות hold_days, נמכרות יחד.
    מקבילה ישירה ל-TopNVolumeStrategy עבור השוואת ניסויים.

לוגיקה:
    1. דירוג כל המניות הזמינות לפי DV (גבוה ראשון), top_n*6 מועמדים
    2. סיבוב ראשון: בוחר מניה רק אם הסקטור עוד לא הגיע ל-max_per_sector
    3. אם נשארו slots — relaxed fill מהמועמדים שדולגו (שמירה על תיק מלא)

הפרמטר החדש:
    max_per_sector: int — ברירת מחדל 2

יורש מ-TopNVolumeStrategy. מבטל רק את generate_buys; כל השאר (sells לפי
hold_days, regime filter, _stocks_only, _is_bull_regime) זהה לבסיס.
"""
from __future__ import annotations
import logging
from collections import Counter
from typing import List

import pandas as pd

from core_engine import Order, Portfolio
from strategies.top_n_volume import TopNVolumeStrategy

logger = logging.getLogger("TopNVolumeSectorStrategy")


class TopNVolumeSectorStrategy(TopNVolumeStrategy):
    """
    TopN Volume עם הגבלת מניות לסקטור (מבנה Block).

    Args:
        top_n: מספר מניות בתיק (10)
        hold_days: ימי החזקה לפני time-exit (90)
        measure_column: עמודה למיון מועמדים (Dollar_Volume_20d_Avg)
        sizing_method: 'equal' או 'relative_dv'
        min_price: מחיר מינימלי למניה (1.0)
        max_per_sector: מקסימום מניות מאותו סקטור (2)
        use_regime_filter: סינון לפי SPY > SMA200 (False)
        candidate_pool_mult: כמה מועמדים לבחון (top_n * mult); ברירת מחדל 6
        relaxed_fill: אם True ולא נמצאו מספיק סקטורים שונים, ממלאים את שאר
                       ה-slots מהמועמדים הטובים שדולגו (ברירת מחדל True).
                       אם False — תיק עלול לצאת חלקי בתקופות של ריכוז סקטורי.
    """

    def __init__(self,
                 top_n: int = 10,
                 hold_days: int = 90,
                 measure_column: str = "Dollar_Volume_20d_Avg",
                 sizing_method: str = "equal",
                 min_price: float = 1.0,
                 max_per_sector: int = 2,
                 use_regime_filter: bool = False,
                 candidate_pool_mult: int = 6,
                 relaxed_fill: bool = True):

        super().__init__(
            top_n=top_n,
            hold_days=hold_days,
            measure_column=measure_column,
            sizing_method=sizing_method,
            min_price=min_price,
            use_regime_filter=use_regime_filter,
        )

        self.max_per_sector = max_per_sector
        self.candidate_pool_mult = candidate_pool_mult
        self.relaxed_fill = relaxed_fill

    # ────────────────────────────────────────────────────────────────────────
    # Core selection logic
    # ────────────────────────────────────────────────────────────────────────

    def _select_with_sector_cap(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        בוחר עד top_n מניות מתוך candidates (כבר ממוין לפי DV יורד),
        תוך כיבוד הגבלה של max_per_sector מניות מאותו סקטור.

        אם relaxed_fill=True ונשארו slots פנויים אחרי הסיבוב הראשון,
        הם מתמלאים מהמועמדים שנדחו (לפי DV יורד).

        Returns:
            DataFrame — תת-קבוצה של candidates עד top_n שורות.
        """
        slots_available = self.top_n

        if "Sector" not in candidates.columns:
            # ללא נתוני סקטור — fall back להתנהגות הרגילה (top_n הראשונים)
            return candidates.head(slots_available)

        sector_count: Counter = Counter()
        selected_idx: List = []
        skipped_idx: List = []

        for idx, row in candidates.iterrows():
            if len(selected_idx) >= slots_available:
                break

            sector = row.get("Sector") or "Unknown"
            # שמירה על "Unknown" כסקטור משלו (אחרת מניות בלי סקטור תופסות הכל)
            if sector_count[sector] < self.max_per_sector:
                selected_idx.append(idx)
                sector_count[sector] += 1
            else:
                skipped_idx.append(idx)

        # Relaxed fill — מילוי slots שלא התמלאו
        if self.relaxed_fill and len(selected_idx) < slots_available:
            remaining = slots_available - len(selected_idx)
            selected_idx.extend(skipped_idx[:remaining])

        return candidates.loc[selected_idx]

    # ────────────────────────────────────────────────────────────────────────
    # generate_buys override (Block — קנייה של כל הסל יחד)
    # ────────────────────────────────────────────────────────────────────────

    def generate_buys(self, current_date, day_data, portfolio: Portfolio) -> List[Order]:
        orders: List[Order] = []

        # תנאי הבסיסי: רק כשמחוץ לפוזיציה (block trade)
        if len(portfolio.positions) > 0 or self.in_position:
            return orders

        # סינון רגי'ם (אם פעיל)
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return orders

        # סינון מניות ולידיות
        valid_data = self._stocks_only(day_data)
        valid_data = valid_data.dropna(subset=[self.measure_column, "Adj_Close"]).copy()
        valid_data = valid_data[valid_data["Adj_Close"] >= self.min_price]

        if valid_data.empty:
            return orders

        # מועמדים — top_n * candidate_pool_mult הראשונים לפי DV
        candidates = valid_data.sort_values(by=self.measure_column, ascending=False)
        candidates = candidates.head(self.top_n * self.candidate_pool_mult)

        # בחירה מודעת-סקטור
        selected = self._select_with_sector_cap(candidates)
        if selected.empty:
            return orders

        # חישוב גודלי פוזיציות (אותה לוגיקה כמו ה-parent)
        safe_cash = portfolio.cash * 0.99
        total_dv = selected[self.measure_column].sum()

        # מעקב סקטורים בריצה — שימושי ל-debugging
        sector_summary = selected.get("Sector", pd.Series(dtype=str)).value_counts().to_dict() \
            if "Sector" in selected.columns else {}

        for _, row in selected.iterrows():
            ticker = row["Ticker"]
            price = row["Adj_Close"]
            dv = row[self.measure_column]

            if self.sizing_method == "relative_dv" and total_dv > 0:
                weight = dv / total_dv
                cash_allocated = safe_cash * weight
            else:
                cash_allocated = safe_cash / len(selected)

            shares = int(cash_allocated // price)

            if shares > 0:
                sector_tag = row.get("Sector", "?") if "Sector" in row.index else "?"
                orders.append(Order(
                    ticker=ticker,
                    date=current_date,
                    price=price,
                    shares=shares,
                    order_type="BUY",
                    reason=f"Top{self.top_n} Sector-Cap{self.max_per_sector} [{sector_tag}]"
                ))

        if orders:
            self.in_position = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{current_date}: Block entry, sectors={sector_summary}")

        return orders
