import pandas as pd
from typing import List
import logging
from core_engine import Order, Portfolio
from strategies.top_n_volume_regime import TopNVolumeRegimeStrategy

logger = logging.getLogger("TopNVolumeTrendStrategy")


class TopNVolumeTrendStrategy(TopNVolumeRegimeStrategy):
    """
    TopN Volume + SL + Regime + Trend Gate

    הוספה יחידה מעל TopNVolumeRegimeStrategy:
      לפני שמועמדת נכנסת לרשימת ה-Top N, היא חייבת להציג
      Return_252d_Pct >= min_yearly_return.

    היפותזה:
      ב-2008 (וב-crashes דומים) הסיגנל "Dollar Volume גבוה" התהפך —
      המניות הכי עתירות-נפח היו אלה שזרקו עליהן בפאניקה. ל-Return_252d_Pct
      שלהן ערך עמוק שלילי בזמן הקנייה. הסינון הזה אמור לחסום אותן בלי
      לפגוע בקניות בתנאי שוק רגילים (שם רוב המועמדים הם ממילא בטרנד חיובי).

    פרמטר חדש:
      min_yearly_return: float (default 0.0)
        סף תשואה שנתית מינימלית במחיר (כשבר, לא %). למשל:
          0.0   = רק מניות שעלו בשנה האחרונה
         -0.10  = מותר עד 10% ירידה (סינון רך יותר)
          0.10  = רק מניות שעלו 10%+ (סינון קשוח)
        מניות עם Return_252d_Pct = NaN (פחות מ-252 ימי מסחר היסטוריה) ייפלו
        מהסינון אוטומטית — וזה מכוון: אין מספיק נתונים להעריך טרנד.
    """

    def __init__(self,
                 top_n: int = 10,
                 hold_days: int = 90,
                 measure_column: str = "Dollar_Volume_20d_Avg",
                 sizing_method: str = "equal",
                 min_price: float = 1.0,
                 stop_loss_pct: float = 0.15,
                 use_regime_filter: bool = True,
                 min_yearly_return: float = 0.0):

        super().__init__(
                top_n=top_n,
                hold_days=hold_days,
                # <--- השורה measure_column=measure_column נמחקה מכאן
                sizing_method=sizing_method,
                min_price=min_price,
                stop_loss_pct=stop_loss_pct,
                use_regime_filter=use_regime_filter,
                )
        # min_yearly_return מוגדר כשבר עשרוני (0.10 = 10%), אבל הנתונים
        # ב-Return_252d_Pct מאוחסנים כאחוזים (10.0 = 10%) — נמיר בעת הסינון.
        self.min_yearly_return = float(min_yearly_return)

    def generate_buys(self, current_date: str, day_data: pd.DataFrame,
                      portfolio: Portfolio) -> List[Order]:
        # אם הנתון לא קיים בכלל ב-DB — נופלים ל-baseline (אזהרה חד-פעמית).
        if 'Return_252d_Pct' not in day_data.columns:
            if not getattr(self, '_warned_missing_return_col', False):
                logger.warning(
                    "Return_252d_Pct לא קיים ב-day_data — נופלים להתנהגות "
                    "TopNVolumeRegimeStrategy ללא Trend Gate."
                )
                self._warned_missing_return_col = True
            return super().generate_buys(current_date, day_data, portfolio)

        # סף השוואה באחוזים (כי כך מאוחסן בעמודה).
        threshold_pct = self.min_yearly_return * 100.0

        # זיהוי שורות "stock" כדי לא לסנן SPY/VIX וכו' (שאר הקוד הבסיסי
        # זקוק להן עבור regime check). שיטה זהה לזו של _stocks_only של ההורה.
        if 'Type' in day_data.columns:
            is_stock = (day_data['Type'] == 'stock')
        else:
            is_stock = ~day_data['Ticker'].str.startswith('^', na=False)

        # מניות שעוברות את שער הטרנד. NaN >= threshold יחזיר False — תקין.
        trend_ok = day_data['Return_252d_Pct'] >= threshold_pct

        # שמירה על: כל מה שאינו "stock" (מדדים/מאקרו) + מניות שעברו את הסף.
        filtered_data = day_data[(~is_stock) | trend_ok]

        n_passed = int((is_stock & trend_ok).sum())
        n_total = int(is_stock.sum())
        if n_total > 0 and n_passed == 0:
            # יום שבו אף מניה לא עברה את שער הטרנד —
            # מקרה אופייני לשפל של בר-מרקט עמוק.
            if getattr(self, '_last_zero_log_date', None) != current_date:
                logger.info(
                    f"{current_date}: 0/{n_total} מניות עברו את "
                    f"min_yearly_return={self.min_yearly_return:.2%} → אין קנייה"
                )
                self._last_zero_log_date = current_date

        return super().generate_buys(current_date, filtered_data, portfolio)
