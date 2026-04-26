import pandas as pd
from typing import List
import logging
from core_engine import Order, Portfolio
from strategies.top_n_volume_sl import TopNVolumeSLStrategy

logger = logging.getLogger("TopNVolumeRegimeStrategy")

class TopNVolumeRegimeStrategy(TopNVolumeSLStrategy):
    """
    אסטרטגיית מחזור דולרי (Top N) עם Stop Loss וסינון משטר שוק (Regime Filter).
    
    האסטרטגיה מוכרת את כל הפוזיציות ויוצאת למזומן כאשר המדד (SPY) נמצא מתחת לממוצע נע 200 (Bear Market).
    """
    def __init__(self, 
                 top_n: int = 10, 
                 hold_days: int = 90,
                 measure_column: str = "Dollar_Volume_20d_Avg",
                 sizing_method: str = "equal",
                 min_price: float = 1.0,
                 stop_loss_pct: float = 0.15,
                 use_regime_filter: bool = True):
        
        super().__init__(top_n=top_n, 
                         hold_days=hold_days, 
                         measure_column=measure_column, 
                         sizing_method=sizing_method, 
                         min_price=min_price, 
                         stop_loss_pct=stop_loss_pct)
        
        self.use_regime_filter = use_regime_filter
        self._regime_block_logged_date = None

    def _is_bull_regime(self, day_data: pd.DataFrame) -> bool:
        """
        בודק אם השוק במגמת עלייה (SPY > SMA200).
        """
        if 'SPY_Close' not in day_data.columns or 'SPY_SMA_200' not in day_data.columns:
            return True
            
        # לקיחת הערך הראשון שאינו NaN עבור היום הנוכחי
        row = day_data[['SPY_Close', 'SPY_SMA_200']].dropna()
        if row.empty:
            return True
            
        spy = row.iloc[0]['SPY_Close']
        sma = row.iloc[0]['SPY_SMA_200']
        return bool(spy > sma)

    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        # 1. בדיקת משטר שוק - אם דובי ומופעל סינון, מוכרים הכל
        if self.use_regime_filter and portfolio.positions and not self._is_bull_regime(day_data):
            orders = []
            for ticker, pos in list(portfolio.positions.items()):
                ticker_row = day_data[day_data['Ticker'] == ticker]
                price = ticker_row.iloc[0]['Adj_Close'] if not ticker_row.empty else pos['buy_price']
                orders.append(Order(
                    ticker=ticker, 
                    date=current_date, 
                    price=price, 
                    shares=pos['shares'],
                    order_type="SELL", 
                    reason="Regime Exit (SPY < SMA200)"
                ))
            
            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, liquidating to cash")
                self._regime_block_logged_date = current_date
            return orders

        # 2. אם לא דובי, משתמשים בלוגיקה הרגילה של Stop Loss ו-Time Exit
        return super().generate_sells(current_date, day_data, portfolio)

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        # בדיקת משטר שוק - אם דובי, לא קונים חדש
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return []
            
        return super().generate_buys(current_date, day_data, portfolio)
