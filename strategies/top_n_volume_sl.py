import pandas as pd
from typing import List
import logging
from core_engine import BaseStrategy, Portfolio, Order

logger = logging.getLogger("TopNVolumeSLStrategy")

class TopNVolumeSLStrategy(BaseStrategy):
    """
    אסטרטגיית מחזור דולרי (Top N) עם מנגנון Stop Loss קשיח.
    """
    def __init__(self, top_n: int = 10, hold_days: int = 90,
                 measure_column: str = "Dollar_Volume_20d_Avg",
                 sizing_method: str = "equal",
                 min_price: float = 1.0,
                 stop_loss_pct: float = 0.15,
                 use_regime_filter: bool = False):
        
        self.top_n = top_n
        self.hold_days = hold_days
        self.measure_column = measure_column
        self.sizing_method = sizing_method
        self.min_price = min_price
        self.stop_loss_pct = stop_loss_pct
        self.use_regime_filter = use_regime_filter
        self._regime_block_logged_date = None

    def _stocks_only(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """סינון תעודות סל ומדדים (בדומה לגרסה 1.1)"""
        if 'Type' in day_data.columns:
            return day_data[day_data['Type'] == 'stock']
        return day_data[~day_data['Ticker'].str.startswith('^', na=False)]

    def _is_bull_regime(self, day_data: pd.DataFrame) -> bool:
        """בודק אם השוק במגמת עלייה (SPY > SMA200)."""
        if 'SPY_Close' not in day_data.columns or 'SPY_SMA_200' not in day_data.columns:
            return True
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
                    ticker=ticker, date=current_date, price=price,
                    shares=pos['shares'], order_type="SELL",
                    reason="Regime Exit (SPY < SMA200)"
                ))
            
            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, liquidating to cash")
                self._regime_block_logged_date = current_date
            return orders

        orders = []
        if not portfolio.positions:
            return orders

        # המרת התאריך הנוכחי לאובייקט datetime כדי שנוכל לחשב הפרשי ימים
        curr_date_dt = pd.to_datetime(current_date)

        for ticker, pos in list(portfolio.positions.items()):
            ticker_data = day_data[day_data['Ticker'] == ticker]
            if ticker_data.empty:
                continue
                
            current_price = ticker_data['Adj_Close'].iloc[0]
            
            # שליפה נכונה של הנתונים מהמילון שמוגדר ב-core_engine.py
            entry_price = pos['buy_price']
            shares = pos['shares']
            buy_date_dt = pd.to_datetime(pos['buy_date'])
            
            # חישוב ימי ההחזקה בפועל
            days_held = (curr_date_dt - buy_date_dt).days

            # בדיקה 1: Stop Loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Stop Loss ({self.stop_loss_pct*100:.1f}%)"
                ))
                continue

            # בדיקה 2: Time Exit
            if days_held >= self.hold_days:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Time Exit ({self.hold_days} days)"
                ))

        return orders

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        # בדיקת משטר שוק - אם דובי, לא קונים חדש
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return []
            
        orders = []
        
        valid_data = self._stocks_only(day_data)
        valid_data = valid_data.dropna(subset=[self.measure_column, 'Adj_Close']).copy()
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]

        if valid_data.empty:
            return orders

        current_positions = len(portfolio.positions)
        slots_available = self.top_n - current_positions
        
        if slots_available <= 0:
            return orders # התיק מלא

        valid_data = valid_data[~valid_data['Ticker'].isin(portfolio.positions.keys())]

        top_stocks = valid_data.sort_values(by=self.measure_column, ascending=False).head(slots_available)
        if top_stocks.empty:
            return orders

        # באפר בטוח יותר למזומן (98%) כדי להכיל את עמלות הברוקר וההחלקה
        safe_cash = portfolio.cash * 0.98
        total_dv = top_stocks[self.measure_column].sum()
        
        # מעקב וירטואלי אחרי המזומן במהלך הלולאה
        virtual_cash = portfolio.cash

        for _, row in top_stocks.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            dv = row[self.measure_column]

            if self.sizing_method == "relative_dv" and total_dv > 0:
                weight = dv / total_dv
                cash_allocated = safe_cash * weight
            else:
                cash_allocated = safe_cash / len(top_stocks)

            shares = int(cash_allocated // price)

            # חישוב עלות משוערת בהתאם ללוגיקה ב-core_engine.py (החלקה של 0.1% + 1$ עמלה)
            estimated_cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission

            if shares > 0 and estimated_cost <= virtual_cash:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=price,
                    shares=shares, order_type="BUY",
                    reason="TopN SL Entry"
                ))
                virtual_cash -= estimated_cost # קיזוז המזומן למניה הבאה

        return orders