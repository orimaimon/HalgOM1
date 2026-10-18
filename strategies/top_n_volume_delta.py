import pandas as pd
from typing import List
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("TopNVolumeDeltaStrategy")

class TopNVolumeDeltaStrategy(BaseEquityStrategy):
    """
    אסטרטגיית מחזור דולרי (Top N) עם פילטר "זינוק במחזור" (Volume Surge).
    
    האסטרטגיה מחפשת מניות בתוך ה-Top N של המחזור הדולרי הממוצע (20 יום),
    אבל מסננת אותן לפי השינוי במחזור הנוכחי ביחס לממוצע.
    
    לוגיקה:
    1. בחירת מועמדים עם מחזור דולרי ממוצע גבוה (נזילות).
    2. חישוב 'Volume Surge' = המחזור של היום חלקי הממוצע של 20 יום.
    3. סינון מניות עם Surge > סף מינימלי (למשל 1.2 = 20% מעל הממוצע).
    4. יציאה מבוססת זמן (hold_days) או סטופ לוס.
    """
    def __init__(self, top_n: int = 10, hold_days: int = 20,
                 min_vol_surge: float = 1.0,
                 min_price: float = 1.0,
                 stop_loss_pct: float = 0.15,
                 use_regime_filter: bool = True):
        
        self.top_n = top_n
        self.hold_days = hold_days
        self.min_vol_surge = min_vol_surge
        self.min_price = min_price
        self.stop_loss_pct = stop_loss_pct
        self.use_regime_filter = use_regime_filter
        self._regime_block_logged_date = None

    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if not portfolio.positions:
            return []

        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return self._generate_regime_exit_orders(current_date, day_data, portfolio)

        orders = []
        ticker_index = self._build_ticker_index(day_data)
        curr_date_dt = pd.to_datetime(current_date)

        for ticker, pos in list(portfolio.positions.items()):
            row = ticker_index.get(ticker)
            if row is None:
                continue
                
            current_price = row['Adj_Close']
            dv_avg = float(row['Dollar_Volume_20d_Avg']) if pd.notna(row.get('Dollar_Volume_20d_Avg')) else None

            entry_price = pos['buy_price']
            buy_date_dt = pd.to_datetime(pos['buy_date'])
            days_held = (curr_date_dt - buy_date_dt).days

            # 1. Stop Loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=pos['shares'], order_type="SELL",
                    reason=f"Stop Loss ({self.stop_loss_pct*100:.1f}%)",
                    avg_dollar_volume=dv_avg,
                ))
                continue

            # 2. Time Exit
            if days_held >= self.hold_days:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=pos['shares'], order_type="SELL",
                    reason=f"Time Exit ({self.hold_days} days)",
                    avg_dollar_volume=dv_avg,
                ))

        return orders

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return []
            
        slots_available = self.top_n - len(portfolio.positions)
        if slots_available <= 0:
            return []

        valid_data = self._get_stocks_only(day_data)
        # אנחנו צריכים את המחזור היומי הנוכחי (Volume * Adj_Close) ואת הממוצע (Dollar_Volume_20d_Avg)
        valid_data = valid_data.dropna(subset=['Dollar_Volume_20d_Avg', 'Adj_Close', 'Volume']).copy()
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]
        valid_data = valid_data[~valid_data['Ticker'].isin(portfolio.positions.keys())]

        if valid_data.empty:
            return []

        # חישוב ה-Surge (השינוי במחזור)
        # Note: 'Volume' הוא כמות מניות, לכן נכפיל במחיר לקבלת מחזור דולרי יומי
        valid_data['Current_DV'] = valid_data['Volume'] * valid_data['Adj_Close']
        valid_data['Vol_Surge'] = valid_data['Current_DV'] / valid_data['Dollar_Volume_20d_Avg']

        # סינון לפי סף הזינוק שביקש המשתמש
        valid_data = valid_data[valid_data['Vol_Surge'] >= self.min_vol_surge]
        
        if valid_data.empty:
            return []

        # בחירת המניות עם המחזור הממוצע הגבוה ביותר מבין אלו שעברו את סינון הזינוק
        top_candidates = valid_data.nlargest(slots_available, 'Dollar_Volume_20d_Avg')
        
        orders = []
        virtual_cash = portfolio.cash

        for _, row in top_candidates.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            dv_avg = row['Dollar_Volume_20d_Avg']
            surge = row['Vol_Surge']

            shares = int(self._calculate_position_size(virtual_cash, self.top_n, len(portfolio.positions) + len(orders), price))

            if shares > 0:
                cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission
                if cost <= virtual_cash:
                    orders.append(Order(
                        ticker=ticker, date=current_date, price=price,
                        shares=shares, order_type="BUY",
                        reason=f"Vol Surge Entry (Surge={surge:.2f})",
                        avg_dollar_volume=float(dv_avg),
                    ))
                    virtual_cash -= cost

        return orders
