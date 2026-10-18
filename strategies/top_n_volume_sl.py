import pandas as pd
from typing import List
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("TopNVolumeSLStrategy")

class TopNVolumeSLStrategy(BaseEquityStrategy):
    """
    אסטרטגיית מחזור דולרי (Top N) עם מנגנון Stop Loss קשיח וסינון משטר שוק.
    אופטימיזציה לביצועים באמצעות אינדקס טיקרים.
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

    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if not portfolio.positions:
            return []

        # 1. בדיקת משטר שוק - אם דובי ומופעל סינון, מוכרים הכל (שימוש ב-Base Helper)
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, liquidating to cash")
                self._regime_block_logged_date = current_date
            return self._generate_regime_exit_orders(current_date, day_data, portfolio)

        orders = []
        ticker_index = self._build_ticker_index(day_data)
        curr_date_dt = pd.to_datetime(current_date)

        for ticker, pos in list(portfolio.positions.items()):
            row = ticker_index.get(ticker)
            if row is None:
                continue
                
            current_price = row['Adj_Close']
            dv = float(row[self.measure_column]) if pd.notna(row.get(self.measure_column)) else None

            entry_price = pos['buy_price']
            shares = pos['shares']
            buy_date_dt = pd.to_datetime(pos['buy_date'])
            days_held = (curr_date_dt - buy_date_dt).days

            # בדיקה 1: Stop Loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Stop Loss ({self.stop_loss_pct*100:.1f}%)",
                    avg_dollar_volume=dv,
                ))
                continue

            # בדיקה 2: Time Exit
            if days_held >= self.hold_days:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Time Exit ({self.hold_days} days)",
                    avg_dollar_volume=dv,
                ))

        return orders

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return []
            
        slots_available = self.top_n - len(portfolio.positions)
        if slots_available <= 0:
            return []

        valid_data = self._get_stocks_only(day_data)
        valid_data = valid_data.dropna(subset=[self.measure_column, 'Adj_Close']).copy()
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]
        valid_data = valid_data[~valid_data['Ticker'].isin(portfolio.positions.keys())]

        if valid_data.empty:
            return []

        top_stocks = valid_data.nlargest(slots_available, self.measure_column)
        total_dv = top_stocks[self.measure_column].sum()
        
        orders = []
        virtual_cash = portfolio.cash

        for _, row in top_stocks.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            dv = row[self.measure_column]

            if self.sizing_method == "relative_dv" and total_dv > 0:
                weight = dv / total_dv
                # כאן לא משתמשים ב-helper כי ההקצאה היא יחסית ולא שווה
                cash_allocated = (virtual_cash * 0.98) * weight
                shares = int(cash_allocated // price)
            else:
                # שימוש ב-Base Helper להקצאה שווה (equal weight)
                shares = int(self._calculate_position_size(virtual_cash, self.top_n, len(portfolio.positions) + len(orders), price))

            if shares > 0:
                cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission
                if cost <= virtual_cash:
                    orders.append(Order(
                        ticker=ticker, date=current_date, price=price,
                        shares=shares, order_type="BUY",
                        reason="TopN SL Entry",
                        avg_dollar_volume=float(dv),
                    ))
                    virtual_cash -= cost

        return orders
