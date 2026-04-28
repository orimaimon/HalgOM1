import pandas as pd
from typing import List
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("TopNVolumeStrategy")

class TopNVolumeStrategy(BaseEquityStrategy):
    """
    אסטרטגיית מחזור דולרי (Top N) בסיסית - גרסה אופטימלית.
    קונה סל מניות ומחזיקה אותן לפרק זמן קצוב.
    """
    def __init__(self, top_n: int = 10, hold_days: int = 90,
                 measure_column: str = "Dollar_Volume_20d_Avg",
                 sizing_method: str = "equal",
                 min_price: float = 1.0,
                 use_regime_filter: bool = False):
        self.top_n = top_n
        self.hold_days = hold_days
        self.measure_column = measure_column
        self.sizing_method = sizing_method
        self.min_price = min_price
        self.use_regime_filter = use_regime_filter

        self.days_in_trade = 0
        self.in_position = False
        self._regime_block_logged_date = None

    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if not self.in_position or not portfolio.positions:
            return []

        # 1. Regime filter
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, liquidating to cash")
                self._regime_block_logged_date = current_date
            self.in_position = False
            self.days_in_trade = 0
            return self._generate_regime_exit_orders(current_date, day_data, portfolio)

        self.days_in_trade += 1

        # 2. Time Exit
        if self.days_in_trade >= self.hold_days:
            orders = []
            ticker_index = self._build_ticker_index(day_data)
            for ticker, pos in list(portfolio.positions.items()):
                row = ticker_index.get(ticker)
                if row is not None:
                    current_price = row['Adj_Close']
                    dv = float(row[self.measure_column]) if pd.notna(row.get(self.measure_column)) else None
                    orders.append(Order(
                        ticker=ticker, date=current_date, price=current_price,
                        shares=pos["shares"], order_type="SELL",
                        reason=f"Time Exit ({self.hold_days} days)",
                        avg_dollar_volume=dv,
                    ))
                else:
                    logger.warning(f"Ticker {ticker} missing data on exit day {current_date}")
            
            self.in_position = False
            self.days_in_trade = 0
            return orders

        return []

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        # קונים רק כשאנחנו מחוץ לפוזיציה
        if self.in_position or len(portfolio.positions) > 0:
            return []

        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return []

        valid_data = self._get_stocks_only(day_data)
        valid_data = valid_data.dropna(subset=[self.measure_column, 'Adj_Close']).copy()
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]

        if valid_data.empty:
            return []

        top_stocks = valid_data.nlargest(self.top_n, self.measure_column)
        total_dv = top_stocks[self.measure_column].sum()

        orders = []
        virtual_cash = portfolio.cash

        for _, row in top_stocks.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            dv = row[self.measure_column]

            if self.sizing_method == "relative_dv" and total_dv > 0:
                weight = dv / total_dv
                cash_allocated = (virtual_cash * 0.98) * weight
                shares = int(cash_allocated // price)
            else:
                shares = int(self._calculate_position_size(virtual_cash, self.top_n, len(orders), price))

            if shares > 0:
                cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission
                if cost <= virtual_cash:
                    orders.append(Order(
                        ticker=ticker, date=current_date, price=price,
                        shares=shares, order_type="BUY",
                        reason=f"TopN Entry",
                        avg_dollar_volume=float(dv),
                    ))
                    virtual_cash -= cost

        if orders:
            self.in_position = True

        return orders
