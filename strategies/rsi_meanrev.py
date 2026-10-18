import pandas as pd
from typing import List
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("RSIMeanReversionStrategy")

class RSIMeanReversionStrategy(BaseEquityStrategy):
    """
    אסטרטגיית Mean Reversion מבוססת RSI — קונה מניות שירדו לאזור oversold
    בתוך מגמת עלייה ארוכת-טווח, ויוצאת כשהן חוזרות לאזור ניטרלי.
    """

    def __init__(self,
                 top_n: int = 8,
                 rsi_buy_threshold: float = 30.0,
                 rsi_sell_threshold: float = 60.0,
                 max_holding_days: int = 30,
                 stop_loss_pct: float = 0.10,
                 min_price: float = 5.0,
                 min_dollar_volume: float = 10_000_000,
                 require_uptrend: bool = True,
                 use_regime_filter: bool = False):

        self.top_n = top_n
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.max_holding_days = max_holding_days
        self.stop_loss_pct = stop_loss_pct
        self.min_price = min_price
        self.min_dv = min_dollar_volume
        self.require_uptrend = require_uptrend
        self.use_regime_filter = use_regime_filter
        self._regime_block_logged_date = None

    def generate_sells(self, current_date: str, day_data: pd.DataFrame,
                       portfolio: Portfolio) -> List[Order]:
        if not portfolio.positions:
            return []

        # 1. Regime exit — שימוש ב-Base Helper
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
            current_rsi = row.get('RSI_14', None)
            dv = float(row['Dollar_Volume_20d_Avg']) if pd.notna(row.get('Dollar_Volume_20d_Avg')) else None

            entry_price = pos['buy_price']
            shares = pos['shares']
            buy_date_dt = pd.to_datetime(pos['buy_date'])
            days_held = (curr_date_dt - buy_date_dt).days

            # בדיקה 1: Stop Loss קשיח
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Stop Loss ({self.stop_loss_pct*100:.1f}%)",
                    avg_dollar_volume=dv
                ))
                continue

            # בדיקה 2: Mean Reversion הצליח (RSI חזר לאזור ניטרלי+)
            if pd.notna(current_rsi) and current_rsi >= self.rsi_sell_threshold:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Mean Reversion (RSI={current_rsi:.1f})",
                    avg_dollar_volume=dv
                ))
                continue

            # בדיקה 3: Time Exit
            if days_held >= self.max_holding_days:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Time Exit ({self.max_holding_days} days)",
                    avg_dollar_volume=dv
                ))

        return orders

    def generate_buys(self, current_date: str, day_data: pd.DataFrame,
                      portfolio: Portfolio) -> List[Order]:
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return []

        slots_available = self.top_n - len(portfolio.positions)
        if slots_available <= 0:
            return []

        # סינון יקום בסיסי (Base Helper)
        valid_data = self._get_stocks_only(day_data)
        valid_data = valid_data.dropna(subset=['RSI_14', 'Adj_Close']).copy()

        # סינון מחיר ונזילות
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]
        if 'Dollar_Volume_20d_Avg' in valid_data.columns:
            valid_data = valid_data[valid_data['Dollar_Volume_20d_Avg'] >= self.min_dv]

        # סינון Dow primary trend (long-term uptrend)
        if self.require_uptrend and 'Return_252d_Pct' in valid_data.columns:
            valid_data = valid_data[valid_data['Return_252d_Pct'] > 0]

        # סינון oversold
        valid_data = valid_data[valid_data['RSI_14'] < self.rsi_buy_threshold]
        valid_data = valid_data[~valid_data['Ticker'].isin(portfolio.positions.keys())]

        if valid_data.empty:
            return []

        # מיון: הכי oversold ראשון (RSI הנמוך ביותר)
        candidates = valid_data.nsmallest(slots_available, 'RSI_14')
        
        orders = []
        virtual_cash = portfolio.cash

        for _, row in candidates.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            rsi = row['RSI_14']
            dv = float(row['Dollar_Volume_20d_Avg']) if pd.notna(row.get('Dollar_Volume_20d_Avg')) else None

            # שימוש ב-Base Helper להקצאה שווה
            shares = int(self._calculate_position_size(virtual_cash, self.top_n, len(portfolio.positions) + len(orders), price))
            
            if shares > 0:
                cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission
                if cost <= virtual_cash:
                    orders.append(Order(
                        ticker=ticker, date=current_date, price=price,
                        shares=shares, order_type="BUY",
                        reason=f"RSI Oversold Entry (RSI={rsi:.1f})",
                        avg_dollar_volume=dv
                    ))
                    virtual_cash -= cost

        return orders
