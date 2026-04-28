import pandas as pd
import numpy as np
from typing import List
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("FactorComboStrategy")

class FactorComboStrategy(BaseEquityStrategy):
    """
    אסטרטגיית מולטי-פקטור המשלבת מומנטום, RSI (Mean Reversion) ומחזור מסחר.
    השיטה: דירוג מניות לפי מספר פקטורים ובחירת ה-Top N עם הציון המשולב הגבוה ביותר.
    """
    def __init__(self, top_n: int = 10, hold_days: int = 30,
                 momentum_weight: float = 0.4,
                 rsi_weight: float = 0.3,
                 volume_weight: float = 0.3,
                 min_price: float = 5.0,
                 min_dollar_volume: float = 10_000_000,
                 stop_loss_pct: float = 0.15,
                 use_regime_filter: bool = True):
        
        self.top_n = top_n
        self.hold_days = hold_days
        self.momentum_weight = momentum_weight
        self.rsi_weight = rsi_weight
        self.volume_weight = volume_weight
        self.min_price = min_price
        self.min_dv = min_dollar_volume
        self.stop_loss_pct = stop_loss_pct
        self.use_regime_filter = use_regime_filter
        self._regime_block_logged_date = None

    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if not portfolio.positions:
            return []

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
            dv = float(row['Dollar_Volume_20d_Avg']) if pd.notna(row.get('Dollar_Volume_20d_Avg')) else None

            entry_price = pos['buy_price']
            shares = pos['shares']
            buy_date_dt = pd.to_datetime(pos['buy_date'])
            days_held = (curr_date_dt - buy_date_dt).days

            # 1. Stop Loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Stop Loss ({self.stop_loss_pct*100:.1f}%)",
                    avg_dollar_volume=dv,
                ))
                continue

            # 2. Time Rebalance
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
        needed_cols = ['Return_120d_Pct', 'RSI_14', 'Dollar_Volume_20d_Avg', 'Adj_Close']
        valid_data = valid_data.dropna(subset=needed_cols).copy()
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]
        valid_data = valid_data[valid_data['Dollar_Volume_20d_Avg'] >= self.min_dv]
        valid_data = valid_data[~valid_data['Ticker'].isin(portfolio.positions.keys())]

        if valid_data.empty:
            return []

        # חישוב ציונים (Z-Score או Percentile Rank)
        # מומנטום - גבוה זה טוב
        valid_data['score_mom'] = valid_data['Return_120d_Pct'].rank(pct=True)
        # RSI - נמוך זה טוב (Oversold)
        valid_data['score_rsi'] = valid_data['RSI_14'].rank(pct=True, ascending=False)
        # מחזור - גבוה זה טוב
        valid_data['score_vol'] = valid_data['Dollar_Volume_20d_Avg'].rank(pct=True)

        valid_data['total_score'] = (
            valid_data['score_mom'] * self.momentum_weight +
            valid_data['score_rsi'] * self.rsi_weight +
            valid_data['score_vol'] * self.volume_weight
        )

        top_candidates = valid_data.nlargest(slots_available, 'total_score')
        
        orders = []
        virtual_cash = portfolio.cash

        for _, row in top_candidates.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            dv = row['Dollar_Volume_20d_Avg']

            shares = int(self._calculate_position_size(virtual_cash, self.top_n, len(portfolio.positions) + len(orders), price))

            if shares > 0:
                cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission
                if cost <= virtual_cash:
                    orders.append(Order(
                        ticker=ticker, date=current_date, price=price,
                        shares=shares, order_type="BUY",
                        reason=f"Factor Entry (Score={row['total_score']:.2f})",
                        avg_dollar_volume=float(dv),
                    ))
                    virtual_cash -= cost

        return orders
