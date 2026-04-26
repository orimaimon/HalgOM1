import pandas as pd
from typing import List, Dict, Any
import logging
from core_engine import BaseEquityStrategy, Order, Portfolio

logger = logging.getLogger("MomentumStrategy")

class MomentumStrategy(BaseEquityStrategy):
    def __init__(self, top_n: int = 10, momentum_col: str = "Return_120d_Pct", 
                 rebalance_days: int = 20, min_price: float = 5.0, 
                 min_dollar_volume: float = 10_000_000.0, atr_stop_mult: float = 3.0, 
                 max_momentum_pct: float = 300.0, use_regime_filter: bool = True):
        self.top_n = top_n
        self.momentum_col = momentum_col
        self.rebalance_days = rebalance_days
        self.min_price = min_price
        self.min_dollar_volume = min_dollar_volume
        self.atr_stop_mult = atr_stop_mult
        self.max_momentum_pct = max_momentum_pct
        self.use_regime_filter = use_regime_filter
        
        self.peak_prices: Dict[str, float] = {} # למעקב אחרי Trailing Stop

    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        sells = []
        if not portfolio.positions:
            return sells

        ticker_index = self._build_ticker_index(day_data)
        is_bull = self._is_bull_regime(day_data) if self.use_regime_filter else True

        for ticker, pos in list(portfolio.positions.items()):
            row = ticker_index.get(ticker)
            if row is None:
                continue
            
            current_price = row['Adj_Close']
            hold_days = (pd.to_datetime(current_date) - pd.to_datetime(pos["buy_date"])).days
            
            # עדכון מחיר שיא ל-Trailing Stop
            if current_price > self.peak_prices.get(ticker, pos["buy_price"]):
                self.peak_prices[ticker] = current_price

            # תנאי 1: פילטר שוק (Regime) - בורחים הכל כשמתחיל Bear Market
            if self.use_regime_filter and not is_bull:
                sells.append(Order(ticker, current_date, current_price, pos["shares"], "SELL", "Regime Exit"))
                self.peak_prices.pop(ticker, None)
                continue

            # תנאי 2: ATR Trailing Stop - חיתוך הפסדים
            atr = row.get('ATR_14', 0)
            if pd.notna(atr) and atr > 0:
                stop_price = self.peak_prices[ticker] - (atr * self.atr_stop_mult)
                if current_price <= stop_price:
                    sells.append(Order(ticker, current_date, current_price, pos["shares"], "SELL", "ATR Trailing Stop"))
                    self.peak_prices.pop(ticker, None)
                    continue

            # תנאי 3: ריבלנס מבוסס זמן
            if hold_days >= self.rebalance_days:
                sells.append(Order(ticker, current_date, current_price, pos["shares"], "SELL", "Rebalance"))
                self.peak_prices.pop(ticker, None)
        
        return sells

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        buys = []
        if len(portfolio.positions) >= self.top_n:
            return buys

        is_bull = self._is_bull_regime(day_data) if self.use_regime_filter else True
        if self.use_regime_filter and not is_bull:
            return buys

        stocks = self._get_stocks_only(day_data)
        valid = stocks[
            (stocks['Adj_Close'] >= self.min_price) & 
            (stocks['Dollar_Volume_20d_Avg'] >= self.min_dollar_volume) &
            (stocks[self.momentum_col].notna()) &
            (stocks[self.momentum_col] <= self.max_momentum_pct)
        ]
        
        # סינון מניות שכבר קיימות בתיק
        valid = valid[~valid['Ticker'].isin(portfolio.positions.keys())]
        if valid.empty:
            return buys

        top_candidates = valid.nlargest(self.top_n - len(portfolio.positions), self.momentum_col)
        
        # שימוש במזומן וירטואלי כדי להבטיח שלא נחרוג בעמלות
        virtual_cash = portfolio.cash
        
        for _, row in top_candidates.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            
            shares = self._calculate_position_size(virtual_cash, self.top_n, len(portfolio.positions) + len(buys), price)
            
            if shares > 0:
                cost = (price * shares * (1 + portfolio.slippage_pct)) + portfolio.commission
                if virtual_cash >= cost:
                    buys.append(Order(ticker, current_date, price, shares, "BUY", "Momentum Entry"))
                    self.peak_prices[ticker] = price
                    virtual_cash -= cost

        return buys