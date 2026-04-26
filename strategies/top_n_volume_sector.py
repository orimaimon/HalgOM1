import pandas as pd
from typing import List
import logging
from core_engine import BaseEquityStrategy, Order, Portfolio

logger = logging.getLogger("TopNVolumeSector")

class TopNVolumeSectorStrategy(BaseEquityStrategy):
    def __init__(self, top_n: int = 10, hold_days: int = 20, sizing_method: str = "equal",
                 min_price: float = 5.0, max_per_sector: int = 2, use_regime_filter: bool = False):
        self.top_n = top_n
        self.hold_days = hold_days
        self.sizing_method = sizing_method
        self.min_price = min_price
        self.max_per_sector = max_per_sector
        self.use_regime_filter = use_regime_filter

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
            
            dv = float(row['Dollar_Volume_20d_Avg']) if pd.notna(row.get('Dollar_Volume_20d_Avg')) else None

            if self.use_regime_filter and not is_bull:
                sells.append(Order(ticker, current_date, current_price, pos["shares"], "SELL", "Regime Exit",
                                   avg_dollar_volume=dv))
                continue

            if hold_days >= self.hold_days:
                sells.append(Order(ticker, current_date, current_price, pos["shares"], "SELL", "Time Rebalance",
                                   avg_dollar_volume=dv))
        
        return sells

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        buys = []
        if len(portfolio.positions) >= self.top_n:
            return buys

        is_bull = self._is_bull_regime(day_data) if self.use_regime_filter else True
        if self.use_regime_filter and not is_bull:
            return buys

        stocks = self._get_stocks_only(day_data)
        valid = stocks[(stocks['Adj_Close'] >= self.min_price) & (stocks['Dollar_Volume_20d_Avg'] > 0)]
        valid = valid[~valid['Ticker'].isin(portfolio.positions.keys())]
        
        if valid.empty:
            return buys

        sorted_candidates = valid.sort_values('Dollar_Volume_20d_Avg', ascending=False)
        ticker_index = self._build_ticker_index(day_data)
        
        # ספירת סקטורים נוכחיים בתיק
        sector_counts = {}
        for ticker in portfolio.positions.keys():
            row = ticker_index.get(ticker)
            if row is not None and pd.notna(row.get('Sector')):
                sec = row['Sector']
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

        virtual_cash = portfolio.cash
        
        for _, row in sorted_candidates.iterrows():
            if len(portfolio.positions) + len(buys) >= self.top_n:
                break
                
            ticker = row['Ticker']
            price = row['Adj_Close']
            sector = row.get('Sector', 'Unknown')
            
            # דילוג אם הגענו למקסימום סקטוריאלי
            if pd.notna(sector) and sector_counts.get(sector, 0) >= self.max_per_sector:
                continue
            
            shares = self._calculate_position_size(virtual_cash, self.top_n, len(portfolio.positions) + len(buys), price)
            
            if shares > 0:
                cost = (price * shares * (1 + portfolio.slippage_pct)) + portfolio.commission
                if virtual_cash >= cost:
                    dv = float(row['Dollar_Volume_20d_Avg']) if pd.notna(row.get('Dollar_Volume_20d_Avg')) else None
                    buys.append(Order(ticker, current_date, price, shares, "BUY", "Top N Sector",
                                     avg_dollar_volume=dv))
                    virtual_cash -= cost
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    
        return buys