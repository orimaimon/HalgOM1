import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("CoreEngine")

# =============================================================================
# 1. Data Model: Order
# =============================================================================
@dataclass
class Order:
    ticker: str
    date: str
    price: float
    shares: float
    order_type: str  # "BUY" or "SELL"
    reason: str = ""

# =============================================================================
# 2. Tax Ledger
# =============================================================================
class TaxLedger:
    def __init__(self, tax_rate: float = 0.25):
        self.tax_rate = tax_rate
        self.loss_carryforward = 0.0
        self.total_tax_paid = 0.0
        self.history: List[Dict[str, Any]] = []

    def process_realized_pnl(self, date: str, gross_pnl: float, ticker: str) -> float:
        tax_paid = 0.0
        loss_offset = 0.0
        
        if gross_pnl < 0:
            self.loss_carryforward += abs(gross_pnl)
            net_pnl = gross_pnl
            status = "הפסד (נצבר לקיזוז)"
        else:
            if self.loss_carryforward > 0:
                if gross_pnl >= self.loss_carryforward:
                    loss_offset = self.loss_carryforward
                    taxable_amount = gross_pnl - self.loss_carryforward
                    self.loss_carryforward = 0.0
                    status = "רווח (קיזוז חלקי)"
                else:
                    loss_offset = gross_pnl
                    taxable_amount = 0.0
                    self.loss_carryforward -= gross_pnl
                    status = "רווח (קופז במלואו)"
            else:
                taxable_amount = gross_pnl
                status = "רווח חייב במס"
            
            tax_paid = taxable_amount * self.tax_rate
            self.total_tax_paid += tax_paid
            net_pnl = gross_pnl - tax_paid

        self.history.append({
            "Date": date,
            "Ticker": ticker,
            "Gross_PnL": round(gross_pnl, 2),
            "Loss_Offset": round(loss_offset, 2),
            "Taxable_Amount": round(taxable_amount if gross_pnl > 0 else 0, 2),
            "Tax_Paid": round(tax_paid, 2),
            "Net_PnL": round(net_pnl, 2),
            "Loss_Carryforward": round(self.loss_carryforward, 2),
            "Status": status
        })
        return net_pnl

# =============================================================================
# 3. Portfolio
# =============================================================================
class Portfolio:
    def __init__(self, initial_capital: float, tax_ledger: TaxLedger, commission_per_trade: float = 1.0, slippage_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.tax_ledger = tax_ledger
        self.trade_history: List[Dict[str, Any]] = []
        
        self.commission = commission_per_trade 
        self.slippage_pct = slippage_pct

    def execute_buy(self, order: Order):
        actual_price = order.price * (1 + self.slippage_pct)
        cost = (actual_price * order.shares) + self.commission
        
        if self.cash >= cost:
            self.cash -= cost
            self.positions[order.ticker] = {
                "shares": order.shares,
                "buy_price": actual_price,
                "buy_date": order.date,
                "reason": order.reason,
                "buy_commission": self.commission # שומרים את עמלת הקנייה
            }
        else:
            logger.debug(f"Rejected BUY {order.ticker}: Need {cost:.2f}$, Have {self.cash:.2f}$")

    def execute_sell(self, order: Order):
        if order.ticker in self.positions:
            pos = self.positions.pop(order.ticker)
            
            actual_price = order.price * (1 - self.slippage_pct)
            proceeds = (order.shares * actual_price) - self.commission
            self.cash += proceeds
            
            # חישוב רווח/הפסד מתוקן (כולל עמלת קנייה)
            buy_cost = (pos["shares"] * pos["buy_price"]) + pos.get("buy_commission", self.commission)
            gross_pnl = proceeds - buy_cost
            net_pnl = self.tax_ledger.process_realized_pnl(order.date, gross_pnl, order.ticker)
            
            hold_days = (pd.to_datetime(order.date) - pd.to_datetime(pos["buy_date"])).days
            
            self.trade_history.append({
                "Ticker": order.ticker,
                "Buy_Date": pos["buy_date"],
                "Sell_Date": order.date,
                "Hold_Days": hold_days,
                "Buy_Price": round(pos["buy_price"], 2),
                "Sell_Price": round(actual_price, 2),
                "Shares": round(pos["shares"], 4),
                "Gross_PnL": round(gross_pnl, 2),
                "Net_PnL": round(net_pnl, 2),
                "Buy_Reason": pos["reason"],
                "Sell_Reason": order.reason
            })

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        pos_value = sum(pos["shares"] * current_prices.get(ticker, pos["buy_price"]) 
                        for ticker, pos in self.positions.items())
        return self.cash + pos_value

# =============================================================================
# 4. Base Equity Strategy (Refactored)
# =============================================================================
class BaseStrategy(ABC):
    @abstractmethod
    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        pass

    @abstractmethod
    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        pass

class BaseEquityStrategy(BaseStrategy):
    """
    מחלקת אב מורחבת הכוללת פונקציות עזר משותפות לכלל אסטרטגיות האקוויטי
    (סינון משטר שוק, הקצאת מזומן וירטואלי, ואינדוקס נתונים מהיר).
    """
    def _build_ticker_index(self, day_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """מייצר מילון שליפות מהיר (O(1)) במקום חיפושי שורות כבדים (O(N)) בתוך הדאטהפריים"""
        return {row['Ticker']: row for _, row in day_data.iterrows()}

    def _is_bull_regime(self, day_data: pd.DataFrame) -> bool:
        """בודק האם השוק במצב חיובי על פי נתוני תעודת הסל של המדד (SPY)"""
        spy_row = day_data[day_data['Ticker'] == 'SPY']
        if not spy_row.empty:
            spy_close = spy_row.iloc[0]['Adj_Close']
            spy_sma = spy_row.iloc[0]['SPY_SMA_200']
            if pd.notna(spy_close) and pd.notna(spy_sma):
                return spy_close > spy_sma
        return True # Default to True if SPY data is missing

    def _get_stocks_only(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """מסנן החוצה אינדקסים ותעודות סל"""
        if 'Type' in day_data.columns:
            return day_data[day_data['Type'] == 'Stock']
        return day_data

    def _calculate_position_size(self, virtual_cash: float, target_positions: int, current_positions: int, price: float) -> float:
        """מחשב כמות מניות לרכישה מתוך המזומן הפנוי, מגן מפני דחיות פקודה בשל עמלות והחלקה"""
        slots_available = target_positions - current_positions
        if slots_available <= 0 or virtual_cash <= 0:
            return 0.0
            
        cash_allocated = (virtual_cash * 0.98) / slots_available # 2% באפר לעמלות והחלקה
        shares = cash_allocated / price
        return shares

# =============================================================================
# 5. The Backtest Engine
# =============================================================================
class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, 
                 initial_capital: float = 100000.0, 
                 commission: float = 1.0, 
                 slippage: float = 0.001):
        
        self.data = data
        self.strategy = strategy
        self.tax_ledger = TaxLedger()
        self.portfolio = Portfolio(initial_capital, self.tax_ledger, commission, slippage)
        self.equity_curve: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"🚀 Starting Engine with Strategy: {self.strategy.__class__.__name__}")
        
        # O(N) grouping instead of O(N^2) masking - Massive Speedup
        grouped_data = self.data.groupby('Date')
        
        for current_date, day_data in grouped_data:
            current_prices = dict(zip(day_data['Ticker'], day_data['Adj_Close']))
            
            # Step 1: Sells
            sells = self.strategy.generate_sells(current_date, day_data, self.portfolio)
            for sell in sells:
                self.portfolio.execute_sell(sell)
                
            # Step 2: Buys
            buys = self.strategy.generate_buys(current_date, day_data, self.portfolio)
            for buy in buys:
                self.portfolio.execute_buy(buy)
                
            # Step 3: Snapshot
            equity = self.portfolio.get_equity(current_prices)
            self.equity_curve.append({
                "Date": current_date,
                "Capital": round(equity, 2),
                "Cash": round(self.portfolio.cash, 2),
                "Open_Positions": len(self.portfolio.positions)
            })
            
        logger.info("✅ Backtest Completed.")
        
        return {
            "Equity_Curve": pd.DataFrame(self.equity_curve),
            "Trades": pd.DataFrame(self.portfolio.trade_history),
            "Taxes": pd.DataFrame(self.tax_ledger.history)
        }