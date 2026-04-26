import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# הגדרת מערכת הלוגים של המנוע
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("CoreEngine")

# =============================================================================
# 1. מודל הנתונים: פקודת מסחר (Order)
# =============================================================================
@dataclass
class Order:
    ticker: str
    date: str
    price: float
    shares: float
    order_type: str  # "BUY" או "SELL"
    reason: str = "" # סיבת הפעולה (למשל: "Stop Loss", "Top N Momentum")

# =============================================================================
# 2. מנהל המיסים (Tax Ledger)
# =============================================================================
class TaxLedger:
    """
    מנהל את חישובי המס באופן ריאליסטי:
    - גובה 25% מס על רווחי הון ריאליים.
    - שומר הפסדים ב'פנקס' (Loss Carryforward) לקיזוז מול רווחים עתידיים.
    """
    def __init__(self, tax_rate: float = 0.25):
        self.tax_rate = tax_rate
        self.loss_carryforward = 0.0
        self.total_tax_paid = 0.0
        self.history: List[Dict[str, Any]] = []

    def process_realized_pnl(self, date: str, gross_pnl: float, ticker: str) -> float:
        tax_paid = 0.0
        loss_offset = 0.0
        
        if gross_pnl < 0:
            # במקרה של הפסד, מוסיפים אותו למגן המס שלנו
            self.loss_carryforward += abs(gross_pnl)
            net_pnl = gross_pnl
            status = "הפסד (נצבר לקיזוז)"
        else:
            # במקרה של רווח, בודקים אם יש לנו הפסדים קודמים לקזז
            if self.loss_carryforward > 0:
                if gross_pnl >= self.loss_carryforward:
                    # הרווח גדול ממגן המס - מקזזים את כולו ומשלמים מס על השארית
                    loss_offset = self.loss_carryforward
                    taxable_amount = gross_pnl - self.loss_carryforward
                    self.loss_carryforward = 0.0
                    status = "רווח (קיזוז חלקי)"
                else:
                    # הרווח קטן ממגן המס - אין תשלום מס, מעדכנים את היתרה
                    loss_offset = gross_pnl
                    taxable_amount = 0.0
                    self.loss_carryforward -= gross_pnl
                    status = "רווח (קופז במלואו)"
            else:
                # אין מגן מס - משלמים מס על כל הרווח
                taxable_amount = gross_pnl
                status = "רווח חייב במס"
            
            tax_paid = taxable_amount * self.tax_rate
            self.total_tax_paid += tax_paid
            net_pnl = gross_pnl - tax_paid

        # תיעוד הפעולה לצורך דוחות ודיבוג
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
# 3. מנהל התיק (Portfolio)
# =============================================================================
class Portfolio:
    """
    מנהל את המזומן והפוזיציות של התיק.
    כולל התחשבות בעמלות ברוקר (Commissions) והחלקת מחירים (Slippage) לדימוי שוק אמיתי.
    """
    def __init__(self, initial_capital: float, tax_ledger: TaxLedger, commission_per_trade: float = 1.0, slippage_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.tax_ledger = tax_ledger
        self.trade_history: List[Dict[str, Any]] = []
        
        self.commission = commission_per_trade 
        self.slippage_pct = slippage_pct

    def execute_buy(self, order: Order):
        # החלקה: קונים קצת יותר ביוקר ממחיר הסגירה
        actual_price = order.price * (1 + self.slippage_pct)
        cost = (actual_price * order.shares) + self.commission
        
        if self.cash >= cost:
            self.cash -= cost
            self.positions[order.ticker] = {
                "shares": order.shares,
                "buy_price": actual_price,
                "buy_date": order.date,
                "reason": order.reason
            }
        else:
            logger.warning(f"Rejected BUY {order.ticker}: Need {cost:.2f}$, Have {self.cash:.2f}$")

    def execute_sell(self, order: Order):
        if order.ticker in self.positions:
            pos = self.positions.pop(order.ticker)
            
            # החלקה: מוכרים קצת יותר בזול ממחיר הסגירה
            actual_price = order.price * (1 - self.slippage_pct)
            proceeds = (order.shares * actual_price) - self.commission
            self.cash += proceeds
            
            # חישוב רווח/הפסד ומיסוי
            gross_pnl = proceeds - (pos["shares"] * pos["buy_price"])
            net_pnl = self.tax_ledger.process_realized_pnl(order.date, gross_pnl, order.ticker)
            
            # חישוב ימי החזקה
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
        """שערוך התיק הנוכחי (Mark-to-Market)"""
        pos_value = sum(pos["shares"] * current_prices.get(ticker, pos["buy_price"]) 
                        for ticker, pos in self.positions.items())
        return self.cash + pos_value

# =============================================================================
# 4. מחלקת האב לאסטרטגיות (Base Strategy)
# =============================================================================
class BaseStrategy(ABC):
    """
    ממשק מופשט שכל אסטרטגיה חייבת לממש.
    המנוע רק שואל את האסטרטגיה מה לעשות, ולא יודע איך היא מקבלת את ההחלטה.
    """
    @abstractmethod
    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        pass

    @abstractmethod
    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        pass

# =============================================================================
# 5. ליבת הסימולטור (The Backtest Engine)
# =============================================================================
class BacktestEngine:
    """
    המנוע הראשי. מריץ את הסימולציה יום אחרי יום בצורה כרונולוגית,
    מבצע קודם מכירות כדי לשחרר מזומן, ואז קניות, ומתעד את עקומת ההון.
    """
    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, 
                 initial_capital: float = 100000.0, 
                 commission: float = 1.0, 
                 slippage: float = 0.001):
        
        self.data = data
        self.strategy = strategy
        self.tax_ledger = TaxLedger()
        self.portfolio = Portfolio(initial_capital, self.tax_ledger, commission, slippage)
        self.equity_curve: List[Dict[str, Any]] = []
        
        # יצירת ציר זמן ייחודי וממוין (חשוב מאוד למניעת Look-ahead)
        self.dates = sorted(self.data['Date'].unique())

    def run(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"🚀 Starting Engine with Strategy: {self.strategy.__class__.__name__}")
        
        for current_date in self.dates:
            # חילוץ נתוני היום הנוכחי בלבד
            day_data = self.data[self.data['Date'] == current_date]
            if day_data.empty:
                continue
                
            # מחירון עדכני לשערוך התיק
            current_prices = dict(zip(day_data['Ticker'], day_data['Adj_Close']))
            
            # שלב 1: איסוף וביצוע פקודות מכירה (מפנה מזומן לקניות)
            sells = self.strategy.generate_sells(current_date, day_data, self.portfolio)
            for sell in sells:
                self.portfolio.execute_sell(sell)
                
            # שלב 2: איסוף וביצוע פקודות קנייה (עם המזומן הפנוי)
            buys = self.strategy.generate_buys(current_date, day_data, self.portfolio)
            for buy in buys:
                self.portfolio.execute_buy(buy)
                
            # שלב 3: צילום מצב סוף יום (Snapshot)
            equity = self.portfolio.get_equity(current_prices)
            self.equity_curve.append({
                "Date": current_date,
                "Capital": round(equity, 2),
                "Cash": round(self.portfolio.cash, 2),
                "Open_Positions": len(self.portfolio.positions)
            })
            
        logger.info("✅ Backtest Completed.")
        
        # החזרת התוצאות בפורמט שמוכן לייצוא לאקסל
        return {
            "Equity_Curve": pd.DataFrame(self.equity_curve),
            "Trades": pd.DataFrame(self.portfolio.trade_history),
            "Taxes": pd.DataFrame(self.tax_ledger.history)
        }