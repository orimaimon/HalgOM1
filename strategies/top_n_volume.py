import pandas as pd
from typing import List
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("TopNVolumeStrategy")


class TopNVolumeStrategy(BaseEquityStrategy):
    """
    אסטרטגיית מחזור דולרי (דור 1) — v1.1

    שינוי מ-v1.0:
      + סינון Type == 'stock' (ללא אינדקסים / ETFs)
        ב-v1.0 האסטרטגיה קנתה ^GSPC, ^NDX וכו' כי ה-Dollar_Volume
        שלהם אסטרונומי. זה יצר תשואה מטעה של +16.8% CAGR שבעיקר
        שיקפה החזקה פסיבית של S&P 500.

    תומכת בשתי שיטות חלוקת הון:
      "equal"        — שווה לכל מניה
      "relative_dv"  — יחסי לנפח הדולרי (המניה עם DV גבוה יותר → חלק גדול יותר)
    """
    def __init__(self, top_n: int = 10, hold_days: int = 90,
                 measure_column: str = "Dollar_Volume_20d_Avg",
                 sizing_method: str = "equal",   # "equal" or "relative_dv"
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

    def generate_sells(self, current_date, day_data, portfolio) -> List[Order]:
        orders = []
        if not self.in_position:
            return orders

        # ── v1.2: Regime filter — יציאה ב-bear market ──
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            for ticker, pos in portfolio.positions.items():
                ticker_data = day_data[day_data['Ticker'] == ticker]
                price = ticker_data.iloc[0]['Adj_Close'] if not ticker_data.empty else pos['buy_price']
                dv = float(ticker_data.iloc[0]['Dollar_Volume_20d_Avg']) if not ticker_data.empty else None
                orders.append(Order(
                    ticker=ticker, date=current_date, price=price,
                    shares=pos["shares"], order_type="SELL",
                    reason="Regime Exit (SPY < SMA200)",
                    avg_dollar_volume=dv,
                ))
            
            self.in_position = False
            self.days_in_trade = 0
            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, exiting positions")
                self._regime_block_logged_date = current_date
            return orders

        self.days_in_trade += 1

        # יציאה מבוססת זמן
        if self.days_in_trade >= self.hold_days:
            for ticker, pos in portfolio.positions.items():
                ticker_data = day_data[day_data['Ticker'] == ticker]
                if not ticker_data.empty:
                    current_price = ticker_data.iloc[0]['Adj_Close']
                    dv = float(ticker_data.iloc[0]['Dollar_Volume_20d_Avg'])
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

    def generate_buys(self, current_date, day_data, portfolio) -> List[Order]:
        orders = []

        # קונים רק כשאנחנו מחוץ לפוזיציה ויש מזומן פנוי
        if len(portfolio.positions) > 0 or self.in_position:
            return orders

        # ── v1.2: Regime check ──
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return orders

        # ── v1.1: סינון אינדקסים/ETFs לפני כל דבר אחר ──
        valid_data = self._get_stocks_only(day_data)
        valid_data = valid_data.dropna(subset=[self.measure_column, 'Adj_Close']).copy()
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]

        if valid_data.empty:
            return orders

        # איתור מניות ה-Top N לפי מחזור דולרי
        top_stocks = valid_data.sort_values(by=self.measure_column, ascending=False).head(self.top_n)
        if top_stocks.empty:
            return orders

        safe_cash = portfolio.cash * 0.99   # באפר לעמלות והחלקה

        total_dv = top_stocks[self.measure_column].sum()

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

            if shares > 0:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=price,
                    shares=shares, order_type="BUY",
                    reason=f"Top {self.top_n} {self.measure_column} ({self.sizing_method})",
                    avg_dollar_volume=float(dv),
                ))

        if orders:
            self.in_position = True

        return orders
