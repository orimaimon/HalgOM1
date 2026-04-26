import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from core_engine import BaseStrategy, Portfolio, Order

logger = logging.getLogger("MomentumStrategy")


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy V3.4 — The Filtered Protector

    שינויים מ-v3.3:
      + סינון Type == 'stock' (ללא אינדקסים/ETFs)
      + max_momentum_pct — זריקת "parabolic blowoffs" (מניות שעלו 300%+ ב-120d)
      + use_regime_filter — בזמן bear market (SPY < SMA200): exit all & stay cash
      + days_since_rebalance מונה רק כשבפוזיציה (היה באג — מנה גם בימים לא רלוונטיים)

    לוגיקה עקרונית:
      1. סינון יקום: Type='stock', min_price, min_dv, momentum lookback קיים
      2. Regime check (אופציונלי): אם bear → sell all, don't re-enter
      3. Rebalance כל N ימים: sell all → buy top-N momentum
      4. ATR Trailing Stop על כל פוזיציה בודדת

    פרמטרים חשובים:
      top_n             — כמה פוזיציות להחזיק במקביל
      momentum_col      — שדה המיון ('Return_60d_Pct' / '120d' / '252d')
      rebalance_days    — תדירות רה-באלאנס (20=חודשי, 60=רבעוני)
      max_momentum_pct  — תקרת מומנטום (None=ללא). 300 = נזרק אם עלה 300%+
      use_regime_filter — True = יציאה ל-cash ב-bear markets
      atr_stop_mult     — רוחב trailing stop (3 = שיא מינוס 3 ATRs)
    """
    def __init__(self,
                 top_n: int = 10,
                 momentum_col: str = "Return_120d_Pct",
                 rebalance_days: int = 20,
                 min_price: float = 5.0,
                 min_dollar_volume: float = 10_000_000,
                 atr_stop_mult: float = 3.0,
                 max_momentum_pct: float = 300.0,
                 use_regime_filter: bool = True):

        self.top_n = top_n
        self.momentum_col = momentum_col
        self.rebalance_days = rebalance_days
        self.min_price = min_price
        self.min_dv = min_dollar_volume
        self.atr_stop_mult = atr_stop_mult
        self.max_momentum_pct = max_momentum_pct
        self.use_regime_filter = use_regime_filter

        self.days_since_rebalance = 0
        self.position_peaks: Dict[str, float] = {}
        self._regime_block_logged_date = None

    def _is_bull_regime(self, day_data: pd.DataFrame) -> bool:
        """
        בודק SPY vs SMA200. החזרה True = אפשר להיות long.
        אם הנתונים לא קיימים — default True (אל תחסום בהיעדר מידע).
        """
        if 'SPY_Close' not in day_data.columns or 'SPY_SMA_200' not in day_data.columns:
            return True
        # SPY_Close ו-SPY_SMA_200 מקודקדים לכל השורות באותו יום — קח את הראשון שאינו NaN
        row = day_data[['SPY_Close', 'SPY_SMA_200']].dropna()
        if row.empty:
            return True
        spy = row.iloc[0]['SPY_Close']
        sma = row.iloc[0]['SPY_SMA_200']
        return bool(spy > sma)

    def _stocks_only(self, day_data: pd.DataFrame) -> pd.DataFrame:
        if 'Type' in day_data.columns:
            return day_data[day_data['Type'] == 'stock']
        return day_data[~day_data['Ticker'].str.startswith('^', na=False)]

    def generate_sells(self, current_date, day_data, portfolio) -> List[Order]:
        orders = []

        # מונה rebalance רץ רק כשבפוזיציה
        if portfolio.positions:
            self.days_since_rebalance += 1

        # ── 1. Regime filter — יציאה כוללת ב-bear market ──────────────────────
        regime_exit = False
        if self.use_regime_filter and portfolio.positions and not self._is_bull_regime(day_data):
            regime_exit = True
            for ticker, pos in list(portfolio.positions.items()):
                ticker_row = day_data[day_data['Ticker'] == ticker]
                price = ticker_row.iloc[0]['Adj_Close'] if not ticker_row.empty else pos['buy_price']
                orders.append(Order(ticker, current_date, price, pos['shares'],
                                    "SELL", "Regime Exit (SPY < SMA200)"))
            self.position_peaks = {}
            self.days_since_rebalance = 0
            # Log once per bear-market entry
            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, exiting all positions")
                self._regime_block_logged_date = current_date
            return orders

        # ── 2. ATR Trailing Stop לכל פוזיציה ──────────────────────────────────
        for ticker, pos in list(portfolio.positions.items()):
            ticker_row = day_data[day_data['Ticker'] == ticker]
            if ticker_row.empty:
                continue

            curr_price = ticker_row.iloc[0]['Adj_Close']
            curr_atr = ticker_row.iloc[0].get('ATR_14', 0) or 0
            if pd.isna(curr_atr):
                curr_atr = 0

            # עדכון שיא מחיר
            if ticker not in self.position_peaks:
                self.position_peaks[ticker] = pos['buy_price']
            self.position_peaks[ticker] = max(self.position_peaks[ticker], curr_price)

            # חישוב stop
            stop_price = self.position_peaks[ticker] - (curr_atr * self.atr_stop_mult)
            if curr_price <= stop_price and curr_atr > 0:
                orders.append(Order(ticker, current_date, curr_price, pos['shares'],
                                    "SELL", f"ATR Stop (ATR:{curr_atr:.2f})"))
                self.position_peaks.pop(ticker, None)

        # ── 3. Rebalance תקופתי ───────────────────────────────────────────────
        if self.days_since_rebalance >= self.rebalance_days and portfolio.positions:
            for ticker, pos in portfolio.positions.items():
                if not any(o.ticker == ticker for o in orders):
                    ticker_row = day_data[day_data['Ticker'] == ticker]
                    price = ticker_row.iloc[0]['Adj_Close'] if not ticker_row.empty else pos['buy_price']
                    orders.append(Order(ticker, current_date, price, pos['shares'],
                                        "SELL", "Rebalance"))
            self.days_since_rebalance = 0
            self.position_peaks = {}

        return orders

    def generate_buys(self, current_date, day_data, portfolio) -> List[Order]:
        orders = []
        slots_available = self.top_n - len(portfolio.positions)
        if slots_available <= 0:
            return orders

        # ── Regime check לפני קנייה ────────────────────────────────────────────
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return orders

        # ── חלוקת הון לפי equity כולל ─────────────────────────────────────────
        current_prices = dict(zip(day_data['Ticker'], day_data['Adj_Close']))
        total_equity = portfolio.get_equity(current_prices)
        target_pos_size = (total_equity * 0.95) / self.top_n

        # ── סינון יקום ────────────────────────────────────────────────────────
        candidates = self._stocks_only(day_data).copy()
        candidates = candidates[
            (candidates['Adj_Close'] >= self.min_price) &
            (candidates['Dollar_Volume_20d_Avg'] >= self.min_dv) &
            (candidates[self.momentum_col].notna())
        ]

        # תקרת מומנטום: זריקת parabolic blowoffs
        if self.max_momentum_pct is not None:
            candidates = candidates[candidates[self.momentum_col] <= self.max_momentum_pct]

        # חייב מומנטום חיובי (אין טעם לקנות מה"top 10" אם כולם אדומים)
        candidates = candidates[candidates[self.momentum_col] > 0]

        if candidates.empty:
            return orders

        # מיון → top N
        top_momentum = candidates.sort_values(by=self.momentum_col, ascending=False).head(slots_available)

        for _, row in top_momentum.iterrows():
            ticker = row['Ticker']
            if ticker in portfolio.positions:
                continue
            price = row['Adj_Close']
            shares = int(target_pos_size // price)
            cost = shares * price
            if shares > 0 and cost <= portfolio.cash:
                mom_val = row[self.momentum_col]
                orders.append(Order(ticker, current_date, price, shares,
                                    "BUY", f"Momentum {mom_val:+.0f}%"))

        return orders
