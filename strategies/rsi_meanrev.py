import pandas as pd
from typing import List
import logging
from core_engine import BaseStrategy, Portfolio, Order

logger = logging.getLogger("RSIMeanReversionStrategy")


class RSIMeanReversionStrategy(BaseStrategy):
    """
    אסטרטגיית Mean Reversion מבוססת RSI — קונה מניות שירדו לאזור oversold
    בתוך מגמת עלייה ארוכת-טווח, ויוצאת כשהן חוזרות לאזור ניטרלי.

    מקור עיוני:
      Technical Analysis — RSI overbought/oversold levels (RSI<30 / RSI>70).
      בשילוב Dow Theory primary trend filter (Return_252d > 0): קונים תיקונים
      בתוך מגמה ראשית עולה, לא "סכינים נופלים" בתוך bear market.

    לוגיקה עקרונית:
      ── כניסה ──
      1. סינון יקום: Type='stock', min_price, min_dollar_volume
      2. (אופציונלי) require_uptrend: Return_252d_Pct > 0
      3. (אופציונלי) Regime filter: SPY > SMA200
      4. מסננים RSI_14 < rsi_buy_threshold (סביב 30 = oversold)
      5. ממיינים ASC לפי RSI (הכי oversold ראשון) → top N

      ── יציאה (הראשון מבין השלושה) ──
      • RSI_14 >= rsi_sell_threshold (חזרה לאזור ניטרלי) — "Mean Reversion"
      • Stop Loss קשיח (% מתחת ל-buy_price)
      • Time Exit (max_holding_days) — מנע "stuck positions" שנשארות שנה ב-30<RSI<50

      ── Regime exit ──
      • אם use_regime_filter=True ו-SPY נופל מתחת ל-SMA200 → סוגרים הכל

    פרמטרים:
      top_n                — כמה פוזיציות במקביל
      rsi_buy_threshold    — מתחת לזה = oversold = candidate for buy (typical 25-35)
      rsi_sell_threshold   — מעל לזה = mean reversion הצליח, לוקחים רווח (typical 50-65)
      max_holding_days     — תקרת זמן לפוזיציה (אחרת היציאה רק על RSI/SL)
      stop_loss_pct        — כמה למטה מ-buy_price = יציאת חירום
      min_price            — סף מחיר מינימלי (סינון "פני אגורה")
      min_dollar_volume    — סף נזילות יומית
      require_uptrend      — True = רק מניות עם Return_252d > 0 (Dow primary trend)
      use_regime_filter    — True = יציאה ל-cash במשטר דובי
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

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _stocks_only(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """סינון תעודות סל ומדדים."""
        if 'Type' in day_data.columns:
            return day_data[day_data['Type'] == 'stock']
        return day_data[~day_data['Ticker'].str.startswith('^', na=False)]

    def _is_bull_regime(self, day_data: pd.DataFrame) -> bool:
        """בודק אם השוק במגמת עלייה (SPY > SMA200). default True בהיעדר נתונים."""
        if 'SPY_Close' not in day_data.columns or 'SPY_SMA_200' not in day_data.columns:
            return True
        row = day_data[['SPY_Close', 'SPY_SMA_200']].dropna()
        if row.empty:
            return True
        spy = row.iloc[0]['SPY_Close']
        sma = row.iloc[0]['SPY_SMA_200']
        return bool(spy > sma)

    # ─────────────────────────────────────────────────────────────────────
    # SELL logic
    # ─────────────────────────────────────────────────────────────────────
    def generate_sells(self, current_date: str, day_data: pd.DataFrame,
                       portfolio: Portfolio) -> List[Order]:
        # 1. Regime exit — אם דובי ומופעל הסינון, מוכרים הכל
        if self.use_regime_filter and portfolio.positions and not self._is_bull_regime(day_data):
            orders = []
            for ticker, pos in list(portfolio.positions.items()):
                ticker_row = day_data[day_data['Ticker'] == ticker]
                price = ticker_row.iloc[0]['Adj_Close'] if not ticker_row.empty else pos['buy_price']
                orders.append(Order(
                    ticker=ticker, date=current_date, price=price,
                    shares=pos['shares'], order_type="SELL",
                    reason="Regime Exit (SPY < SMA200)"
                ))

            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, liquidating to cash")
                self._regime_block_logged_date = current_date
            return orders

        orders = []
        if not portfolio.positions:
            return orders

        curr_date_dt = pd.to_datetime(current_date)

        for ticker, pos in list(portfolio.positions.items()):
            ticker_data = day_data[day_data['Ticker'] == ticker]
            if ticker_data.empty:
                continue

            row = ticker_data.iloc[0]
            current_price = row['Adj_Close']
            current_rsi = row.get('RSI_14', None)

            entry_price = pos['buy_price']
            shares = pos['shares']
            buy_date_dt = pd.to_datetime(pos['buy_date'])
            days_held = (curr_date_dt - buy_date_dt).days

            # בדיקה 1: Stop Loss קשיח
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Stop Loss ({self.stop_loss_pct*100:.1f}%)"
                ))
                continue

            # בדיקה 2: Mean Reversion הצליח (RSI חזר לאזור ניטרלי+)
            if pd.notna(current_rsi) and current_rsi >= self.rsi_sell_threshold:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Mean Reversion (RSI={current_rsi:.1f})"
                ))
                continue

            # בדיקה 3: Time Exit
            if days_held >= self.max_holding_days:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=shares, order_type="SELL",
                    reason=f"Time Exit ({self.max_holding_days} days)"
                ))

        return orders

    # ─────────────────────────────────────────────────────────────────────
    # BUY logic
    # ─────────────────────────────────────────────────────────────────────
    def generate_buys(self, current_date: str, day_data: pd.DataFrame,
                      portfolio: Portfolio) -> List[Order]:
        # Regime check
        if self.use_regime_filter and not self._is_bull_regime(day_data):
            return []

        orders = []

        # סינון יקום בסיסי
        valid_data = self._stocks_only(day_data)
        valid_data = valid_data.dropna(subset=['RSI_14', 'Adj_Close']).copy()

        # סינון מחיר ונזילות
        valid_data = valid_data[valid_data['Adj_Close'] >= self.min_price]
        if 'Dollar_Volume_20d_Avg' in valid_data.columns:
            valid_data = valid_data.dropna(subset=['Dollar_Volume_20d_Avg'])
            valid_data = valid_data[valid_data['Dollar_Volume_20d_Avg'] >= self.min_dv]

        # סינון Dow primary trend (long-term uptrend)
        if self.require_uptrend and 'Return_252d_Pct' in valid_data.columns:
            valid_data = valid_data.dropna(subset=['Return_252d_Pct'])
            valid_data = valid_data[valid_data['Return_252d_Pct'] > 0]

        # סינון oversold
        valid_data = valid_data[valid_data['RSI_14'] < self.rsi_buy_threshold]

        if valid_data.empty:
            return orders

        # אל תקנה מניה שכבר בתיק
        valid_data = valid_data[~valid_data['Ticker'].isin(portfolio.positions.keys())]
        if valid_data.empty:
            return orders

        # בדיקת מקום פנוי בתיק
        slots_available = self.top_n - len(portfolio.positions)
        if slots_available <= 0:
            return orders

        # מיון: הכי oversold ראשון (RSI הנמוך ביותר)
        candidates = valid_data.sort_values(by='RSI_14', ascending=True).head(slots_available)
        if candidates.empty:
            return orders

        # הקצאת הון — equal weight (מתאים יותר ל-mean reversion מ-relative_dv)
        safe_cash = portfolio.cash * 0.98
        cash_per_position = safe_cash / len(candidates)
        virtual_cash = portfolio.cash

        for _, row in candidates.iterrows():
            ticker = row['Ticker']
            price = row['Adj_Close']
            rsi = row['RSI_14']

            shares = int(cash_per_position // price)
            if shares <= 0:
                continue

            # חישוב עלות משוערת בהתאם ל-core_engine (slippage + commission)
            estimated_cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission

            if estimated_cost <= virtual_cash:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=price,
                    shares=shares, order_type="BUY",
                    reason=f"RSI Oversold Entry (RSI={rsi:.1f})"
                ))
                virtual_cash -= estimated_cost

        return orders
