import pandas as pd
from typing import List, Optional
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("TopNVolumeStrategy")

class TopNVolumeStrategy(BaseEquityStrategy):
    """
    אסטרטגיית מחזור דולרי (Top N) בסיסית — גרסה אופטימלית.
    קונה סל מניות ומחזיקה אותן לפרק זמן קצוב (ימי מסחר).

    regime_entry_buffer: SPY must be above SMA200 by this fraction before re-entry.
                         0.02 = 2% cushion. Prevents whipsaw in choppy markets.
    regime_confirm_days: SPY must hold above the entry threshold for this many
                         consecutive days before the strategy re-enters.
                         Both default to 0/1 → identical behaviour to the old filter.
    """
    def __init__(self, top_n: int = 10, hold_days: int = 90,
                 measure_column: str = "Dollar_Volume_20d_Avg",
                 sizing_method: str = "equal",
                 min_price: float = 1.0,
                 use_regime_filter: bool = False,
                 regime_entry_buffer: float = 0.0,
                 regime_confirm_days: int = 1):
        self.top_n = top_n
        self.hold_days = hold_days
        self.measure_column = measure_column
        self.sizing_method = sizing_method
        self.min_price = min_price
        self.use_regime_filter = use_regime_filter
        self.regime_entry_buffer = regime_entry_buffer
        self.regime_confirm_days = max(1, regime_confirm_days)

        self.days_in_trade = 0

        # Stateful regime tracking
        self._in_bull: bool = True       # confirmed regime
        self._bull_streak: int = 0       # consecutive days above entry threshold
        self._prev_in_bull: Optional[bool] = None  # for transition logging

    # ── Regime helpers ────────────────────────────────────────────────────────

    def _get_spy_values(self, day_data: pd.DataFrame):
        """Returns (spy_close, spy_sma) or (None, None)."""
        if 'SPY_Close' in day_data.columns and 'SPY_SMA_200' in day_data.columns:
            row = day_data[['SPY_Close', 'SPY_SMA_200']].dropna()
            if not row.empty:
                return float(row.iloc[0]['SPY_Close']), float(row.iloc[0]['SPY_SMA_200'])
        spy_row = day_data[day_data['Ticker'] == 'SPY']
        if not spy_row.empty:
            c = spy_row.iloc[0].get('Adj_Close')
            s = spy_row.iloc[0].get('SPY_SMA_200')
            if pd.notna(c) and pd.notna(s):
                return float(c), float(s)
        return None, None

    def _update_regime(self, current_date: str, day_data: pd.DataFrame) -> bool:
        """
        Stateful regime update — call once per day from generate_sells.

        Exit logic  (fast):  switch to BEAR immediately when SPY < SMA200.
                             Capital protection takes priority.
        Entry logic (slow):  switch to BULL only when SPY > SMA200*(1+buffer)
                             for confirm_days consecutive days.
                             Eliminates false re-entries during choppy recoveries.
        """
        spy, sma = self._get_spy_values(day_data)
        if spy is None:
            return self._in_bull  # no data — hold current regime

        if self._in_bull:
            if spy < sma:
                self._in_bull = False
                self._bull_streak = 0
        else:
            if spy > sma * (1 + self.regime_entry_buffer):
                self._bull_streak += 1
                if self._bull_streak >= self.regime_confirm_days:
                    self._in_bull = True
                    self._bull_streak = 0
            else:
                self._bull_streak = 0

        # Log on regime transitions only
        if self._in_bull != self._prev_in_bull:
            direction = "BULL" if self._in_bull else "BEAR"
            action = "re-entering market" if self._in_bull else "liquidating to cash"
            logger.info(f"{current_date}: Regime -> {direction}, {action}")
            self._prev_in_bull = self._in_bull

        return self._in_bull

    # ── Strategy interface ────────────────────────────────────────────────────

    def generate_sells(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if self.use_regime_filter:
            in_bull = self._update_regime(current_date, day_data)
            if not in_bull and portfolio.positions:
                self.days_in_trade = 0
                return self._generate_regime_exit_orders(current_date, day_data, portfolio)
        else:
            in_bull = True

        if not portfolio.positions:
            return []

        self.days_in_trade += 1

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
            self.days_in_trade = 0
            return orders

        return []

    def generate_buys(self, current_date: str, day_data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if portfolio.positions:
            return []

        if self.use_regime_filter and not self._in_bull:
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
                        reason="TopN Entry",
                        avg_dollar_volume=float(dv),
                    ))
                    virtual_cash -= cost

        return orders
