import pandas as pd
import numpy as np
from typing import List
import logging
from core_engine import BaseEquityStrategy, Portfolio, Order

logger = logging.getLogger("TopNDvMcStrategy")


class TopNDvMcStrategy(BaseEquityStrategy):
    """
    DV/MC Composite Factor Strategy.

    הסיבה לפורמט הקומפוזיטי:
    ■ DV/MC לבד (cross-sectional) בוחר small/mid-caps ספקולטיביות — תנודתי מאוד
    ■ DV/MC + מומנטום כ-Z-scores משולבים מאזן בין פעילות מוסדית, כיוון מחיר
      ועוצמת תנועה — מייצר אות יציב יותר

    ציון = w_dvmc * rank(DV_MC)
          + w_mom  * rank(Momentum)
          - w_vol  * rank(Daily_Volatility)

    כל rank הוא percentile יומי (0-1) בתוך יקום הסינון.
    מניות עם ציון גבוה: פעילות מוסדית + מגמה + תנודתיות נמוכה יחסית.
    """

    def __init__(self,
                 top_n: int = 10,
                 hold_days: int = 30,
                 min_price: float = 5.0,
                 min_market_cap: float = 1_000_000_000,
                 min_dollar_volume: float = 20_000_000,
                 dvmc_weight: float = 0.4,
                 mom_weight: float = 0.4,
                 vol_weight: float = 0.2,
                 momentum_col: str = "Return_120d_Pct",
                 min_momentum: float = 0.05,        # מינימום מומנטום לכניסה
                 stop_loss_pct: float = 0.15,
                 use_regime_filter: bool = True,
                 sizing_method: str = "equal"):

        self.top_n = top_n
        self.hold_days = hold_days
        self.min_price = min_price
        self.min_market_cap = min_market_cap
        self.min_dv = min_dollar_volume
        self.dvmc_weight = dvmc_weight
        self.mom_weight = mom_weight
        self.vol_weight = vol_weight
        self.momentum_col = momentum_col
        self.min_momentum = min_momentum
        self.stop_loss_pct = stop_loss_pct
        self.use_regime_filter = use_regime_filter
        self.sizing_method = sizing_method
        self._regime_block_logged_date = None

    def _is_allowed_regime(self, day_data: pd.DataFrame) -> bool:
        """בודק אם המשטר מאפשר מסחר. תומך גם ב-Market_Regime וגם ב-SPY SMA."""
        if not self.use_regime_filter:
            return True
        # עדיפות: Market_Regime (מורכב יותר) אם קיים
        if 'Market_Regime' in day_data.columns:
            regime = day_data['Market_Regime'].dropna()
            if not regime.empty:
                r = str(regime.iloc[0]).lower()
                return r in ('bull', 'sideways')
        # fallback: SPY SMA 200
        return self._is_bull_regime(day_data)

    # ──────────────────────────────────────────────────────────────────────────
    # Sells
    # ──────────────────────────────────────────────────────────────────────────

    def generate_sells(self, current_date: str, day_data: pd.DataFrame,
                       portfolio: Portfolio) -> List[Order]:
        if not portfolio.positions:
            return []

        if not self._is_allowed_regime(day_data):
            if self._regime_block_logged_date != current_date:
                logger.info(f"{current_date}: Regime → BEAR, liquidating")
                self._regime_block_logged_date = current_date
            return self._generate_regime_exit_orders(current_date, day_data, portfolio)

        orders = []
        ticker_index = self._build_ticker_index(day_data)
        curr_dt = pd.to_datetime(current_date)

        for ticker, pos in list(portfolio.positions.items()):
            row = ticker_index.get(ticker)
            if row is None:
                continue

            current_price = float(row['Adj_Close'])
            dv = float(row['Dollar_Volume_20d_Avg']) if pd.notna(row.get('Dollar_Volume_20d_Avg')) else None
            days_held = (curr_dt - pd.to_datetime(pos['buy_date'])).days

            if current_price <= pos['buy_price'] * (1 - self.stop_loss_pct):
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=pos['shares'], order_type="SELL",
                    reason=f"Stop Loss ({self.stop_loss_pct*100:.1f}%)",
                    avg_dollar_volume=dv,
                ))
                continue

            if days_held >= self.hold_days:
                orders.append(Order(
                    ticker=ticker, date=current_date, price=current_price,
                    shares=pos['shares'], order_type="SELL",
                    reason=f"Time Exit ({self.hold_days}d)",
                    avg_dollar_volume=dv,
                ))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    # Buys — Composite Factor Score
    # ──────────────────────────────────────────────────────────────────────────

    def generate_buys(self, current_date: str, day_data: pd.DataFrame,
                      portfolio: Portfolio) -> List[Order]:
        if not self._is_allowed_regime(day_data):
            return []

        slots = self.top_n - len(portfolio.positions)
        if slots <= 0:
            return []

        # ── סינון בסיסי ──────────────────────────────────────────────────────
        valid = self._get_stocks_only(day_data)
        needed = ['Adj_Close', 'Dollar_Volume_20d_Avg', 'Market_Cap',
                  self.momentum_col, 'Daily_Volatility_20d']
        valid = valid.dropna(subset=needed).copy()
        valid = valid[valid['Adj_Close'] >= self.min_price]
        valid = valid[valid['Market_Cap'] >= self.min_market_cap]
        valid = valid[valid['Dollar_Volume_20d_Avg'] >= self.min_dv]
        valid = valid[valid[self.momentum_col] >= self.min_momentum]
        valid = valid[~valid['Ticker'].isin(portfolio.positions)]

        if len(valid) < max(10, slots):
            return []

        # ── חישוב DV/MC עם cap על outliers ───────────────────────────────────
        valid['DV_MC'] = valid['Dollar_Volume_20d_Avg'] / valid['Market_Cap']
        p99 = valid['DV_MC'].quantile(0.99)
        valid['DV_MC'] = valid['DV_MC'].clip(upper=p99)

        # ── Composite Score (percentile ranks — scale 0 to 1) ─────────────────
        valid['r_dvmc'] = valid['DV_MC'].rank(pct=True)
        valid['r_mom']  = valid[self.momentum_col].rank(pct=True)
        valid['r_vol']  = valid['Daily_Volatility_20d'].rank(pct=True, ascending=False)

        total_w = self.dvmc_weight + self.mom_weight + self.vol_weight
        valid['score'] = (
            self.dvmc_weight * valid['r_dvmc'] +
            self.mom_weight  * valid['r_mom']  +
            self.vol_weight  * valid['r_vol']
        ) / total_w

        candidates = valid.nlargest(slots, 'score')

        orders = []
        virtual_cash = portfolio.cash
        total_score = candidates['score'].sum()

        for _, row in candidates.iterrows():
            ticker = row['Ticker']
            price = float(row['Adj_Close'])
            dv = float(row['Dollar_Volume_20d_Avg'])
            score = float(row['score'])
            dvmc = float(row['DV_MC'])
            mom = float(row[self.momentum_col])

            if self.sizing_method == "score_weighted" and total_score > 0:
                weight = score / total_score
                shares = int((virtual_cash * 0.98 * weight) // price)
            else:
                shares = int(self._calculate_position_size(
                    virtual_cash, self.top_n,
                    len(portfolio.positions) + len(orders), price
                ))

            if shares > 0:
                cost = (shares * price * (1 + portfolio.slippage_pct)) + portfolio.commission
                if cost <= virtual_cash:
                    orders.append(Order(
                        ticker=ticker, date=current_date, price=price,
                        shares=shares, order_type="BUY",
                        reason=f"DvMC+Mom (score={score:.2f} dvmc={dvmc:.4f} mom={mom*100:.0f}%)",
                        avg_dollar_volume=dv,
                    ))
                    virtual_cash -= cost

        return orders
