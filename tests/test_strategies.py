"""
Unit tests for strategy classes.
Run with: pytest tests/test_strategies.py -v
"""
import pytest
import pandas as pd
import numpy as np
from typing import List

from core_engine import Order, TaxLedger, Portfolio
from strategies.top_n_volume import TopNVolumeStrategy
from strategies.top_n_volume_sl import TopNVolumeSLStrategy
from strategies.rsi_meanrev import RSIMeanReversionStrategy
from strategies.factor_combo import FactorComboStrategy


# ─── Shared helpers ───────────────────────────────────────────────────────────

CAPITAL    = 100_000.0
COMMISSION = 1.0
SLIPPAGE   = 0.001

def make_portfolio(capital: float = CAPITAL) -> Portfolio:
    return Portfolio(capital, TaxLedger(), commission_per_trade=COMMISSION, slippage_pct=SLIPPAGE)


def make_day_data(
    n: int = 10,
    base_price: float = 50.0,
    base_dv: float = 50_000_000.0,
    spy_close: float = 500.0,
    spy_sma:   float = 400.0,   # bull by default
    include_rsi: bool = False,
    include_return: bool = False,
) -> pd.DataFrame:
    """יוצר DataFrame יומי עם n מניות."""
    tickers = [f"T{i:02d}" for i in range(n)]
    df = pd.DataFrame({
        "Date":                 pd.Timestamp("2021-06-01"),
        "Ticker":               tickers,
        "Type":                 "stock",
        "Adj_Close":            [base_price + i * 5 for i in range(n)],
        "Dollar_Volume_20d_Avg":[base_dv * (n - i) for i in range(n)],  # T00 has highest DV
        "SPY_Close":            spy_close,
        "SPY_SMA_200":          spy_sma,
    })
    if include_rsi:
        df["RSI_14"] = [25.0 + i * 3 for i in range(n)]   # T00 most oversold
    if include_return:
        df["Return_252d_Pct"] = [0.05 + i * 0.02 for i in range(n)]
        df["Return_120d_Pct"] = [0.03 + i * 0.01 for i in range(n)]
    return df


def _load_portfolio_with_positions(strat_class, params: dict, day_data: pd.DataFrame, date="2021-01-01") -> Portfolio:
    """בונה תיק עם פוזיציות פתוחות על ידי הפעלת generate_buys."""
    p = make_portfolio()
    strat = strat_class(**params)
    orders = strat.generate_buys(date, day_data, p)
    for o in orders:
        p.execute_buy(o)
    return p, strat


# ─── TopNVolumeStrategy ───────────────────────────────────────────────────────

class TestTopNVolumeStrategy:

    def _make(self, **kw):
        defaults = dict(top_n=5, hold_days=10, sizing_method="equal", min_price=1.0)
        defaults.update(kw)
        return TopNVolumeStrategy(**defaults)

    def test_buys_top_n_stocks(self):
        strat = self._make(top_n=3)
        p = make_portfolio()
        data = make_day_data(n=10)
        orders = strat.generate_buys("2021-06-01", data, p)
        assert len(orders) == 3

    def test_buys_highest_volume_first(self):
        strat = self._make(top_n=3)
        p = make_portfolio()
        data = make_day_data(n=10)
        orders = strat.generate_buys("2021-06-01", data, p)
        tickers = [o.ticker for o in orders]
        assert "T00" in tickers   # T00 has highest DV

    def test_no_buys_when_already_in_position(self):
        strat = self._make(top_n=3)
        p = make_portfolio()
        data = make_day_data(n=10)
        buys = strat.generate_buys("2021-06-01", data, p)
        for b in buys:
            p.execute_buy(b)
        # נסיון נוסף — כבר בפוזיציה
        orders2 = strat.generate_buys("2021-06-02", data, p)
        assert orders2 == []

    def test_no_buys_below_min_price(self):
        strat = self._make(top_n=5, min_price=1000.0)
        p = make_portfolio()
        data = make_day_data(n=10, base_price=50.0)   # כל המניות מתחת ל-1000
        orders = strat.generate_buys("2021-06-01", data, p)
        assert orders == []

    def test_time_exit_after_hold_days(self):
        strat = self._make(top_n=3, hold_days=5)
        p = make_portfolio()
        data = make_day_data(n=10)
        buys = strat.generate_buys("2021-06-01", data, p)
        for b in buys:
            p.execute_buy(b)
        # מדמים 5 ימי מסחר
        for d in range(5):
            sells = strat.generate_sells(f"2021-06-0{d+2}", data, p)
        assert len(sells) == 3
        assert all("Time Exit" in s.reason for s in sells)

    def test_no_sells_before_hold_days(self):
        strat = self._make(top_n=3, hold_days=10)
        p = make_portfolio()
        data = make_day_data(n=10)
        buys = strat.generate_buys("2021-06-01", data, p)
        for b in buys:
            p.execute_buy(b)
        # רק 3 ימים
        for d in range(3):
            sells = strat.generate_sells(f"2021-06-0{d+2}", data, p)
        assert sells == []

    def test_regime_exit_in_bear_market(self):
        strat = self._make(top_n=3, use_regime_filter=True)
        p = make_portfolio()
        bull_data = make_day_data(n=10, spy_close=500, spy_sma=400)
        buys = strat.generate_buys("2021-06-01", bull_data, p)
        for b in buys:
            p.execute_buy(b)

        bear_data = make_day_data(n=10, spy_close=300, spy_sma=400)  # SPY < SMA → Bear
        sells = strat.generate_sells("2021-06-02", bear_data, p)
        assert len(sells) == 3
        assert all("Regime" in s.reason for s in sells)

    def test_no_buys_in_bear_regime(self):
        strat = self._make(top_n=3, use_regime_filter=True)
        p = make_portfolio()
        bear_data = make_day_data(n=10, spy_close=300, spy_sma=400)
        orders = strat.generate_buys("2021-06-01", bear_data, p)
        assert orders == []

    def test_relative_dv_sizing_allocates_proportionally(self):
        strat = self._make(top_n=2, sizing_method="relative_dv")
        p = make_portfolio()
        data = make_day_data(n=5)
        orders = strat.generate_buys("2021-06-01", data, p)
        assert len(orders) == 2
        # T00 (highest DV) should get more shares than T01
        shares = {o.ticker: o.shares for o in orders}
        assert shares["T00"] > shares["T01"]

    def test_all_orders_are_buy_type(self):
        strat = self._make(top_n=3)
        p = make_portfolio()
        data = make_day_data(n=10)
        orders = strat.generate_buys("2021-06-01", data, p)
        assert all(o.order_type == "BUY" for o in orders)


# ─── TopNVolumeSLStrategy ─────────────────────────────────────────────────────

class TestTopNVolumeSLStrategy:

    def _make(self, **kw):
        defaults = dict(top_n=5, hold_days=30, min_price=1.0, stop_loss_pct=0.15)
        defaults.update(kw)
        return TopNVolumeSLStrategy(**defaults)

    def _open_position(self, strat, ticker="T00", buy_price=100.0, date="2021-01-01"):
        p = make_portfolio()
        p.positions[ticker] = {
            "shares": 100.0, "buy_price": buy_price,
            "buy_date": date, "reason": "test", "buy_commission": COMMISSION
        }
        return p

    def test_stop_loss_triggers_sell(self):
        strat = self._make(stop_loss_pct=0.15)
        p = self._open_position(strat, buy_price=100.0)
        # מחיר ירד ב-20% — מתחת ל-stop loss של 15%
        data = make_day_data(n=1, base_price=79.0)
        data.iloc[0, data.columns.get_loc("Ticker")] = "T00"
        sells = strat.generate_sells("2021-06-01", data, p)
        assert len(sells) == 1
        assert "Stop Loss" in sells[0].reason

    def test_no_stop_loss_within_threshold(self):
        strat = self._make(stop_loss_pct=0.15, hold_days=60)
        p = self._open_position(strat, buy_price=100.0, date="2021-05-30")
        data = make_day_data(n=1, base_price=90.0)   # ירד 10% — מעל stop loss של 15%
        data.iloc[0, data.columns.get_loc("Ticker")] = "T00"
        sells = strat.generate_sells("2021-06-01", data, p)  # רק 2 ימים — לא time exit
        assert sells == []

    def test_time_exit_by_calendar_days(self):
        strat = self._make(hold_days=30)
        p = make_portfolio()
        p.positions["T00"] = {
            "shares": 100.0, "buy_price": 100.0,
            "buy_date": "2021-01-01", "reason": "test", "buy_commission": COMMISSION
        }
        data = make_day_data(n=1, base_price=100.0)
        data.iloc[0, data.columns.get_loc("Ticker")] = "T00"
        sells = strat.generate_sells("2021-02-01", data, p)   # 31 ימים לאחר מכן
        assert len(sells) == 1
        assert "Time Exit" in sells[0].reason

    def test_fills_open_slots_on_buy(self):
        strat = self._make(top_n=5)
        p = make_portfolio()
        # 2 פוזיציות קיימות → צריך למלא עוד 3
        p.positions["EXISTING1"] = {"shares": 10, "buy_price": 50, "buy_date": "2021-01-01", "reason": "", "buy_commission": 1}
        p.positions["EXISTING2"] = {"shares": 10, "buy_price": 50, "buy_date": "2021-01-01", "reason": "", "buy_commission": 1}
        data = make_day_data(n=10)
        # מסירים את המניות הקיימות מהדאטה
        data = data[~data["Ticker"].isin(["EXISTING1", "EXISTING2"])].reset_index(drop=True)
        orders = strat.generate_buys("2021-06-01", data, p)
        assert len(orders) == 3

    def test_no_buys_when_portfolio_full(self):
        strat = self._make(top_n=3)
        p = make_portfolio()
        for i in range(3):
            p.positions[f"X{i}"] = {"shares": 10, "buy_price": 50, "buy_date": "2021-01-01", "reason": "", "buy_commission": 1}
        orders = strat.generate_buys("2021-06-01", make_day_data(n=10), p)
        assert orders == []

    def test_no_duplicate_positions(self):
        strat = self._make(top_n=5)
        p = make_portfolio()
        data = make_day_data(n=10)
        # T00 כבר בפוזיציה
        p.positions["T00"] = {"shares": 10, "buy_price": 50, "buy_date": "2021-01-01", "reason": "", "buy_commission": 1}
        orders = strat.generate_buys("2021-06-01", data, p)
        tickers = [o.ticker for o in orders]
        assert "T00" not in tickers


# ─── RSIMeanReversionStrategy ─────────────────────────────────────────────────

class TestRSIMeanReversionStrategy:

    def _make(self, **kw):
        defaults = dict(top_n=5, rsi_buy_threshold=30.0, rsi_sell_threshold=60.0,
                        max_holding_days=20, stop_loss_pct=0.10,
                        min_price=1.0, min_dollar_volume=1_000_000,
                        require_uptrend=False, use_regime_filter=False)
        defaults.update(kw)
        return RSIMeanReversionStrategy(**defaults)

    def test_buys_only_oversold(self):
        strat = self._make(top_n=5, rsi_buy_threshold=30.0)
        p = make_portfolio()
        data = make_day_data(n=10, include_rsi=True)
        # RSI: T00=25, T01=28, T02=31... רק T00 ו-T01 מתחת ל-30
        orders = strat.generate_buys("2021-06-01", data, p)
        assert all(o.ticker in ["T00", "T01"] for o in orders)

    def test_buys_most_oversold_first(self):
        strat = self._make(top_n=3, rsi_buy_threshold=40.0)
        p = make_portfolio()
        data = make_day_data(n=10, include_rsi=True)
        orders = strat.generate_buys("2021-06-01", data, p)
        if len(orders) >= 2:
            # T00 (RSI=25) צריך להופיע לפני T01 (RSI=28)
            tickers = [o.ticker for o in orders]
            assert tickers.index("T00") < tickers.index("T01")

    def test_mean_reversion_exit_when_rsi_recovers(self):
        strat = self._make(rsi_sell_threshold=60.0)
        p = make_portfolio()
        p.positions["T00"] = {"shares": 100, "buy_price": 50.0,
                              "buy_date": "2021-01-01", "reason": "", "buy_commission": 1}
        data = make_day_data(n=1, include_rsi=True)
        data.iloc[0, data.columns.get_loc("Ticker")] = "T00"
        data["RSI_14"] = 65.0   # מעל rsi_sell_threshold=60
        sells = strat.generate_sells("2021-06-01", data, p)
        assert len(sells) == 1
        assert "Mean Reversion" in sells[0].reason

    def test_no_exit_when_rsi_still_low(self):
        strat = self._make(rsi_sell_threshold=60.0, max_holding_days=100)
        p = make_portfolio()
        p.positions["T00"] = {"shares": 100, "buy_price": 50.0,
                              "buy_date": "2021-05-30", "reason": "", "buy_commission": 1}
        data = make_day_data(n=1, include_rsi=True)
        data.iloc[0, data.columns.get_loc("Ticker")] = "T00"
        data["RSI_14"] = 45.0   # מתחת ל-60 → לא יוצאים
        sells = strat.generate_sells("2021-06-01", data, p)
        assert sells == []

    def test_uptrend_filter_excludes_negative_return(self):
        strat = self._make(top_n=5, rsi_buy_threshold=40.0, require_uptrend=True)
        p = make_portfolio()
        data = make_day_data(n=10, include_rsi=True, include_return=True)
        # נגדיר תשואה שלילית לכולן חוץ מ-T09
        data["Return_252d_Pct"] = -0.05
        data.loc[data["Ticker"] == "T09", "Return_252d_Pct"] = 0.10
        orders = strat.generate_buys("2021-06-01", data, p)
        # רק T09 עובר את סינון המגמה, ורק אם RSI < threshold
        tickers = [o.ticker for o in orders]
        for t in tickers:
            assert data.loc[data["Ticker"] == t, "Return_252d_Pct"].values[0] > 0

    def test_stop_loss_priority_over_mean_reversion(self):
        strat = self._make(stop_loss_pct=0.10, rsi_sell_threshold=60.0)
        p = make_portfolio()
        p.positions["T00"] = {"shares": 100, "buy_price": 100.0,
                              "buy_date": "2021-01-01", "reason": "", "buy_commission": 1}
        data = make_day_data(n=1, base_price=85.0, include_rsi=True)  # ירד 15% → Stop Loss
        data.iloc[0, data.columns.get_loc("Ticker")] = "T00"
        data["RSI_14"] = 70.0   # גם RSI גבוה → Stop Loss אמור לנצח (continue)
        sells = strat.generate_sells("2021-06-01", data, p)
        assert len(sells) == 1
        assert "Stop Loss" in sells[0].reason


# ─── FactorComboStrategy ──────────────────────────────────────────────────────

class TestFactorComboStrategy:

    def _make(self, **kw):
        defaults = dict(top_n=5, hold_days=20, momentum_weight=0.4,
                        rsi_weight=0.3, volume_weight=0.3,
                        min_price=1.0, min_dollar_volume=1_000_000,
                        stop_loss_pct=0.15, use_regime_filter=False)
        defaults.update(kw)
        return FactorComboStrategy(**defaults)

    def test_buys_top_n_by_combined_score(self):
        strat = self._make(top_n=3)
        p = make_portfolio()
        data = make_day_data(n=10, include_rsi=True, include_return=True)
        orders = strat.generate_buys("2021-06-01", data, p)
        assert len(orders) <= 3
        assert all(o.order_type == "BUY" for o in orders)

    def test_excludes_stocks_below_min_dollar_volume(self):
        strat = self._make(top_n=5, min_dollar_volume=500_000_000.0)
        p = make_portfolio()
        data = make_day_data(n=10, base_dv=1_000_000, include_rsi=True, include_return=True)
        # כל ה-DV קטן מ-500M
        orders = strat.generate_buys("2021-06-01", data, p)
        assert orders == []

    def test_factor_scores_sum_to_one(self):
        from mc_strategies import build_factor_combo_strategy
        params = {"top_n": 5, "hold_days": 20, "momentum_weight": 0.5,
                  "rsi_weight": 0.3, "volume_weight": 0.2, "min_price": 1.0,
                  "min_dollar_volume": 1_000_000, "stop_loss_pct": 0.15,
                  "use_regime_filter": False}
        strat = build_factor_combo_strategy(params)
        total = strat.momentum_weight + strat.rsi_weight + strat.volume_weight
        assert total == pytest.approx(1.0)

    def test_zero_weights_default_to_equal(self):
        from mc_strategies import build_factor_combo_strategy
        params = {"top_n": 5, "hold_days": 20, "momentum_weight": 0.0,
                  "rsi_weight": 0.0, "volume_weight": 0.0, "min_price": 1.0,
                  "min_dollar_volume": 1_000_000, "stop_loss_pct": 0.15,
                  "use_regime_filter": False}
        strat = build_factor_combo_strategy(params)
        assert strat.momentum_weight == pytest.approx(0.33, abs=0.01)
        assert strat.rsi_weight      == pytest.approx(0.33, abs=0.01)
        assert strat.volume_weight   == pytest.approx(0.34, abs=0.01)
