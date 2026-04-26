"""
Unit tests for core_engine.py — TaxLedger, Portfolio, BacktestEngine.
Run with: pytest tests/test_core_engine.py -v
"""
import pytest
import pandas as pd
from typing import List

from core_engine import (
    Order, TaxLedger, Portfolio, BacktestEngine,
    BaseStrategy,
)

# ─── Constants ───────────────────────────────────────────────────────────────

TAX_RATE   = 0.25
COMMISSION = 1.0
SLIPPAGE   = 0.001
CAPITAL    = 100_000.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_portfolio(capital: float = CAPITAL) -> Portfolio:
    return Portfolio(capital, TaxLedger(TAX_RATE), commission_per_trade=COMMISSION, slippage_pct=SLIPPAGE)


def buy(ticker="AAPL", price=100.0, shares=10.0, date="2020-01-02", avg_dv=None) -> Order:
    return Order(ticker=ticker, date=date, price=price, shares=shares,
                 order_type="BUY", avg_dollar_volume=avg_dv)


def sell(ticker="AAPL", price=120.0, shares=10.0, date="2020-06-01", avg_dv=None) -> Order:
    return Order(ticker=ticker, date=date, price=price, shares=shares,
                 order_type="SELL", avg_dollar_volume=avg_dv)


# ─── TaxLedger ───────────────────────────────────────────────────────────────

class TestTaxLedger:

    def test_loss_adds_to_carryforward_no_tax(self):
        ledger = TaxLedger(TAX_RATE)
        net = ledger.process_realized_pnl("2020-01-01", -500.0, "AAPL")
        assert net == -500.0
        assert ledger.loss_carryforward == 500.0
        assert ledger.total_tax_paid == 0.0

    def test_gain_no_carryforward_taxed_at_rate(self):
        ledger = TaxLedger(TAX_RATE)
        net = ledger.process_realized_pnl("2020-01-01", 1_000.0, "AAPL")
        assert net == pytest.approx(750.0)
        assert ledger.total_tax_paid == pytest.approx(250.0)

    def test_gain_smaller_than_carryforward_fully_offset(self):
        ledger = TaxLedger(TAX_RATE)
        ledger.process_realized_pnl("2020-01-01", -1_000.0, "AAPL")   # carryforward = 1000
        net = ledger.process_realized_pnl("2020-02-01",   600.0, "GOOG")  # fully offset
        assert net == pytest.approx(600.0)        # no tax
        assert ledger.total_tax_paid == pytest.approx(0.0)
        assert ledger.loss_carryforward == pytest.approx(400.0)

    def test_gain_larger_than_carryforward_taxed_on_remainder(self):
        ledger = TaxLedger(TAX_RATE)
        ledger.process_realized_pnl("2020-01-01", -200.0, "AAPL")   # carryforward = 200
        net = ledger.process_realized_pnl("2020-02-01",  500.0, "GOOG")  # taxable = 300
        expected_tax = 300.0 * TAX_RATE
        assert net == pytest.approx(500.0 - expected_tax)
        assert ledger.total_tax_paid == pytest.approx(expected_tax)
        assert ledger.loss_carryforward == pytest.approx(0.0)

    def test_multiple_gains_accumulate_total_tax(self):
        ledger = TaxLedger(TAX_RATE)
        ledger.process_realized_pnl("2020-01-01", 400.0, "A")
        ledger.process_realized_pnl("2020-02-01", 600.0, "B")
        assert ledger.total_tax_paid == pytest.approx(1_000.0 * TAX_RATE)

    def test_history_has_one_entry_per_call(self):
        ledger = TaxLedger()
        ledger.process_realized_pnl("2020-01-01", -100.0, "X")
        ledger.process_realized_pnl("2020-02-01",  200.0, "Y")
        assert len(ledger.history) == 2

    def test_net_pnl_equals_gross_minus_tax(self):
        ledger = TaxLedger(TAX_RATE)
        gross = 800.0
        net = ledger.process_realized_pnl("2020-01-01", gross, "Z")
        assert net == pytest.approx(gross * (1 - TAX_RATE))


# ─── Portfolio — Slippage ────────────────────────────────────────────────────

class TestSlippage:

    def setup_method(self):
        self.p = make_portfolio()

    def test_none_avg_dv_returns_flat_slippage(self):
        assert self.p._compute_slippage(100.0, 10.0, None) == SLIPPAGE

    def test_zero_avg_dv_returns_flat_slippage(self):
        assert self.p._compute_slippage(100.0, 10.0, 0.0) == SLIPPAGE

    def test_large_order_increases_slippage(self):
        # $1M order into $2M avg daily volume → participation = 0.5
        s = self.p._compute_slippage(100.0, 10_000.0, 2_000_000.0)
        assert s > SLIPPAGE

    def test_extreme_order_capped_at_5_percent(self):
        s = self.p._compute_slippage(100.0, 1_000_000.0, 100.0)
        assert s == pytest.approx(0.05)

    def test_slippage_monotonically_increases_with_order_size(self):
        s_small = self.p._compute_slippage(100.0,    100.0, 10_000_000.0)
        s_large = self.p._compute_slippage(100.0, 10_000.0, 10_000_000.0)
        assert s_large > s_small


# ─── Portfolio — Execute Buy ─────────────────────────────────────────────────

class TestExecuteBuy:

    def setup_method(self):
        self.p = make_portfolio(CAPITAL)

    def test_buy_creates_position(self):
        self.p.execute_buy(buy())
        assert "AAPL" in self.p.positions

    def test_buy_reduces_cash(self):
        cash_before = self.p.cash
        self.p.execute_buy(buy(price=100.0, shares=10.0))
        assert self.p.cash < cash_before

    def test_buy_cash_reduction_matches_cost(self):
        # actual_price = 100 * (1 + 0.001) = 100.1; cost = 100.1 * 10 + 1 = 1002
        self.p.execute_buy(buy(price=100.0, shares=10.0, avg_dv=None))
        expected_actual_price = 100.0 * (1 + SLIPPAGE)
        expected_cash = CAPITAL - (expected_actual_price * 10.0 + COMMISSION)
        assert self.p.cash == pytest.approx(expected_cash)

    def test_buy_stores_correct_share_count(self):
        self.p.execute_buy(buy(shares=15.0))
        assert self.p.positions["AAPL"]["shares"] == 15.0

    def test_buy_stores_slippage_adjusted_price(self):
        self.p.execute_buy(buy(price=100.0, avg_dv=None))
        stored = self.p.positions["AAPL"]["buy_price"]
        assert stored == pytest.approx(100.0 * (1 + SLIPPAGE))

    def test_insufficient_cash_rejects_buy(self):
        self.p.execute_buy(buy(price=100_000.0, shares=100.0))  # costs ~$10M
        assert "AAPL" not in self.p.positions
        assert self.p.cash == pytest.approx(CAPITAL)

    def test_buy_stores_date(self):
        self.p.execute_buy(buy(date="2021-03-15"))
        assert self.p.positions["AAPL"]["buy_date"] == "2021-03-15"


# ─── Portfolio — Execute Sell ────────────────────────────────────────────────

class TestExecuteSell:

    def setup_method(self):
        self.p = make_portfolio(CAPITAL)
        self.p.execute_buy(buy(price=100.0, shares=10.0, date="2020-01-02"))
        self.cash_after_buy = self.p.cash

    def test_full_sell_removes_position(self):
        self.p.execute_sell(sell(shares=10.0))
        assert "AAPL" not in self.p.positions

    def test_sell_increases_cash(self):
        self.p.execute_sell(sell(price=120.0, shares=10.0))
        assert self.p.cash > self.cash_after_buy

    def test_profitable_sell_positive_gross_pnl(self):
        self.p.execute_sell(sell(price=150.0, shares=10.0))
        assert self.p.trade_history[-1]["Gross_PnL"] > 0

    def test_loss_sell_negative_gross_pnl(self):
        self.p.execute_sell(sell(price=50.0, shares=10.0))
        assert self.p.trade_history[-1]["Gross_PnL"] < 0

    def test_partial_sell_reduces_shares(self):
        self.p.execute_sell(sell(shares=4.0))
        assert "AAPL" in self.p.positions
        assert self.p.positions["AAPL"]["shares"] == pytest.approx(6.0)

    def test_oversell_capped_and_position_closed(self):
        self.p.execute_sell(sell(shares=999.0))   # only own 10
        assert "AAPL" not in self.p.positions

    def test_sell_nonexistent_ticker_leaves_cash_unchanged(self):
        cash_before = self.p.cash
        self.p.execute_sell(sell(ticker="MSFT"))
        assert self.p.cash == cash_before

    def test_hold_days_positive(self):
        self.p.execute_sell(sell(date="2020-06-01"))
        assert self.p.trade_history[-1]["Hold_Days"] > 0

    def test_hold_days_calculation(self):
        self.p.execute_sell(sell(date="2020-01-12"))  # 10 days after 2020-01-02
        assert self.p.trade_history[-1]["Hold_Days"] == 10

    def test_trade_history_recorded(self):
        self.p.execute_sell(sell())
        assert len(self.p.trade_history) == 1
        trade = self.p.trade_history[0]
        assert trade["Ticker"] == "AAPL"


# ─── Portfolio — Get Equity ──────────────────────────────────────────────────

class TestGetEquity:

    def test_equity_cash_only_equals_capital(self):
        p = make_portfolio(50_000.0)
        assert p.get_equity({}) == pytest.approx(50_000.0)

    def test_equity_with_position_uses_current_price(self):
        p = make_portfolio(CAPITAL)
        p.execute_buy(buy(price=100.0, shares=10.0))
        eq = p.get_equity({"AAPL": 200.0})
        assert eq == pytest.approx(p.cash + 10.0 * 200.0)

    def test_equity_above_initial_when_position_appreciates(self):
        p = make_portfolio(CAPITAL)
        p.execute_buy(buy(price=100.0, shares=10.0))
        eq = p.get_equity({"AAPL": 200.0})
        assert eq > CAPITAL

    def test_equity_falls_back_to_buy_price_when_no_current_price(self):
        p = make_portfolio(CAPITAL)
        p.execute_buy(buy(price=100.0, shares=10.0, avg_dv=None))
        buy_price = p.positions["AAPL"]["buy_price"]
        eq_no_price   = p.get_equity({})
        eq_with_price  = p.get_equity({"AAPL": buy_price})
        assert eq_no_price == pytest.approx(eq_with_price)

    def test_equity_is_cash_plus_all_position_values(self):
        p = make_portfolio(CAPITAL)
        p.execute_buy(buy("AAPL", price=100.0, shares=5.0))
        p.execute_buy(buy("MSFT", price=200.0, shares=3.0))
        eq = p.get_equity({"AAPL": 110.0, "MSFT": 210.0})
        assert eq == pytest.approx(p.cash + 5.0 * 110.0 + 3.0 * 210.0)


# ─── BacktestEngine ──────────────────────────────────────────────────────────

class _BuyOnDay1SellOnDay3(BaseStrategy):
    """Buy AAPL on first day, sell on third day."""

    _BUY_DATE  = pd.Timestamp("2020-01-02")
    _SELL_DATE = pd.Timestamp("2020-01-06")

    def generate_sells(self, current_date, day_data, portfolio) -> List[Order]:
        if current_date == self._SELL_DATE and "AAPL" in portfolio.positions:
            price = float(day_data.loc[day_data["Ticker"] == "AAPL", "Adj_Close"].iloc[0])
            shares = portfolio.positions["AAPL"]["shares"]
            return [Order("AAPL", str(current_date), price, shares, "SELL")]
        return []

    def generate_buys(self, current_date, day_data, portfolio) -> List[Order]:
        if current_date == self._BUY_DATE and "AAPL" not in portfolio.positions:
            price = float(day_data.loc[day_data["Ticker"] == "AAPL", "Adj_Close"].iloc[0])
            return [Order("AAPL", str(current_date), price, 10.0, "BUY")]
        return []


class _NeverTradesStrategy(BaseStrategy):
    def generate_sells(self, *_) -> List[Order]:
        return []
    def generate_buys(self, *_) -> List[Order]:
        return []


def _make_market_data() -> pd.DataFrame:
    dates  = pd.date_range("2020-01-02", periods=5, freq="B")
    prices = [100.0, 105.0, 110.0, 115.0, 120.0]
    return pd.DataFrame({"Date": dates, "Ticker": "AAPL", "Adj_Close": prices})


class TestBacktestEngine:

    def setup_method(self):
        self.data = _make_market_data()

    def _run(self, strategy=None):
        strategy = strategy or _BuyOnDay1SellOnDay3()
        engine = BacktestEngine(self.data, strategy, initial_capital=CAPITAL)
        return engine.run()

    def test_run_returns_required_keys(self):
        result = self._run()
        assert set(result.keys()) == {"Equity_Curve", "Trades", "Taxes"}

    def test_equity_curve_one_row_per_trading_day(self):
        result = self._run()
        n_days = self.data["Date"].nunique()
        assert len(result["Equity_Curve"]) == n_days

    def test_equity_curve_has_expected_columns(self):
        result = self._run()
        assert {"Date", "Capital", "Cash", "Open_Positions"}.issubset(result["Equity_Curve"].columns)

    def test_no_trades_strategy_produces_empty_trades(self):
        result = self._run(_NeverTradesStrategy())
        assert len(result["Trades"]) == 0

    def test_trade_recorded_after_sell(self):
        result = self._run()
        assert len(result["Trades"]) == 1
        assert result["Trades"].iloc[0]["Ticker"] == "AAPL"

    def test_equity_stays_near_initial_capital_after_buy(self):
        result = self._run()
        first_equity = result["Equity_Curve"].iloc[0]["Capital"]
        assert abs(first_equity - CAPITAL) < 5_000.0

    def test_no_trades_equity_equals_initial_capital_throughout(self):
        result = self._run(_NeverTradesStrategy())
        for _, row in result["Equity_Curve"].iterrows():
            assert row["Capital"] == pytest.approx(CAPITAL)

    def test_open_positions_reflects_portfolio_state(self):
        result = self._run()
        ec = result["Equity_Curve"]
        # Day 1: bought AAPL → 1 open position
        assert ec.iloc[0]["Open_Positions"] == 1
        # Day 3 (index 2): sold → 0 open positions
        assert ec.iloc[2]["Open_Positions"] == 0

    def test_tax_entry_created_after_profitable_sell(self):
        result = self._run()
        assert len(result["Taxes"]) == 1
