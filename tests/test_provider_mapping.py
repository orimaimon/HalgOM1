"""The TradingView field mapping lives in the provider module; fields.COLUMNS only marks provider columns."""
import screener.data.tradingview as tv
from screener.fields import CANDLES, COLUMNS, provider_columns

# This snapshot represents the literal result of `list(tv_fields()) + list(EXTRA_TV_FIELDS)`
# as it was output from `origin/main` before refactoring.
FIELDS_SNAPSHOT = [
    "name", "description", "exchange", "sector", "industry", "country", "type", "indexes",
    "number_of_employees", "close", "change", "change_abs", "gap", "premarket_change",
    "postmarket_change", "volume", "average_volume_30d_calc", "relative_volume_10d_calc",
    "market_cap_basic", "float_shares_outstanding", "total_shares_outstanding_current",
    "price_earnings_ttm", "earnings_per_share_forecast_next_fy", "price_earnings_growth_ttm",
    "price_sales_current", "price_book_fq", "price_to_cash_f_operating_activities_ttm",
    "price_free_cash_flow_ttm", "enterprise_value_ebitda_ttm", "earnings_per_share_diluted_ttm",
    "earnings_per_share_diluted_yoy_growth_ttm", "earnings_per_share_diluted_yoy_growth_fq",
    "total_revenue_ttm", "total_revenue_yoy_growth_ttm", "total_revenue_yoy_growth_fq",
    "return_on_assets", "return_on_equity", "return_on_invested_capital", "current_ratio_fq",
    "quick_ratio_fq", "long_term_debt_to_equity_fq", "debt_to_equity_fq", "gross_margin_ttm",
    "operating_margin_ttm", "net_margin_ttm", "free_cash_flow_margin_ttm", "dividends_yield_current",
    "dividend_payout_ratio_ttm", "Perf.W", "Perf.1M", "Perf.3M", "Perf.6M", "Perf.YTD", "Perf.Y",
    "Perf.5Y", "Volatility.W", "Volatility.M", "RSI", "SMA20", "SMA50", "SMA150", "SMA200",
    "close[1]", "SMA20[1]", "SMA50[1]", "SMA200[1]", "close|1W", "RSI|1W", "SMA50|1W", "High.5D",
    "Low.5D", "High.1M", "Low.1M", "High.1M[1]", "High.3M[1]", "High.6M[1]", "price_52_week_high[1]",
    "Low.1M[1]", "Low.3M[1]", "Low.6M[1]", "price_52_week_low[1]", "price_52_week_high",
    "price_52_week_low", "ATR", "ATRP", "beta_1_year", "earnings_release_next_date",
    "recommendation_mark", "price_target_average", "Candle.Engulfing.Bullish",
    "Candle.Engulfing.Bearish", "Candle.Hammer", "Candle.InvertedHammer", "Candle.HangingMan",
    "Candle.ShootingStar", "Candle.MorningStar", "Candle.EveningStar", "Candle.3WhiteSoldiers",
    "Candle.3BlackCrows", "Candle.Harami.Bullish", "Candle.Harami.Bearish", "Candle.Kicking.Bullish",
    "Candle.Kicking.Bearish", "Candle.AbandonedBaby.Bullish", "Candle.AbandonedBaby.Bearish",
    "Candle.Marubozu.White", "Candle.Marubozu.Black", "Candle.Doji", "Candle.Doji.Dragonfly",
    "Candle.Doji.Gravestone",
    "is_primary", "typespecs", "currency", "update_mode"
]


def test_tv_mapping_covers_exactly_the_provider_columns():
    provider = provider_columns()
    assert sorted(tv.TV_MAPPING.values()) == sorted(provider)     # one TradingView field per provider column
    for key in CANDLES:
        assert COLUMNS[key].source == "provider" and COLUMNS[key].kind == "flag"
        assert any(k == key and f.startswith("Candle.") for f, k in tv.TV_MAPPING.items())


def test_fetch_universe_request_and_columns_unchanged(monkeypatch):
    sent = {}

    def fake_scan(market, query, timeout):
        sent.update(query)
        return [{"s": "NASDAQ:AAPL", "d": [1.0] * len(query["columns"])}]

    monkeypatch.setattr(tv, "_scan", fake_scan)
    df = tv.fetch_universe()
    assert sent["columns"] == FIELDS_SNAPSHOT                     # same fields, same order as before
    assert set(df.columns) == {"tv_symbol", *provider_columns()}  # clean() drops the extra TV fields
