from screener.data.tradingview import TV_MAPPING, fetch_universe, EXTRA_TV_FIELDS
from screener.fields import COLUMNS, provider_columns, CANDLES
import pandas as pd

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


def test_tv_mapping_consistency():
    # every column with source="provider" has exactly one TradingView field
    prov_keys = provider_columns()
    mapped_keys = [v for v in TV_MAPPING.values()]
    assert set(prov_keys) == set(mapped_keys)

    # every TradingView mapping key is a provider column (no orphans)
    for k, v in TV_MAPPING.items():
        assert v in prov_keys

    # no duplicate TradingView fields
    assert len(TV_MAPPING) == len(prov_keys)

    # every CANDLES key is a provider "flag" column
    for c in CANDLES:
        assert COLUMNS[c].source == "provider"
        assert COLUMNS[c].kind == "flag"

def test_fetch_universe_payload_and_columns(monkeypatch):
    captured_payload = {}

    def mock_scan(market, query, timeout):
        captured_payload.update(query)
        # return a fake response with one row, populated with dummy float data mostly
        d = {k: 1.0 for k in query["columns"]}
        d["s"] = "NASDAQ:AAPL"
        return [{"s": d["s"], "d": list(d.values())[:-1]}] # -1 because 's' isn't part of 'd' typically... actually we can just return what TV returns.
        # TV returns {"s": "symbol", "d": [col1, col2, ...]}

    def mock_scan_accurate(market, query, timeout):
        captured_payload.update(query)
        d = [1.0] * len(query["columns"])
        return [{"s": "NASDAQ:AAPL", "d": d}]

    import screener.data.tradingview as tv
    monkeypatch.setattr(tv, "_scan", mock_scan_accurate)

    df = tv.fetch_universe()

    # assert payload["columns"] equals the snapshot list (same order)
    assert captured_payload["columns"] == FIELDS_SNAPSHOT

    # assert the returned DataFrame's columns equal the canonical provider columns (the set main returned)
    # The set main returned is the provider columns + tv_symbol
    expected_cols = ["tv_symbol"] + provider_columns()
    # Wait, fetch_universe clean() strips EXTRA_TV_FIELDS and is_primary, etc
    # Let's check exactly what columns are left
    assert set(df.columns) == set(expected_cols)
