"""
Tests for screener/compare.py (stock comparison engine).
"""
import pandas as pd
import pytest

from screener import compare, i18n


def make_synthetic_universe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker": "AAPL",
            "market_cap": 2.1e9,
            "price": 150.0,
            "change": 1.5,
            "pe": 25.0,
            "fwd_pe": 20.0,
            "ps": 6.0,
            "peg": 1.2,
            "sales_growth": 12.3,
            "eps_growth": 10.0,
            "gross_margin": 45.0,
            "net_margin": 20.0,
            "roe": 30.0,
            "debt_eq": 1.5,
            "div_yield": 0.5,
            "perf_y": 15.0,
            "sma200_pct": 5.0,
            "rsi": 55.0,
            "beta": 1.1,
            "score": 85.0,
        },
        {
            "ticker": "MSFT",
            "market_cap": 3.0e9,
            "price": 300.0,
            "change": -0.5,
            "pe": 35.0,
            "fwd_pe": 28.0,
            "ps": 10.0,
            "peg": 2.0,
            "sales_growth": 18.0,
            "eps_growth": 15.0,
            "gross_margin": 68.0,
            "net_margin": 35.0,
            "roe": 40.0,
            "debt_eq": 0.5,
            "div_yield": 0.8,
            "perf_y": 25.0,
            "sma200_pct": 12.0,
            "rsi": 62.0,
            "beta": 0.9,
            "score": 92.0,
        },
        {
            "ticker": "GOOGL",
            "market_cap": 1.8e9,
            "price": 120.0,
            "change": 0.0,
            "pe": -15.0,  # negative PE
            "fwd_pe": 18.0,
            "ps": 5.0,
            "peg": 0.9,
            "sales_growth": 12.3,  # tie with AAPL
            "eps_growth": float("nan"),  # missing
            "gross_margin": 55.0,
            "net_margin": 22.0,
            "roe": 25.0,
            "debt_eq": 0.2,
            "div_yield": 0.0,
            "perf_y": 10.0,
            "sma200_pct": 2.0,
            "rsi": 48.0,
            "beta": 1.0,
            "score": 78.0,
        },
        {
            "ticker": "AMZN",
            "market_cap": 1.5e9,
            "price": 130.0,
            "change": 2.0,
            "pe": 50.0,
            "fwd_pe": 40.0,
            "ps": 3.0,
            "peg": 1.5,
            "sales_growth": 15.0,
            "eps_growth": 20.0,
            "gross_margin": 40.0,
            "net_margin": 8.0,
            "roe": 15.0,
            "debt_eq": 0.8,
            "div_yield": 0.0,
            "perf_y": 30.0,
            "sma200_pct": 8.0,
            "rsi": 58.0,
            "beta": 1.2,
            "score": 80.0,
        },
    ])


def test_three_tickers_table_structure():
    universe = make_synthetic_universe()
    tickers = ["AAPL", "MSFT", "GOOGL"]
    df = compare.compare(universe, tickers)

    assert list(df.columns) == ["metric", "AAPL", "MSFT", "GOOGL"]
    assert list(df.index) == [m for m, _ in compare.METRICS]

    # Check formatted values
    assert df.loc["sales_growth", "AAPL"] == "12.3%"
    assert df.loc["market_cap", "AAPL"] == "$2.1B"
    assert df.loc["eps_growth", "GOOGL"] == "—"  # NaN -> "—"

    # Check Hebrew metric names
    assert df.loc["pe", "metric"] == i18n.column("pe", "P/E", i18n.HE)
    assert df.loc["roe", "metric"] == i18n.column("roe", "ROE", i18n.HE)


def test_best_selection_rules():
    universe = make_synthetic_universe()
    tickers = ["AAPL", "MSFT", "GOOGL"]
    df = compare.compare(universe, tickers)
    best = df.attrs["best"]

    # Lowest positive P/E wins (AAPL=25.0, MSFT=35.0, GOOGL=-15.0 ignored)
    assert best["pe"] == "AAPL"

    # Highest ROE wins (MSFT=40.0 > AAPL=30.0 > GOOGL=25.0)
    assert best["roe"] == "MSFT"

    # Metric with direction "" has no best
    assert "market_cap" not in best
    assert "price" not in best
    assert "rsi" not in best

    # A tie has no best (sales_growth: AAPL=12.3, GOOGL=12.3, MSFT=18.0 -> MSFT wins; let's check custom tie)
    assert best["sales_growth"] == "MSFT"

    # Custom test for tie & single value
    custom_metrics = [
        ("sales_growth", "high"),
        ("eps_growth", "high"),
        ("pe", "low"),
    ]
    # In custom subset:
    # AAPL(12.3) vs GOOGL(12.3) -> tie for max if no other competing, or if tied for max
    df_tie = compare.compare(universe, ["AAPL", "GOOGL"], metrics=custom_metrics)
    assert "sales_growth" not in df_tie.attrs["best"]

    # Single value competing: eps_growth has AAPL=10.0, GOOGL=NaN -> single value competing -> no best
    assert "eps_growth" not in df_tie.attrs["best"]


def test_missing_ticker_duplicates_and_limit():
    universe = make_synthetic_universe()

    # Missing ticker in attrs["missing"], not in columns
    df = compare.compare(universe, ["aapl", "INVALID", "msft", "AAPL"])
    assert list(df.columns) == ["metric", "AAPL", "MSFT"]
    assert df.attrs["missing"] == ["INVALID"]

    # 5 tickers -> ValueError with Hebrew message
    with pytest.raises(ValueError, match="ניתן להשוות עד 4 מניות בלבד"):
        compare.compare(universe, ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])


def test_summary_line():
    universe = make_synthetic_universe()
    tickers = ["AAPL", "MSFT", "GOOGL"]

    summary = compare.summary_line(universe, tickers)
    df = compare.compare(universe, tickers)
    best = df.attrs["best"]

    # Count wins from best
    counts = {}
    for t in best.values():
        counts[t] = counts.get(t, 0) + 1

    assert "MSFT" in summary
    assert "AAPL" in summary
    assert " מובילה ב-" in summary
    assert "nan" not in summary.lower()
    assert "none" not in summary.lower()

    # Verify counts in summary line match
    for t, cnt in counts.items():
        assert f"{t}" in summary
        assert str(cnt) in summary

    # Empty summary line when nothing competes
    df_empty = compare.summary_line(universe, ["INVALID_1", "INVALID_2"])
    assert df_empty == ""


def test_negative_debt_to_equity_never_wins():
    import pandas as pd
    from screener import compare
    uni = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "debt_eq": [-2.0, 0.4, 0.9]})
    df = compare.compare(uni, ["AAA", "BBB", "CCC"], metrics=[("debt_eq", "low")])
    assert df.attrs["best"] == {"debt_eq": "BBB"}


def test_compare_stocks_two_and_three_tickers():
    universe = make_synthetic_universe()

    # 2 tickers
    res2 = compare.compare_stocks(["AAPL", "MSFT"], universe)
    assert set(res2.keys()) == {"metrics_table", "highlights"}
    assert list(res2["metrics_table"].columns) == ["metric", "AAPL", "MSFT"]
    assert "Lowest P/E" in res2["highlights"]["AAPL"]
    assert "Highest ROE" in res2["highlights"]["MSFT"]
    assert "Highest 1Y Return" in res2["highlights"]["MSFT"]
    assert "Lowest Debt" in res2["highlights"]["MSFT"]

    # 3 tickers
    res3 = compare.compare_stocks(["AAPL", "MSFT", "GOOGL"], universe)
    assert set(res3.keys()) == {"metrics_table", "highlights"}
    assert list(res3["metrics_table"].columns) == ["metric", "AAPL", "MSFT", "GOOGL"]
    assert "Lowest P/E" in res3["highlights"]["AAPL"]
    assert "Highest ROE" in res3["highlights"]["MSFT"]
    assert "Lowest Debt" in res3["highlights"]["GOOGL"]


def test_compare_stocks_five_tickers_raises():
    universe = make_synthetic_universe()
    with pytest.raises(ValueError, match="ניתן להשוות עד 4 מניות בלבד"):
        compare.compare_stocks(["AAPL", "MSFT", "GOOGL", "AMZN", "META"], universe)


def test_compare_stocks_missing_tickers():
    universe = make_synthetic_universe()
    res = compare.compare_stocks(["AAPL", "NONEXISTENT"], universe)
    assert list(res["metrics_table"].columns) == ["metric", "AAPL"]
    assert res["metrics_table"].attrs["missing"] == ["NONEXISTENT"]
    assert "AAPL" in res["highlights"]
    assert "NONEXISTENT" not in res["highlights"]


def test_compare_stocks_ties():
    universe = make_synthetic_universe()
    # AAPL and GOOGL both have sales_growth = 12.3 in synthetic universe
    res = compare.compare_stocks(["AAPL", "GOOGL"], universe)
    # Since sales_growth tied, neither gets "Highest Sales Growth"
    assert "Highest Sales Growth" not in res["highlights"]["AAPL"]
    assert "Highest Sales Growth" not in res["highlights"]["GOOGL"]
