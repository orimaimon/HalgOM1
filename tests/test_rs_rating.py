"""
Unit tests for screener/rs_rating.py
"""
import time
import numpy as np
import pandas as pd
import pytest

from screener.rs_rating import add_rs_rating, rs_rating, rs_score


def test_ordering_of_known_stocks():
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "perf_q": [10.0, 20.0, 30.0, 40.0, 50.0],
        "perf_h": [5.0, 15.0, 25.0, 35.0, 45.0],
        "perf_y": [2.0, 12.0, 22.0, 32.0, 42.0],
        "perf_m": [1.0, 11.0, 21.0, 31.0, 41.0],
    })
    scores = rs_score(df)
    ratings = rs_rating(df)

    # Scores should strictly increase as performance increases
    assert list(scores) == sorted(scores)
    # Ratings should strictly increase as performance increases
    assert list(ratings) == sorted(ratings)
    # Highest stock (META) gets 99 rating
    assert ratings.iloc[-1] == 99.0
    # Lowest stock (AAPL) gets lowest rating
    assert ratings.iloc[0] < ratings.iloc[1]


def test_missing_column_renormalisation():
    # DataFrame missing 'perf_h'
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "perf_q": [10.0, 20.0],
        # perf_h missing
        "perf_y": [30.0, 40.0],
        "perf_m": [5.0, 15.0],
    })
    scores = rs_score(df)
    # Available weights sum = 0.4 + 0.2 + 0.2 = 0.8
    # Stock A: (0.4*10 + 0.2*30 + 0.2*5) / 0.8 = (4 + 6 + 1) / 0.8 = 11 / 0.8 = 13.75
    # Stock B: (0.4*20 + 0.2*40 + 0.2*15) / 0.8 = (8 + 8 + 3) / 0.8 = 19 / 0.8 = 23.75
    assert pytest.approx(scores.iloc[0]) == 13.75
    assert pytest.approx(scores.iloc[1]) == 23.75


def test_nan_row_stays_nan():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "perf_q": [10.0, np.nan],
        "perf_h": [20.0, np.nan],
        "perf_y": [30.0, np.nan],
        "perf_m": [40.0, np.nan],
    })
    scores = rs_score(df)
    ratings = rs_rating(df)

    assert not pd.isna(scores.iloc[0])
    assert pd.isna(scores.iloc[1])

    assert ratings.iloc[0] == 99.0
    assert pd.isna(ratings.iloc[1])


def test_ties_share_average_rank():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "perf_q": [10.0, 20.0, 20.0],
        "perf_h": [10.0, 20.0, 20.0],
        "perf_y": [10.0, 20.0, 20.0],
        "perf_m": [10.0, 20.0, 20.0],
    })
    ratings = rs_rating(df)
    # Stock B and C have identical scores and should have identical ratings
    assert ratings.iloc[1] == ratings.iloc[2]
    assert ratings.iloc[0] < ratings.iloc[1]


def test_single_row_frame():
    # Valid row yields 99
    df_valid = pd.DataFrame({"ticker": ["A"], "perf_q": [10.0]})
    rating_valid = rs_rating(df_valid)
    assert rating_valid.iloc[0] == 99.0

    # NaN row yields NaN
    df_nan = pd.DataFrame({"ticker": ["A"], "perf_q": [np.nan]})
    rating_nan = rs_rating(df_nan)
    assert pd.isna(rating_nan.iloc[0])


def test_input_not_mutated():
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "perf_q": [10.0, 20.0],
        "perf_h": [5.0, 15.0],
    })
    df_copy = df.copy()

    _ = rs_score(df)
    _ = rs_rating(df)
    df_out = add_rs_rating(df)

    # df must match original copy
    pd.testing.assert_frame_equal(df, df_copy)
    # Output df has new rs_rating column
    assert "rs_rating" in df_out.columns
    assert "rs_rating" not in df.columns


def test_empty_frame():
    df_empty = pd.DataFrame()

    scores = rs_score(df_empty)
    assert scores.empty

    ratings = rs_rating(df_empty)
    assert ratings.empty

    df_out = add_rs_rating(df_empty)
    assert "rs_rating" in df_out.columns
    assert len(df_out) == 0


def test_performance_6000_rows_under_200ms():
    np.random.seed(42)
    n = 6000
    df = pd.DataFrame({
        "ticker": [f"TICK{i}" for i in range(n)],
        "perf_q": np.random.randn(n) * 10,
        "perf_h": np.random.randn(n) * 15,
        "perf_y": np.random.randn(n) * 20,
        "perf_m": np.random.randn(n) * 5,
    })

    t0 = time.perf_counter()
    df_res = add_rs_rating(df)
    elapsed = time.perf_counter() - t0

    assert len(df_res) == n
    assert "rs_rating" in df_res.columns
    # Check execution time is well under 0.2s (200ms)
    assert elapsed < 0.2, f"RS rating on {n} rows took {elapsed:.4f}s (expected < 0.2s)"
