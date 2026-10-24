"""Tests for portfolio correlation heatmap and market stress-test UI component (screener/ui/correlation_view.py)."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from screener.portfolio import PORTFOLIO, WATCHLIST, PList
from screener.ui import correlation_view


@pytest.fixture
def mock_universe():
    return pd.DataFrame([
        {"ticker": "NVDA", "company": "NVIDIA Corp", "sector": "Technology", "price": 100.0, "beta": 1.5},
        {"ticker": "AMD", "company": "Advanced Micro Devices", "sector": "Technology", "price": 150.0, "beta": 1.2},
    ])


@pytest.fixture
def sample_portfolio():
    return PList(
        "Tech Fund",
        PORTFOLIO,
        [
            {"ticker": "NVDA", "quantity": 100.0, "price": 90.0, "date": "2024-01-01", "side": "buy"},
            {"ticker": "AMD", "quantity": 100.0, "price": 140.0, "date": "2024-01-01", "side": "buy"},
        ],
    )


@pytest.fixture
def sample_watchlist():
    return PList(
        "Watchlist",
        WATCHLIST,
        [
            {"ticker": "NVDA"},
            {"ticker": "AMD"},
        ],
    )


def test_correlation_view_correlated_positions(sample_portfolio, mock_universe):
    """Render with 2 open positions (NVDA, AMD) and correlated prices (corr > 0.90).

    Should render cr_stress_table, cr_heatmap, and emit the Hebrew high-correlation warning.
    """
    np.random.seed(42)
    days = 60
    base_returns = np.random.normal(0.001, 0.02, days)
    nvda_prices = 100.0 * np.exp(np.cumsum(base_returns + np.random.normal(0, 0.001, days)))
    amd_prices = 150.0 * np.exp(np.cumsum(base_returns + np.random.normal(0, 0.001, days)))

    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    nvda_df = pd.DataFrame({"adj_close": nvda_prices, "close": nvda_prices}, index=dates)
    amd_df = pd.DataFrame({"adj_close": amd_prices, "close": amd_prices}, index=dates)

    def mock_prices_history(ticker):
        if ticker == "NVDA":
            return nvda_df
        if ticker == "AMD":
            return amd_df
        return pd.DataFrame()

    with patch("screener.data.prices.history", side_effect=mock_prices_history), \
         patch("streamlit.subheader") as mock_subheader, \
         patch("streamlit.dataframe") as mock_dataframe, \
         patch("streamlit.warning") as mock_warning, \
         patch("streamlit.plotly_chart") as mock_plotly:

        correlation_view.render(sample_portfolio, mock_universe)

        # Check subheaders rendered
        assert mock_subheader.call_count == 2

        # Check cr_stress_table rendered
        mock_dataframe.assert_called_once()
        _, df_kwargs = mock_dataframe.call_args
        assert df_kwargs.get("key") == "cr_stress_table"

        # Check high-correlation warning emitted
        mock_warning.assert_called_once()
        warning_msg = mock_warning.call_args[0][0]
        assert "NVDA ו-AMD זזות יחד" in warning_msg or "AMD ו-NVDA זזות יחד" in warning_msg

        # Check cr_heatmap rendered
        mock_plotly.assert_called_once()
        _, plotly_kwargs = mock_plotly.call_args
        assert plotly_kwargs.get("key") == "cr_heatmap"


def test_correlation_view_empty_price_history(sample_portfolio, mock_universe):
    """Render when prices.history returns empty frames.

    Renders cr_stress_table and shows t("Not enough price history for correlation matrix").
    """
    with patch("screener.data.prices.history", return_value=pd.DataFrame()), \
         patch("streamlit.dataframe") as mock_dataframe, \
         patch("streamlit.plotly_chart") as mock_plotly, \
         patch("screener.ui.correlation_view.hebrew_note") as mock_hebrew_note:

        correlation_view.render(sample_portfolio, mock_universe)

        # cr_stress_table rendered
        mock_dataframe.assert_called_once()
        _, df_kwargs = mock_dataframe.call_args
        assert df_kwargs.get("key") == "cr_stress_table"

        # Heatmap not rendered
        mock_plotly.assert_not_called()

        # Not enough price history note rendered
        mock_hebrew_note.assert_called_once_with("אין מספיק היסטוריית מחירים שמורה לחישוב מתאם (נדרשות לפחות 2 מניות עם היסטוריה)")


def test_correlation_view_watchlist_no_output(sample_watchlist, mock_universe):
    """Watchlist produces no output and no error."""
    with patch("streamlit.dataframe") as mock_dataframe, \
         patch("streamlit.plotly_chart") as mock_plotly, \
         patch("screener.ui.correlation_view.hebrew_note") as mock_hebrew_note:

        correlation_view.render(sample_watchlist, mock_universe)

        mock_dataframe.assert_not_called()
        mock_plotly.assert_not_called()
        mock_hebrew_note.assert_not_called()


def test_correlation_view_error_path(sample_portfolio, mock_universe, caplog):
    """Error path logs exception and renders hebrew_note(t("Error displaying correlation view"))."""
    with patch("screener.portfolio_risk.exposure", side_effect=RuntimeError("Test error")), \
         patch("screener.ui.correlation_view.hebrew_note") as mock_hebrew_note:

        with caplog.at_level(logging.ERROR, logger="screener.app"):
            correlation_view.render(sample_portfolio, mock_universe)

        mock_hebrew_note.assert_called_once_with("שגיאה בהצגת ניתוח המתאם והתרחישים")
        assert "Error rendering correlation view" in caplog.text
