import numpy as np
import pandas as pd
import pytest

from screener.fields import CANDLES
from screener.signals import CONSOLIDATION, detect_vcp, summarize


def test_detect_vcp_synthetic_valid():
    """Test detection on synthetic rows meeting VCP criteria."""
    # Standard pct fields
    row_pct = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "price": 100.0,
        "high52": 105.0,  # 100 / 105 = 95.2% (within 15% of 52W high)
        "sma50": 90.0,    # 100 > 90 (above SMA50)
    })
    assert detect_vcp(row_pct) is True

    # 5D/10D alias fields
    row_aliases = pd.Series({
        "range_5d_pct": 2.5,
        "range_10d_pct": 5.0,
        "range_1m_pct": 10.0,
        "price": 50.0,
        "high52": 52.0,
        "sma50": 45.0,
    })
    assert detect_vcp(row_aliases) is True

    # Using pct distance fields (high52_pct, sma50_pct)
    row_pct_dist = pd.Series({
        "range_1w": 4.0,
        "range_2w": 8.0,
        "range_1m": 15.0,
        "high52_pct": -5.0,  # 5% below high
        "sma50_pct": 10.0,   # 10% above SMA50
    })
    assert detect_vcp(row_pct_dist) is True


def test_detect_vcp_range_expansion():
    """Test non-detection when range expands or does not contract strictly."""
    # 1W range > 2W range
    row_exp1 = pd.Series({
        "range_1w_pct": 8.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "price": 100.0,
        "high52": 105.0,
        "sma50": 90.0,
    })
    assert detect_vcp(row_exp1) is False

    # 2W range > 1M range
    row_exp2 = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": 15.0,
        "range_1m_pct": 12.0,
        "price": 100.0,
        "high52": 105.0,
        "sma50": 90.0,
    })
    assert detect_vcp(row_exp2) is False

    # Equal ranges (1W == 2W)
    row_equal = pd.Series({
        "range_1w_pct": 6.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "price": 100.0,
        "high52": 105.0,
        "sma50": 90.0,
    })
    assert detect_vcp(row_equal) is False


def test_detect_vcp_downtrend_or_below_sma50():
    """Test non-detection when price is below SMA50."""
    row_below_sma = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "price": 80.0,
        "high52": 85.0,
        "sma50": 90.0,  # price (80) < sma50 (90)
    })
    assert detect_vcp(row_below_sma) is False

    row_sma_pct_neg = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "high52_pct": -2.0,
        "sma50_pct": -1.5,  # below SMA50
    })
    assert detect_vcp(row_sma_pct_neg) is False


def test_detect_vcp_far_from_52w_high():
    """Test non-detection when price is more than 15% below 52-week high."""
    row_far_high = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "price": 70.0,
        "high52": 100.0,  # price is 30% below 52W high (> 15%)
        "sma50": 60.0,
    })
    assert detect_vcp(row_far_high) is False

    row_high52_pct_low = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "high52_pct": -20.0,  # 20% below high
        "sma50_pct": 5.0,
    })
    assert detect_vcp(row_high52_pct_low) is False


def test_detect_vcp_missing_fields():
    """Test non-detection when key data fields are missing or NaN."""
    # Missing 1W range
    row_missing_r1 = pd.Series({
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "price": 100.0,
        "high52": 105.0,
        "sma50": 90.0,
    })
    assert detect_vcp(row_missing_r1) is False

    # NaN in 2W range
    row_nan_r2 = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": np.nan,
        "range_1m_pct": 12.0,
        "price": 100.0,
        "high52": 105.0,
        "sma50": 90.0,
    })
    assert detect_vcp(row_nan_r2) is False

    # Missing SMA50
    row_no_sma = pd.Series({
        "range_1w_pct": 3.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "price": 100.0,
        "high52": 105.0,
    })
    assert detect_vcp(row_no_sma) is False


def test_vcp_consolidation_signal_integration():
    """Test integration of VCP signal in CONSOLIDATION list and summarize()."""
    base_row = {k: 0 for k in CANDLES}
    base_row.update({
        "high52_prev": 110.0,
        "low52_prev": 50.0,
        "high_1m_prev": 110.0,
        "low_1m_prev": 50.0,
        "high_3m_prev": 110.0,
        "low_3m_prev": 50.0,
        "high_6m_prev": 110.0,
        "low_6m_prev": 50.0,
        "sma50_prev": 90.0,
        "sma200_prev": 80.0,
        "sma20_prev": 95.0,
        "sma200": 80.0,
        "sma20": 95.0,
        "close_prev": 100.0,
        "rel_volume": 1.0,
    })

    row0 = dict(base_row)
    row0.update({
        "ticker": "AAPL",
        "range_1w_pct": 3.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "atr_pct": 5.0,
        "price": 100.0,
        "high52": 105.0,
        "sma50": 90.0,
    })

    row1 = dict(base_row)
    row1.update({
        "ticker": "MSFT",
        "range_1w_pct": 10.0,
        "range_2w_pct": 6.0,
        "range_1m_pct": 12.0,
        "atr_pct": 2.0,
        "price": 100.0,
        "high52": 105.0,
        "sma50": 90.0,
    })

    df = pd.DataFrame([row0, row1])

    # Check VCP mask directly from CONSOLIDATION
    vcp_signal = CONSOLIDATION[0]
    vcp_mask = vcp_signal.fn(df)
    assert bool(vcp_mask.iloc[0]) is True
    assert bool(vcp_mask.iloc[1]) is False

    # Check summarize output
    signals_text, n_signals = summarize(df)
    assert "Tight 1M" in signals_text.iloc[0]
    assert n_signals.iloc[0] >= 1.0
