"""Tests for the Shares Outstanding filter."""
import numpy as np
import pandas as pd
import pytest

from screener import filters, i18n


def test_shares_filter_semantics():
    df = pd.DataFrame({
        "ticker": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
        "price": [10.0] * 7,
        "avg_volume": [100e3] * 7,
        "market_cap": [100e6] * 7,
        "shares_out": [10e6, 49e6, 50e6, 200e6, np.nan, 2e9, 1e9],
    })

    under_50m = filters.apply_filters(df, {"shares_out": "Under 50M"})
    assert sorted(under_50m["ticker"].tolist()) == ["T1", "T2"]

    over_1b = filters.apply_filters(df, {"shares_out": "Over 1B"})
    assert sorted(over_1b["ticker"].tolist()) == ["T6"]


def test_shares_filter_group_position_and_i18n():
    f = filters.FILTERS_BY_KEY.get("shares_out")
    assert f is not None, "shares_out filter must exist"
    assert f.group == "Descriptive"

    # Verify position right after float_shares in FILTERS
    keys = [filter_obj.key for filter_obj in filters.FILTERS]
    float_idx = keys.index("float_shares")
    shares_idx = keys.index("shares_out")
    assert shares_idx == float_idx + 1

    # Verify filter label and option labels translate to Hebrew
    translated_label = i18n.filter_label(f.key, f.label, i18n.HE)
    assert translated_label == "מניות מונפקות"

    for label in filters.option_labels(f):
        he_opt = i18n.option(label, i18n.HE)
        assert he_opt != label or label == "Any"


def test_shares_filter_screen_saved_selection():
    df = pd.DataFrame({
        "ticker": ["T1", "T2", "T3"],
        "price": [10.0] * 3,
        "avg_volume": [100e3] * 3,
        "market_cap": [100e6] * 3,
        "shares_out": [50e6, 150e6, np.nan],
    })

    res = filters.screen(df, {"shares_out": "Over 100M"}, junk=False)
    assert res["ticker"].tolist() == ["T2"]
