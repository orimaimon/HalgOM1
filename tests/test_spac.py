import numpy as np
import pandas as pd

from screener.filters import JUNK_RULES, hide_junk, junk_mask
from screener.i18n import UI


def test_spac_identification():
    df = pd.DataFrame(
        {
            "ticker": ["FOO", "BAR", "BAZ", "QUX", "ACM", "BIG", "NAN"],
            "company": [
                "Foo Acquisition Corp",
                "Bar Acquisition Corp II",
                "Baz Merger Corp",
                "Qux Blank Check Co",
                "Acme Corp",
                "Big Acquisition Corp",
                np.nan,
            ],
            "industry": [
                "Software",
                "Financial Conglomerates",
                "Software",
                "Software",
                "Software",
                "Software",
                "Software",
            ],
            "price": [10.05, 45.0, 10.40, 9.90, 10.0, 30.0, np.nan],
            "avg_volume": [100e3] * 7,
            "market_cap": [100e6] * 7,
        }
    )

    mask = junk_mask(df)
    # Junk rows: FOO (0), BAR (1), BAZ (2), QUX (3)
    # Non-junk rows: ACM (4), BIG (5), NAN (6)
    assert list(mask) == [True, True, True, True, False, False, False]

    cleaned = hide_junk(df, on=True)
    assert list(cleaned["ticker"]) == ["ACM", "BIG", "NAN"]


def test_spac_i18n():
    spac_rule_label = JUNK_RULES[3][0]
    assert spac_rule_label == "SPAC (blank-check company)"
    assert spac_rule_label in UI
    assert UI[spac_rule_label] == "חברת רכישה ייעודית (SPAC)"
