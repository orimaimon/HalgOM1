"""Screener tests — run offline on synthetic data (no network)."""
import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from screener import export
from screener import presets as P
from screener.data.store import Store, load_universe
from screener.data.tradingview import clean
from screener.derive import add_derived
from screener.fields import COLUMNS, VIEWS, tv_fields
from screener.filters import ANY, FILTERS, FILTERS_BY_KEY, apply_filters, option_labels, search
from screener.market import compute_regime
from screener.scores import add_scores

TODAY = dt.date(2026, 10, 18)


def make_raw(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for key, col in COLUMNS.items():
        if not col.tv:
            continue
        if col.kind == "text":
            continue
        if col.kind == "date":
            data[key] = pd.Timestamp(TODAY) + pd.to_timedelta(rng.integers(-20, 90, n), unit="D")
        else:
            data[key] = rng.normal(10, 30, n)
    df = pd.DataFrame(data)
    df["ticker"] = [f"T{i:03d}" for i in range(n)]
    df["company"] = [f"Company {i}" for i in range(n)]
    df["exchange"] = rng.choice(["NASDAQ", "NYSE", "AMEX"], n)
    df["sector"] = rng.choice(["Technology Services", "Finance", "Health Technology"], n)
    df["industry"] = rng.choice(["Software", "Banks", "Biotech", "Semis"], n)
    df["country"] = rng.choice(["United States", "Israel", "Canada"], n, p=[0.8, 0.1, 0.1])
    df["sec_type"] = "stock"
    df["tv_symbol"] = df["exchange"] + ":" + df["ticker"]
    df["price"] = rng.uniform(1, 400, n)
    df["market_cap"] = 10 ** rng.uniform(7, 12.5, n)
    for c in ("volume", "avg_volume", "float_shares", "shares_out"):
        df[c] = 10 ** rng.uniform(4, 8, n)
    df["high52"] = df["price"] * rng.uniform(1.0, 1.8, n)
    df["low52"] = df["price"] * rng.uniform(0.4, 1.0, n)
    for c in ("sma20", "sma50", "sma200"):
        df[c] = df["price"] * rng.uniform(0.7, 1.3, n)
    df["rsi"] = rng.uniform(5, 95, n)
    df.loc[::17, "pe"] = np.nan          # missing fundamentals are common
    df.loc[::5, "pe"] = -abs(df.loc[::5, "pe"])
    return df


@pytest.fixture(scope="module")
def df():
    return add_derived(make_raw(), today=TODAY)


# ── registry ────────────────────────────────────────────────────────────────

def test_tv_fields_unique_and_views_valid():
    assert len(tv_fields()) > 50
    for name, cols in VIEWS.items():
        assert cols[0] == "ticker", name
        assert all(c in COLUMNS for c in cols), name


def test_filters_reference_known_columns():
    for f in FILTERS:
        assert f.column in COLUMNS, f.key
        assert f.tooltip, f"{f.key} has no tooltip"
        labels = [o.label for o in f.options]
        assert len(labels) == len(set(labels)), f"duplicate option label in {f.key}"


# ── filters ─────────────────────────────────────────────────────────────────

def test_every_option_returns_valid_mask(df):
    for f in FILTERS:
        for label in option_labels(f, df)[1:]:
            out = apply_filters(df, {f.key: label})
            assert len(out) <= len(df), (f.key, label)


def test_any_and_unknown_selections_are_ignored(df):
    assert len(apply_filters(df, {"pe": ANY, "price": ""})) == len(df)
    assert len(apply_filters(df, {"nope": "Over 5", "pe": "Not an option"})) == len(df)


def test_numeric_semantics_and_nan_never_passes(df):
    out = apply_filters(df, {"price": "Over $10", "pe": "Under 15"})
    assert (out["price"] > 10).all()
    assert out["pe"].between(0, 15, inclusive="neither").all()   # negative P/E is not "cheap"
    assert out["pe"].notna().all()


def test_combined_column_option(df):
    out = apply_filters(df, {"sma50": "SMA50 above SMA200"})
    assert (out["sma50"] > out["sma200"]).all()


def test_dynamic_country_options(df):
    labels = option_labels(FILTERS_BY_KEY["country"], df)
    assert labels[:3] == [ANY, "USA", "Foreign (ex-USA)"]
    assert (apply_filters(df, {"country": "USA"})["country"] == "United States").all()
    assert (apply_filters(df, {"country": "Foreign (ex-USA)"})["country"] != "United States").all()


def test_search(df):
    assert search(df, "T001, T002")["ticker"].tolist() == ["T001", "T002"]
    assert search(df, "AAA BBB").empty                       # all-caps words = ticker list
    assert len(search(df, "company 1")) >= 1                 # free text = substring
    assert len(search(df, "")) == len(df)


# ── presets ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(P.BUILTIN))
def test_builtin_presets_use_valid_options(name, df):
    p = P.BUILTIN[name]
    assert p.view in VIEWS
    assert p.sort is None or p.sort in COLUMNS
    for key, label in p.filters.items():
        f = FILTERS_BY_KEY[key]
        assert f.dynamic or label in option_labels(f), (name, key, label)


def test_user_presets_roundtrip(tmp_path):
    path = tmp_path / "presets.json"
    P.save_user("מסך שלי", P.Preset({"pe": "Under 15"}, view="Valuation"), path)
    assert P.load_user(path)["מסך שלי"].filters == {"pe": "Under 15"}
    P.delete_user("מסך שלי", path)
    assert P.load_user(path) == {}


# ── derive / scores ─────────────────────────────────────────────────────────

def test_derived_columns(df):
    row = df.iloc[0]
    assert row["sma50_pct"] == pytest.approx((row["price"] / row["sma50"] - 1) * 100)
    assert df["range52"].dropna().between(0, 1).all()
    assert (df["high52_pct"].dropna() <= 0).all()
    assert (df["days_to_earnings"].dropna() >= 0).all()
    assert df.loc[df["earnings_date"] < pd.Timestamp(TODAY), "days_to_earnings"].isna().all()


def test_scores_rank_negative_pe_worst():
    d = pd.DataFrame({"pe": [5, 10, 50, -3], "ps": [1, 2, 5, 1], "pb": [1, 2, 5, 1],
                      "ev_ebitda": [5, 8, 30, -2], "pfcf": [5, 10, 40, -1]})
    s = add_scores(d)
    assert s["score_value"].iloc[0] == s["score_value"].max()
    assert s["score_value"].iloc[3] < s["score_value"].iloc[1]


def test_scores_in_range(df):
    for c in ("score", "score_value", "score_quality", "score_momentum"):
        assert df[c].dropna().between(0, 100).all()


# ── market regime (same rules as DB/build_simulation_db.py) ─────────────────

def _market(vix, spy, spy200, hyg, hyg20):
    return pd.DataFrame({"close": [spy, 1, hyg, vix], "change": 0.0, "sma20": [1, 1, hyg20, 1],
                         "sma50": 1.0, "sma200": [spy200, 1, 1, 1]}, index=["SPY", "QQQ", "HYG", "VIX"])


@pytest.mark.parametrize("args,expected", [
    ((15, 110, 100, 80, 79), "bull"),
    ((30, 90, 100, 80, 79), "bear"),
    ((20, 110, 100, 78, 79), "sideways"),
    ((17, 90, 100, 78, 79), "bear"),
    ((float("nan"), 110, 100, 80, 79), "unknown"),
])
def test_regime(args, expected):
    assert compute_regime(_market(*args)).label == expected


# ── cleaning ────────────────────────────────────────────────────────────────

def test_clean_drops_junk():
    raw = make_raw(5)
    raw["is_primary"] = [True, True, False, True, True]
    raw["currency"] = ["USD", "USD", "USD", "CAD", "USD"]
    raw["typespecs"] = [["common"]] * 5
    raw.loc[4, "ticker"] = raw.loc[0, "ticker"]   # duplicate ticker
    raw.loc[4, "market_cap"] = raw.loc[0, "market_cap"] * 2
    raw["earnings_date"] = 1795608000
    out = clean(raw)
    assert len(out) == 2
    assert out["ticker"].is_unique
    assert "is_primary" not in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["earnings_date"])


# ── store / offline fallback ────────────────────────────────────────────────

def test_store_cache_then_stale_fallback(tmp_path):
    store = Store(tmp_path)
    raw = make_raw(20)
    snap = load_universe(store, lambda: raw, dt.timedelta(minutes=15))
    assert snap.source == "live" and len(store.snapshot_dates()) == 1

    def boom():
        raise RuntimeError("offline")

    assert load_universe(store, boom, dt.timedelta(minutes=15)).source == "cache"
    stale = load_universe(store, boom, dt.timedelta(minutes=15), force=True)
    assert stale.source == "stale" and "offline" in stale.error and len(stale.df) == 20


def test_store_raises_without_cache(tmp_path):
    def boom():
        raise RuntimeError("offline")

    with pytest.raises(RuntimeError):
        load_universe(Store(tmp_path), boom, dt.timedelta(minutes=15))


# ── export ──────────────────────────────────────────────────────────────────

def test_excel_export(df):
    cols = VIEWS["Valuation"]
    data = export.to_excel(df.head(10), cols, [("P/E", "Under 15")], dt.datetime(2026, 10, 18, 15, 30), "t")
    wb = load_workbook(io.BytesIO(data))
    ws = wb["Results"]
    headers = [c.value for c in ws[1]]
    assert headers[: len(cols)] == [COLUMNS[c].label for c in cols]
    assert headers[-3:] == export.JOURNAL_COLUMNS
    mcap = ws.cell(row=2, column=cols.index("market_cap") + 1)
    assert isinstance(mcap.value, float) and ',,,"B"' in mcap.number_format   # numeric, shown as B
    assert ws.cell(row=2, column=1).hyperlink.target.startswith("https://www.tradingview.com/symbols/")
    assert wb["Filters"]["A8"].value == "P/E"


def test_excel_billion_format_scales_correctly():
    # each trailing comma divides by 1,000 → billions need three
    assert export.EXCEL_FORMATS["money"].startswith('[>=1000000000]$0.00,,,"B"')


def test_watchlist_and_csv(df):
    wl = export.to_watchlist(df.head(12))
    assert wl.startswith("###")
    assert all(p.startswith("###") or ":" in p for p in wl.split(","))
    assert export.to_csv(df.head(3), ["ticker", "price"]).startswith(b"\xef\xbb\xbfTicker,Price")
