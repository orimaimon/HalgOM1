"""The Download page's logic (screener/data/download.py) and the page itself. Offline: yf.download is faked."""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import pytest

from screener.data import download as DL
from tests.test_frontend import app  # noqa: F401  (fixture)

FIELDS = ["Adj Close", "Close", "High", "Low", "Open", "Volume", "Dividends", "Stock Splits"]
TODAY = dt.date(2026, 10, 22)


def fake_download(tickers, group_by="column", tz=None, fail=(), days=5, **_):
    """yfinance's shape: MultiIndex columns named (Price, Ticker), or (Ticker, Price) with group_by=ticker."""
    idx = pd.date_range("2026-09-01", periods=days, freq="B", name="Date", tz=tz)
    cols = {}
    for i, t in enumerate(tickers):
        for f in FIELDS:
            v = 100.0 * (i + 1) if f != "Volume" else 1000.0
            cols[(f, t)] = [float("nan") if t in fail else 0.0 if f in ("Dividends", "Stock Splits") else v + d
                            for d in range(days)]
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Price", "Ticker"])
    if group_by == "ticker":
        df = df.swaplevel(axis=1).sort_index(axis=1)
    return df


@pytest.fixture
def fake_yf(monkeypatch):
    calls = []

    def dl(**kw):
        calls.append(kw)
        DL.yf.shared._ERRORS = {"NOSUCH": "YFPricesMissingError('possibly delisted; no price data found')"}
        return fake_download(kw["tickers"], kw.get("group_by", "column"), fail=("NOSUCH",))

    monkeypatch.setattr(DL.yf, "download", dl)
    return calls


def test_parse_tickers():
    assert DL.parse_tickers("aapl, brk.b\nTEVA.TA;^gspc  aapl BF.B vod.l") == \
        ["AAPL", "BRK-B", "TEVA.TA", "^GSPC", "BF-B", "VOD.L"]
    assert DL.parse_tickers("  ") == []


def test_validate():
    ok = DL.Request(["AAPL"], start=dt.date(2025, 1, 1), end=dt.date(2025, 2, 1))
    assert DL.validate(ok, TODAY) == []
    assert DL.validate(DL.Request([]), TODAY) == ["Enter at least one ticker", "Choose a start date"]
    bad = DL.Request(["AAPL"], start=dt.date(2025, 2, 1), end=dt.date(2025, 1, 1))
    assert "Start date must be before end date" in DL.validate(bad, TODAY)
    # intraday history is limited on Yahoo
    assert DL.validate(DL.Request(["AAPL"], use_period=True, period="1y", interval="1m"), TODAY)
    assert DL.validate(DL.Request(["AAPL"], use_period=True, period="5d", interval="1m"), TODAY) == []
    assert DL.validate(DL.Request(["AAPL"], start=TODAY - dt.timedelta(days=90), interval="5m"), TODAY)
    assert DL.validate(DL.Request(["AAPL"], use_period=True, period="1y", interval="1h"), TODAY) == []
    assert DL.validate(DL.Request(["A"] * (DL.MAX_TICKERS + 1), use_period=True), TODAY)


def test_kwargs_period_or_dates():
    r = DL.Request(["AAPL"], use_period=True, period="ytd", start=dt.date(2020, 1, 1))
    kw = r.kwargs()
    assert kw["period"] == "ytd" and "start" not in kw and kw["progress"] is False
    r = DL.Request(["AAPL"], start=dt.date(2020, 1, 1), end=dt.date(2020, 2, 1), repair=True, group_by="ticker")
    kw = r.kwargs()
    assert kw["start"] == "2020-01-01" and kw["end"] == "2020-02-01" and "period" not in kw
    assert kw["repair"] is True and kw["group_by"] == "ticker"


@pytest.mark.parametrize("group_by", ["column", "ticker"])
def test_fetch_and_shapes(fake_yf, group_by):
    req = DL.Request(["AAPL", "MSFT", "NOSUCH"], use_period=True, group_by=group_by)
    res = DL.fetch(req)
    assert fake_yf[0]["tickers"] == ["AAPL", "MSFT", "NOSUCH"]
    assert res.ok == ["AAPL", "MSFT"] and list(res.failed) == ["NOSUCH"]
    assert "possibly delisted" in res.failed["NOSUCH"]

    long = DL.to_long(res)
    assert list(long.columns[:8]) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert len(long) == 10 and set(long["Ticker"]) == {"AAPL", "MSFT"}

    wide = DL.to_wide(res)
    first = "Open AAPL" if group_by == "column" else "AAPL Open"
    assert wide.columns[0] == first and len(wide) == 5 and len(wide.columns) == 16

    frames = DL.per_ticker(res)
    assert list(frames) == ["AAPL", "MSFT"] and frames["MSFT"].index.name == "Date"
    summ = DL.summary(res)
    assert list(summ["Rows"]) == [5, 5]


def test_single_ticker_flat_columns(monkeypatch):
    flat = fake_download(["NVDA"]).droplevel("Ticker", axis=1)
    monkeypatch.setattr(DL.yf, "download", lambda **kw: flat)
    res = DL.fetch(DL.Request(["NVDA"], use_period=True, multi_level_index=False))
    assert res.failed == {} and len(DL.to_long(res)) == 5


def test_nothing_downloaded(monkeypatch):
    monkeypatch.setattr(DL.yf, "download", lambda **kw: pd.DataFrame())
    res = DL.fetch(DL.Request(["AAPL"], use_period=True))
    assert list(res.failed) == ["AAPL"] and DL.to_long(res).empty


def test_intraday_utc_is_labelled(monkeypatch):
    monkeypatch.setattr(DL.yf, "download", lambda **kw: fake_download(kw["tickers"], tz="UTC"))
    res = DL.fetch(DL.Request(["AAPL", "SPY"], use_period=True, period="5d", interval="15m"))
    long = DL.to_long(res)
    assert long.columns[0] == "Datetime (UTC)" and long.iloc[:, 0].dt.tz is None   # Excel needs naive


@pytest.mark.parametrize("layout", ["sheets", "long", "wide"])
def test_excel(fake_yf, layout):
    res = DL.fetch(DL.Request(["AAPL", "MSFT", "NOSUCH"], use_period=True))
    book = pd.read_excel(io.BytesIO(DL.to_excel(res, layout)), sheet_name=None)
    expected = ["Parameters", "Failed"] + (["AAPL", "MSFT"] if layout == "sheets" else ["Data"])
    assert list(book) == expected
    assert "repair" in set(book["Parameters"]["Parameter"])
    assert list(book["Failed"]["Ticker"]) == ["NOSUCH"]


def test_excel_too_many_rows(fake_yf, monkeypatch):
    monkeypatch.setattr(DL, "EXCEL_MAX_ROWS", 3)
    res = DL.fetch(DL.Request(["AAPL"], use_period=True))
    with pytest.raises(ValueError):
        DL.to_excel(res, "long")


def test_csv(fake_yf):
    res = DL.fetch(DL.Request(["AAPL", "MSFT"], use_period=True))
    data = DL.to_csv(res, "long")
    assert data.startswith(b"\xef\xbb\xbf")
    assert pd.read_csv(io.BytesIO(data), encoding="utf-8-sig").shape == (10, 10)
    assert pd.read_csv(io.BytesIO(DL.to_csv(res, "wide")), encoding="utf-8-sig").shape == (5, 17)


def test_download_page(app, fake_yf):  # noqa: F811
    app.run()
    app.session_state["page"] = "Download"
    app.run()
    assert not app.exception, app.exception
    assert app.button(key="dl_go").disabled                     # no tickers yet
    app.text_area(key="dl_text").input("aapl, msft nosuch").run()
    app.button(key="dl_go").click().run()
    assert not app.exception, app.exception
    assert fake_yf[-1]["tickers"] == ["AAPL", "MSFT", "NOSUCH"]
    result, long, _ = app.session_state["dl_result"]
    assert result.ok == ["AAPL", "MSFT"] and len(long) == 10
    # the inclusive end date in the form becomes yfinance's exclusive end
    end = app.session_state["dl_end"]
    assert fake_yf[-1]["end"] == (end + dt.timedelta(days=1)).isoformat()

    app.radio(key="dl_src").set_value("Screener results").run()
    app.button(key="dl_go").click().run()
    assert not app.exception, app.exception
    assert len(fake_yf[-1]["tickers"]) > 2
