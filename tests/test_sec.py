from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import requests

from screener.data import sec


class FakeSession:
    def __init__(self, responses: dict[str, tuple[int, Any]] | None = None):
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> requests.Response:
        self.calls.append({"url": url, "headers": headers or {}})
        res = requests.Response()
        if url in self.responses:
            status, data = self.responses[url]
            res.status_code = status
            if isinstance(data, (dict, list)):
                res._content = json.dumps(data).encode("utf-8")
            elif isinstance(data, str):
                res._content = data.encode("utf-8")
            else:
                res._content = b""
        else:
            res.status_code = 404
            res._content = b""
        return res


def make_frame_data(tag: str, period: str, cik_vals: dict[int, float], pts: int | None = None) -> dict[str, Any]:
    items = []
    for cik, val in cik_vals.items():
        items.append({
            "accn": f"000{cik}-25-000001",
            "cik": cik,
            "entityName": f"Company {cik}",
            "loc": "US",
            "start": "2025-01-01",
            "end": "2025-03-31",
            "val": val,
        })
    num_pts = pts if pts is not None else max(len(items), 1000)
    # If pts requested is higher than len(items), pad with dummy items
    if pts is not None and pts > len(items):
        for i in range(len(items), pts):
            dummy_cik = 900000 + i
            items.append({
                "accn": f"000{dummy_cik}-25-000001",
                "cik": dummy_cik,
                "entityName": f"Dummy {dummy_cik}",
                "loc": "US",
                "start": "2025-01-01",
                "end": "2025-03-31",
                "val": 100.0,
            })
    return {
        "taxonomy": "us-gaap",
        "tag": tag,
        "ccp": period,
        "uom": "USD",
        "pts": num_pts,
        "data": items,
    }


def test_missing_user_agent(tmp_path, monkeypatch):
    data_dir = tmp_path / "sec"
    ua_file = tmp_path / "sec_user_agent.txt"
    # Missing file
    res = sec.update(data_dir=data_dir, user_agent_path=ua_file)
    assert res is None

    # Empty file
    ua_file.write_text("   \n \n", encoding="utf-8")
    res = sec.update(data_dir=data_dir, user_agent_path=ua_file)
    assert res is None


def test_user_agent_and_headers_in_requests(tmp_path, monkeypatch):
    data_dir = tmp_path / "sec"
    ua_file = tmp_path / "sec_user_agent.txt"
    ua_file.write_text("Tester tester@example.invalid\n", encoding="utf-8")

    session = FakeSession()
    monkeypatch.setattr(sec, "_sleep", lambda s: None)

    # Force run with fake session
    res = sec.update(data_dir=data_dir, user_agent_path=ua_file, session=session)
    # Even if 404s, calls were made
    assert len(session.calls) > 0
    for call in session.calls:
        headers = call["headers"]
        assert headers.get("User-Agent") == "Tester tester@example.invalid"
        assert headers.get("Accept-Encoding") == "gzip, deflate"


def test_request_pacing(tmp_path, monkeypatch):
    data_dir = tmp_path / "sec"
    ua_file = tmp_path / "sec_user_agent.txt"
    ua_file.write_text("Tester tester@example.invalid\n", encoding="utf-8")

    sleeps = []
    monkeypatch.setattr(sec, "_sleep", lambda s: sleeps.append(s))

    session = FakeSession()
    sec.update(data_dir=data_dir, user_agent_path=ua_file, session=session)

    assert len(sleeps) > 0
    for s in sleeps:
        assert s >= 0.2


def test_ticker_map_function(tmp_path):
    data_dir = tmp_path / "sec"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Missing file -> {}
    assert sec.ticker_map(data_dir=data_dir) == {}

    # Write tickers
    raw = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1092014, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc."},
    }
    (data_dir / "company_tickers.json").write_text(json.dumps(raw), encoding="utf-8")

    tm = sec.ticker_map(data_dir=data_dir)
    assert tm == {320193: "AAPL", 1092014: "BRK.B"}


def test_step_back_over_404_and_small_frames(tmp_path, monkeypatch):
    data_dir = tmp_path / "sec"
    ua_file = tmp_path / "sec_user_agent.txt"
    ua_file.write_text("Tester tester@example.invalid\n", encoding="utf-8")

    monkeypatch.setattr(sec, "_sleep", lambda s: None)

    # Ref date: 2025-10-23 -> Most recent ended quarter is CY2025Q3
    ref_today = dt.date(2025, 10, 23)

    tickers_json = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }

    responses = {
        "https://www.sec.gov/files/company_tickers.json": (200, tickers_json),
        # CY2025Q3 returns 404
        # CY2025Q2 returns 200 with 500 companies (<1000)
        "https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2025Q2.json": (
            200, make_frame_data("Revenues", "CY2025Q2", {320193: 100.0}, pts=500)
        ),
        # CY2025Q1 returns 200 with 1200 companies (>=1000) -> selected as q0
        "https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2025Q1.json": (
            200, make_frame_data("Revenues", "CY2025Q1", {320193: 100.0}, pts=1200)
        ),
    }

    session = FakeSession(responses)
    out_path = sec.update(
        data_dir=data_dir,
        user_agent_path=ua_file,
        today=ref_today,
        session=session,
    )

    assert out_path is not None
    df = sec.load(data_dir=data_dir)
    assert not df.empty
    assert df.iloc[0]["last_period"] == "CY2025Q1"


def test_revenue_fallback_and_ttm_math(tmp_path, monkeypatch):
    data_dir = tmp_path / "sec"
    ua_file = tmp_path / "sec_user_agent.txt"
    ua_file.write_text("Tester tester@example.invalid\n", encoding="utf-8")
    monkeypatch.setattr(sec, "_sleep", lambda s: None)

    ref_today = dt.date(2025, 10, 23) # q0 = CY2025Q2 (if valid)

    # 3 companies:
    # 320193 (AAPL): standard company, Q4 CY2024 derived from Year (400) - Q1(90) - Q2(100) - Q3(110) = 100
    #                Fallback concept used for CY2025Q2 (concept 2 instead of concept 1)
    # 101010 (FAIL): missing quarter -> TTM is NaN
    # 202020 (NEG): prev TTM revenue <= 0 -> growth is NaN
    tickers_json = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 101010, "ticker": "FAIL", "title": "Missing Co."},
        "2": {"cik_str": 202020, "ticker": "NEG", "title": "Negative Co."},
    }

    # Quarters when q0 = CY2025Q2:
    # q0: CY2025Q2
    # q1: CY2025Q1
    # q2: CY2024Q4 (needs derivation)
    # q3: CY2024Q3
    # q4: CY2024Q2
    # q5: CY2024Q1
    # q6: CY2023Q4
    # q7: CY2023Q3
    # Annual needed: CY2024, CY2023

    responses = {
        "https://www.sec.gov/files/company_tickers.json": (200, tickers_json),
    }

    # Revenue concept 1: Revenues
    # Note: for CY2025Q2, AAPL is missing in Revenues, but present in concept 2 (RevenueFromContractWithCustomer...)
    rev_q2_2025 = make_frame_data("Revenues", "CY2025Q2", {101010: 50.0, 202020: 10.0}, pts=1000)
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2025Q2.json"] = (200, rev_q2_2025)

    rev2_q2_2025 = make_frame_data("RevenueFromContractWithCustomerExcludingAssessedTax", "CY2025Q2", {320193: 120.0}, pts=1000)
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax/USD/CY2025Q2.json"] = (200, rev2_q2_2025)

    # CY2025Q1
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2025Q1.json"] = (
        200, make_frame_data("Revenues", "CY2025Q1", {320193: 110.0, 101010: 50.0, 202020: 10.0}, pts=1000)
    )

    # CY2024 (Annual for Q4 derivation)
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2024.json"] = (
        200, make_frame_data("Revenues", "CY2024", {320193: 400.0, 101010: 200.0, 202020: -10.0}, pts=1000)
    )

    # CY2024Q3
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2024Q3.json"] = (
        200, make_frame_data("Revenues", "CY2024Q3", {320193: 110.0, 101010: 50.0, 202020: -5.0}, pts=1000)
    )

    # CY2024Q2
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2024Q2.json"] = (
        200, make_frame_data("Revenues", "CY2024Q2", {320193: 100.0, 202020: -5.0}, pts=1000) # 101010 missing here
    )

    # CY2024Q1
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2024Q1.json"] = (
        200, make_frame_data("Revenues", "CY2024Q1", {320193: 90.0, 101010: 50.0, 202020: -5.0}, pts=1000)
    )

    # CY2023Q4
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2023Q4.json"] = (
        200, make_frame_data("Revenues", "CY2023Q4", {320193: 80.0, 202020: 5.0}, pts=1000)
    )

    # CY2023Q3
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2023Q3.json"] = (
        200, make_frame_data("Revenues", "CY2023Q3", {320193: 70.0, 202020: 5.0}, pts=1000)
    )

    # Net income
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2025Q2.json"] = (
        200, make_frame_data("NetIncomeLoss", "CY2025Q2", {320193: 24.0}, pts=1000)
    )
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2025Q1.json"] = (
        200, make_frame_data("NetIncomeLoss", "CY2025Q1", {320193: 22.0}, pts=1000)
    )
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2024.json"] = (
        200, make_frame_data("NetIncomeLoss", "CY2024", {320193: 80.0}, pts=1000)
    )
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2024Q3.json"] = (
        200, make_frame_data("NetIncomeLoss", "CY2024Q3", {320193: 22.0}, pts=1000)
    )
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2024Q2.json"] = (
        200, make_frame_data("NetIncomeLoss", "CY2024Q2", {320193: 20.0}, pts=1000)
    )
    responses["https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2024Q1.json"] = (
        200, make_frame_data("NetIncomeLoss", "CY2024Q1", {320193: 18.0}, pts=1000)
    )

    session = FakeSession(responses)
    sec.update(data_dir=data_dir, user_agent_path=ua_file, today=ref_today, session=session)

    df = sec.load(data_dir=data_dir).set_index("ticker")

    # AAPL:
    # Q4 2024 derived = 400 - (90 + 100 + 110) = 100
    # Q4 2024 net income derived = 80 - (18 + 20 + 22) = 20
    # Last 4 quarters revenue: Q2 2025(120) + Q1 2025(110) + Q4 2024(100) + Q3 2024(110) = 440
    # Prev 4 quarters revenue: Q2 2024(100) + Q1 2024(90) + Q4 2023(80) + Q3 2023(70) = 340
    # Revenue growth TTM: (440 - 340) / 340 * 100 = 29.41176%
    # Last 4 quarters net income: Q2 2025(24) + Q1 2025(22) + Q4 2024(20) + Q3 2024(22) = 88
    # Net margin TTM: 88 / 440 * 100 = 20.0%
    aapl = df.loc["AAPL"]
    assert aapl["revenue_ttm"] == 440.0
    assert aapl["revenue_ttm_prev"] == 340.0
    assert pytest.approx(aapl["revenue_growth_ttm"], 0.001) == (440.0 - 340.0) / 340.0 * 100.0
    assert aapl["net_income_ttm"] == 88.0
    assert pytest.approx(aapl["net_margin_ttm"], 0.001) == 20.0

    # FAIL: missing Q2 2024 -> revenue_ttm is NaN
    fail = df.loc["FAIL"]
    assert np.isnan(fail["revenue_ttm"])

    # NEG: prev TTM revenue = (-5) + (-5) + 5 + 5 = 0 <= 0 -> growth is NaN
    neg = df.loc["NEG"]
    assert np.isnan(neg["revenue_growth_ttm"])


def test_cache_behavior_and_30_days_ticker_map(tmp_path, monkeypatch):
    data_dir = tmp_path / "sec"
    ua_file = tmp_path / "sec_user_agent.txt"
    ua_file.write_text("Tester tester@example.invalid\n", encoding="utf-8")
    monkeypatch.setattr(sec, "_sleep", lambda s: None)

    ref_today = dt.date(2025, 10, 23)
    tickers_json = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    # Provide a default frame response for any frame requested
    class DefaultingFakeSession(FakeSession):
        def get(self, url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> requests.Response:
            if url.startswith("https://data.sec.gov/api/xbrl/frames/"):
                # extract concept and period from URL
                parts = url.rstrip(".json").split("/")
                concept = parts[-3]
                period = parts[-1]
                res = requests.Response()
                res.status_code = 200
                res._content = json.dumps(make_frame_data(concept, period, {320193: 100.0}, pts=1000)).encode("utf-8")
                self.calls.append({"url": url, "headers": headers or {}})
                return res
            return super().get(url, headers, timeout)

    responses = {
        "https://www.sec.gov/files/company_tickers.json": (200, tickers_json),
    }

    session1 = DefaultingFakeSession(responses)
    sec.update(data_dir=data_dir, user_agent_path=ua_file, today=ref_today, session=session1)

    assert len(session1.calls) > 0

    # Second update with fresh files -> 0 frame requests, 0 ticker map requests
    session2 = DefaultingFakeSession(responses)
    sec.update(data_dir=data_dir, user_agent_path=ua_file, today=ref_today, session=session2)
    assert len(session2.calls) == 0

    # Simulate ticker map older than 30 days
    ticker_file = data_dir / "company_tickers.json"
    old_mtime = time.time() - (31 * 86400)
    os.utime(ticker_file, (old_mtime, old_mtime))

    session3 = DefaultingFakeSession(responses)
    sec.update(data_dir=data_dir, user_agent_path=ua_file, today=ref_today, session=session3)
    # Ticker map requested again
    assert len(session3.calls) == 1
    assert session3.calls[0]["url"] == "https://www.sec.gov/files/company_tickers.json"


def test_ticker_normalization_and_unmapped_company_filtering(tmp_path, monkeypatch):
    data_dir = tmp_path / "sec"
    ua_file = tmp_path / "sec_user_agent.txt"
    ua_file.write_text("Tester tester@example.invalid\n", encoding="utf-8")
    monkeypatch.setattr(sec, "_sleep", lambda s: None)

    ref_today = dt.date(2025, 10, 23)

    # Ticker map only has BRK-B (1092014)
    tickers_json = {
        "0": {"cik_str": 1092014, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc."}
    }

    # Frame data contains 1092014 and unmapped company 999999
    frame_data = make_frame_data("Revenues", "CY2025Q2", {1092014: 1000.0, 999999: 500.0}, pts=1000)

    responses = {
        "https://www.sec.gov/files/company_tickers.json": (200, tickers_json),
        "https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2025Q2.json": (200, frame_data),
    }

    session = FakeSession(responses)
    sec.update(data_dir=data_dir, user_agent_path=ua_file, today=ref_today, session=session)

    df = sec.load(data_dir=data_dir)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "BRK.B"
    assert 999999 not in df["cik"].values


def test_no_email_address_in_repo():
    repo_root = Path(__file__).resolve().parent.parent
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    search_dirs = [repo_root / "screener", repo_root / "tests"]
    found_emails = []

    for s_dir in search_dirs:
        for path in s_dir.rglob("*"):
            if path.is_file():
                # Skip .venv directory if inside screener
                if ".venv" in path.parts:
                    continue
                # Skip binary / bytecode files
                if path.suffix in (".pyc", ".parquet", ".bad", ".tmp", ".so", ".exe"):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    matches = email_pattern.findall(text)
                    for m in matches:
                        domain = m.split("@")[-1].lower()
                        if not (domain == "example.invalid" or domain.startswith("example.")):
                            found_emails.append((str(path.relative_to(repo_root)), m))
                except Exception:
                    pass

    assert found_emails == [], f"Found forbidden email address(es) in repo: {found_emails}"
