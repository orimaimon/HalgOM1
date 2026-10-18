# HalgOM — project context

Two products live in this repo:

1. **HalgOM backtester** (root `*.py`, `strategies/`): Monte Carlo + walk-forward research engine for
   US equity strategies. See `README.md`.
2. **HalgOM Screener** (`screener/`, `run_screener.bat`): a local, Finviz-style stock screener built for a
   non-technical end user (Ori's relative, "דוד חיים") who runs it on his own Windows PC.

The user (Ori) writes in Hebrew; answer in Hebrew. UI terms stay in English (Finviz terminology),
end-user explanations/tooltips are Hebrew.

## Data is NOT in the repo

`DB/*.parquet` (~400MB) and `DB/mc_runs.sqlite` are gitignored and exceed GitHub limits.
- Backtests need `DB/simulation_db.parquet`; rebuild with `python DB/build_simulation_db.py`
  (downloads 20y of OHLCV from yfinance — long) or work on code/tests only.
- The screener needs no local data: it fetches live from TradingView and caches in `screener/_data/`.

## Running

```bash
# screener deps are pinned in a lock file (do not use `uv run --with-requirements`:
# it resolved an incompatible numpy 1.x + pyarrow combo once)
uv venv --python 3.12 screener/.venv
uv pip install --python screener/.venv -r screener/requirements.lock pytest
screener/.venv/bin/python -m pytest tests/ -q          # Windows: screener\.venv\Scripts\python
screener/.venv/bin/python -m streamlit run screener/app.py
```

Known: `tests/test_strategies.py::TestTopNVolumeStrategy::test_no_buys_in_bear_regime` fails
(pre-existing, backtester side — not caused by the screener).

## Screener architecture

- `screener/fields.py` — column registry (TradingView field, label, format kind, Hebrew tooltip) + table views.
  Percent columns hold percent numbers (5.0 == 5%).
- `screener/filters.py` — Finviz-style dropdown filters; each option is a predicate on the whole DataFrame.
- `screener/data/tradingview.py` — direct `requests` client for `scanner.tradingview.com` (unofficial,
  ~15 min delayed). Universe = US primary listings, common stocks + ADRs (`is_primary`, `typespecs`).
  We do NOT use the `tradingview-screener` package: its default query silently filters out ETFs.
  ONE request for the whole universe (all fields) per 15-min refresh + one small market request.
  Never add parallel or more frequent queries (unofficial API, block risk).
  Field syntax: `|1W` = weekly timeframe, `[1]` = previous bar (works). `[5]` and deeper offsets return
  nothing (checked live 2026-10-18), so "crossed in the last N days" is not available — only "today".
  Unknown fields come back as null (NaN), not as an error: verify fill rates live before adding one.
- `screener/data/store.py` — latest-cache (15 min), one snapshot per day (+ `YYYY-MM-DD.json` with the
  data source: provider / update_mode / delay), offline fallback ("stale"). A cache missing required
  columns (saved by an older version) is refetched.
- `screener/derive.py`, `scores.py` — derived columns and Value/Quality/Momentum percentile scores.
  `add_derived` tolerates old snapshots: missing provider columns become NaN.
- `screener/signals.py` — signal definitions shared by the Signals filters and the "Signals" / "# Signals"
  columns (candles, N-month breakouts vs yesterday's high, crosses today, volume spike, consolidation).
- `screener/history.py` — "New since yesterday": same pipeline on the previous snapshot day.
- `screener/groups.py` — Groups page (sector/industry averages) and the plotly treemap heatmap.
- `screener/market.py` — regime header, same rules as `compute_macro_regime` in `DB/build_simulation_db.py`;
  plus SPY ADX trend strength as a separate badge (does not change the regime rules).
- `screener/export.py` — Excel (numbers stay numeric; custom formats like `0.00,,,"B"`), CSV, TradingView watchlist.
- `screener/app.py` — Streamlit UI (pages: Screener / Groups / Heatmap). Selecting a row embeds the official
  TradingView Advanced Chart widget. Widget state is re-assigned each run (`keep_widget_state`) so filters
  survive a visit to another page.

Rules:
- Keep the provider swappable: providers return canonical column keys from `fields.COLUMNS`.
- `run_screener.bat` must stay ASCII with CRLF line endings (cmd breaks otherwise; see `.gitattributes`).
- Built-in preset option labels must match filter labels exactly (tests enforce this).
- `tests/test_screener.py::test_app_smoke` runs the whole app offline with `streamlit.testing.v1.AppTest`
  (provider and data dirs monkeypatched); paths in store/presets/export are resolved at call time for this.

## Screener roadmap

Phase 1 — done (2026-10-18).

Phase 2 — done on branch `screener-phase2` (2026-10-18; PR, not merged to main yet):
Signals tab + "Signals" column, Index filter, weekly filters, SMA150 + Stage 2 (Minervini) preset,
"Hide junk" (on by default), "New since yesterday", Groups page, Heatmap (plotly), SPY ADX badge,
data-source metadata per snapshot.

Phase 2 leftovers:
- Remaining Finviz gaps: IPO date, shares-outstanding filter. (No short-float data on TradingView.)
- SPACs (~$10, tiny range) dominate "Tight 1M" signals; consider excluding them in "Hide junk".

Phase 3:
- HalgOM strategy presets via a two-stage funnel (TradingView prefilter → yfinance history for ~200 names).
- Watchlist that tracks performance since added; daily email/Telegram report of preset changes.
- Optional: real-time data via the user's own TradingView `sessionid` cookie (account-flag risk — opt-in only).

## Backtester research notes (as of 2026-05-17 — verify against `DB/mc_runs.sqlite` before relying on them)

Production candidate validated on 11 OOS folds: `topn_focused_v2`, top_n=6, hold_days=130, min_price=2.0,
equal sizing, regime filter on, `regime_entry_buffer=0.01`, `regime_confirm_days=1`
(median excess vs QQQ ≈ +9.5pp across folds). DVMC is inferior. See `SANDBOX.md` for the latest session log.
