# HalgOM Screener — project context

This repo is the **HalgOM Screener** (`screener/`, `run_screener.bat`): a local, Finviz-style stock screener
built for a non-technical end user (Ori's relative, "דוד חיים") who runs it on his own Windows PC.
`run_screener.bat` pulls `main` on every launch, so `main` is what the end user runs: keep it stable and
work through branches + PRs.

The HalgOM backtester (Monte Carlo / walk-forward engine, strategies, `DB/` build scripts) used to live here
too; it was removed on 2026-10-18 (Ori keeps a backup; it is also in git history before the
"screener-only" commit).

The user (Ori) writes in Hebrew; answer in Hebrew. The screener UI is Hebrew (RTL) by default with an
English switch (`screener/i18n.py`); tooltips and explanations are Hebrew in both languages.

## Data

The screener needs no data in the repo: it fetches live from TradingView and keeps its cache, daily
snapshots, saved screens, exports and logs in `screener/_data/` (gitignored).

## Running

```bash
# screener deps are pinned in a lock file (do not use `uv run --with-requirements`:
# it resolved an incompatible numpy 1.x + pyarrow combo once)
uv venv --python 3.12 screener/.venv
uv pip install --python screener/.venv -r screener/requirements.lock pytest
screener/.venv/bin/python -m pytest tests/ -q          # Windows: screener\.venv\Scripts\python
screener/.venv/bin/python -m streamlit run screener/app.py
```

## Screener architecture

- `screener/fields.py` — column registry (canonical keys, label, format kind, source marker, Hebrew tooltip) + table views.
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
- `screener/data/prices.py` — history layer from yfinance (adjusted daily OHLCV, local daily cache).
- `screener/data/store.py` — latest-cache (15 min), one snapshot per day (+ `YYYY-MM-DD.json` with the
  data source: provider / update_mode / delay), offline fallback ("stale"). A cache missing required
  columns (saved by an older version) is refetched.
- `screener/derive.py`, `scores.py` — derived columns and Value/Quality/Momentum percentile scores.
  `add_derived` tolerates old snapshots: missing provider columns become NaN.
- `screener/signals.py` — signal definitions shared by the Signals filters and the "Signals" / "# Signals"
  columns (candles, N-month breakouts vs yesterday's high, crosses today, volume spike, consolidation).
- `screener/history.py` — "New since yesterday": same pipeline on the previous snapshot day.
- `screener/groups.py` — Groups page (sector/industry averages) and the plotly treemap heatmap.
- `screener/i18n.py` — display language. Internal values stay English (filter options, preset names, views,
  pages, sector/country values) so saved screens, presets and tests are language-independent; only display
  is translated (`format_func`, `display_values`). Hebrew style: Hebrew + familiar English abbreviation
  ("מכפיל רווח (P/E)"); table headers keep short abbreviations; sectors and countries translated, the 128
  industries stay English; unknown strings fall back to English. RTL for the interface, tables/heatmap/chart
  stay LTR. Tests require a Hebrew translation for every filter, option, column, view and preset.
- `screener/lifecycle.py` — "Exit" button (clean `Runtime.stop()`, the console window closes) and an idle
  watchdog that stops the server after 15 minutes with no open browser tab.
- `screener/market.py` — regime header, same rules as `compute_macro_regime` in the HalgOM backtester;
  plus SPY ADX trend strength as a separate badge (does not change the regime rules).
- `screener/export.py` — Excel (numbers stay numeric; custom formats like `0.00,,,"B"`), CSV, TradingView watchlist.
- `.streamlit/config.toml` + `screener/theme.py` — the look ("ledger": paper background, ink-blue accent,
  IBM Plex Sans Hebrew). Green/red mean gain/loss only; take colors from `theme.py`, don't hard-code hex.
- `screener/guide.html` — Hebrew user guide, shown on the "Guide" page. It quotes UI names; when a button, page,
  tab, view or built-in preset is renamed, update the guide (tests enforce the presets and page/tab/view names).
  `.gitignore` ignores `*.html`, so the guide is un-ignored explicitly.
- `screener/portfolio.py` — watchlists and portfolios: one CSV/Excel import format (one row per trade, Hebrew headers
  allowed; tickers only = watchlist), average-cost positions and P&L from the universe's prices (no extra requests),
  stored per machine in `screener/_data/portfolios.json`. Import problems are `Problem` codes, Hebrew text in i18n.
- `screener/app.py` — Streamlit UI (pages: Screener / Groups / Heatmap / Portfolios / Guide). Selecting a row embeds the official
  TradingView Advanced Chart widget. Widget state is re-assigned each run (`keep_widget_state`) so filters
  survive a visit to another page.

Rules:
- Keep the provider swappable: providers return canonical column keys from `fields.COLUMNS`.
- `run_screener.bat` must stay ASCII with CRLF line endings (cmd breaks otherwise; see `.gitattributes`).
- `run_screener.bat` updates itself while cmd is running it, and cmd resumes at the same byte offset.
  Launchers before 2026-10-19 ran `git pull` in an update block (`if not defined SCREENER_NO_UPDATE ...`)
  that must stay byte-identical at bytes 529–785 (an old launcher that pulls the new file resumes at 785).
  Edit below it freely; above it keep the length (pad the `rem ---` line). The block is now switched off
  (`SCREENER_NO_UPDATE=1` just before it, restored right after it).
- Updates: `screener/update.py` (dulwich, no git needed), run by step 4 of the launcher, which must stay ONE line
  (update.py may rewrite the .bat; the line restarts the launcher with `cmd /c call "%~f0"` on exit code 3, which
  keeps a folder name with `&` or parentheses quoted). It turns a ZIP/copied folder into a checkout of main,
  fast-forwards a clean checkout on main, and leaves anything else alone (local edits, staged files, untracked files
  in the way, another branch); it never blocks the start (45 s deadline for the whole fetch). A checkout writes
  `.git/HALGOM_CHECKOUT` first and removes it last, so a cut-off checkout is finished on the next run.
  Remote: SSH with a read-only deploy key if `%USERPROFILE%\.halgom\deploy_key` exists (a `deploy_key` or
  `deploy_key.txt` next to the .bat is moved there on first run; `deploy_key*` is gitignored, never commit it),
  trying port 22, then ssh.github.com:443, then anonymous HTTPS. GitHub's SSH host keys are pinned in update.py.
  Log: `screener/_data/logs/update.log`.
- Launcher behavior: fixed port 8501 (if another program holds it, the launcher says so); a second launch only
  opens the browser (health check; `SCREENER_NO_BROWSER=1` skips the browser); a desktop shortcut is created once
  (`screener/_data/shortcut.done`); a venv that does not start (copied from another PC) or is older than Python
  3.11 is rebuilt; without uv it falls back to an installed Python; a failed package install still runs the
  update (the fix may be in it) before stopping.
- CI (`.github/workflows/tests.yml`): pytest on Linux + Windows, and a launcher smoke test that runs the real
  `run_screener.bat` on a clean Windows runner (health endpoint only, never the app page, so no TradingView calls).
- Built-in preset option labels must match filter labels exactly (tests enforce this).
- `tests/test_screener.py::test_app_smoke` runs the whole app offline with `streamlit.testing.v1.AppTest`
  (provider and data dirs monkeypatched); paths in store/presets/export are resolved at call time for this.

## Screener roadmap

Phase 1 — done (2026-10-18).

Phase 2 — done (2026-10-18; PRs #2 phase 2, #3 Hebrew UI, then the screener-only cleanup):
Signals tab + "Signals" column, Index filter, weekly filters, SMA150 + Stage 2 (Minervini) preset,
"Hide junk" (on by default), "New since yesterday", Groups page, Heatmap (plotly), SPY ADX badge,
data-source metadata per snapshot.

Phase 2 leftovers:
- Remaining Finviz gaps: IPO date, shares-outstanding filter. (No short-float data on TradingView.)
- SPACs (~$10, tiny range) dominate "Tight 1M" signals; consider excluding them in "Hide junk".

Phase 3:
- HalgOM strategy presets via a two-stage funnel (TradingView prefilter → yfinance history for ~200 names).
- **[DONE]** Watchlists (performance since added) and portfolios (Portfolios page). Next: splits/dividends after the
  trade date and value over time vs SPY/QQQ (needs the price-history layer); prices for ETFs held in a portfolio.
- Daily email/Telegram report of preset changes.
- Optional: real-time data via the user's own TradingView `sessionid` cookie (account-flag risk — opt-in only).

Later — less dependence on TradingView (discussed 2026-10-18; add layers, don't replace it; own branch/PR):
1. **[DONE]** Move the TradingView field mapping out of `fields.COLUMNS` (`Col.tv`) into the provider module, so the
   registry holds canonical keys only and each provider brings its own mapping. Small, safe first step.
2. **[DONE]** History layer from yfinance (data layer done, no UI yet): adjusted daily OHLCV stored locally, updated once a day. Enables a real
   adjusted price history, total return incl. dividends (TradingView has neither — checked live: no
   adjusted/total-return field, `adjustment` options are ignored), crosses "within N days" (`[5]` offsets
   are empty on TradingView) and breakouts over any window, computed by us. TradingView stays the fast
   snapshot source. Costs: storage, daily update, first-run download time, yfinance rate limits.
3. Fallback provider when TradingView blocks or changes (today the app only falls back to stale cache).
Coupling to remove along the way: signals use TradingView-computed fields (`Candle.*`, `[1]`, `High.xM`);
the market header and the chart widget are TradingView too. Candidate sources: yfinance (free, unofficial),
SEC EDGAR (official fundamentals), paid APIs (FMP, Polygon, EODHD, Tiingo, Alpha Vantage — check current
pricing/limits before choosing), Finviz (scraping or paid Elite).

## Backtester note (for the Phase 3 strategy presets; the backtester itself is not in this repo)

As of 2026-05-17 the production candidate was `topn_focused_v2`: top_n=6, hold_days=130, min_price=2.0,
equal sizing, regime filter on, `regime_entry_buffer=0.01`, `regime_confirm_days=1` (median excess vs QQQ
≈ +9.5pp across 11 OOS folds). Verify against the backtester before building a preset on it.
