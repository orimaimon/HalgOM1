"""
build_simulation_db.py  v3.0
────────────────────────────
Parquet DB for simulations — 20yr, Common Stocks only, multi-exchange.

Changes from v2.0:
  + Schema v3.0 (stored in parquet metadata)
  + Market_Cap + Shares_Outstanding (opt-in via --with-market-cap)
  + Multi-horizon returns: Return_60d_Pct, Return_120d_Pct, Return_252d_Pct
  + ATR_14 (Average True Range, 14-day)
  + Separate benchmarks.parquet (SPY + QQQ + VIX with Adj_Close)
  + Incremental update mode (--incremental): download only delta
  + Enhanced validation (gap detection per ticker)
  + Default exchange = NASDAQ only

Pipeline:
  1. Load ticker lists (NASDAQ default)
  2. Filter Common Stocks only (no ETF/warrant/unit/right/preferred/SPAC)
  3. Download OHLCV from yfinance (batches + checkpoint)
  4. [Optional] Fetch Shares_Outstanding for Market_Cap
  5. Download benchmark indices + macro data
  6. Compute indicators: DV_20d, RSI_14, Return_20d/60d/120d/252d, ATR_14, Volatility
  7. Compute macro: VIX, SPY SMA200, HYG SMA20
  8. Compute Market Regime: bull / bear / sideways
  9. Map sectors
 10. Validate + save (with schema_version metadata)
 11. Save separate benchmarks.parquet (SPY + QQQ + VIX)

Usage:
  python build_simulation_db.py                        # NASDAQ full build
  python build_simulation_db.py --with-market-cap      # + historical Market_Cap
  python build_simulation_db.py --incremental          # Update existing DB
  python build_simulation_db.py --skip-benchmarks      # Main DB only
  python build_simulation_db.py --from-parquet X.parquet
  python build_simulation_db.py --resume
"""

import argparse, datetime, json, logging, random, re, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("build_simulation_db.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Constants
SCHEMA_VERSION   = "v3.0"
OUTPUT_DEFAULT   = Path("simulation_db.parquet")
BENCHMARK_OUTPUT = Path("benchmarks.parquet")
SHARES_CACHE     = Path("shares_cache.parquet")
CHECKPOINT_FILE  = Path("sim_db_checkpoint.json")
YEARS_BACK       = 20
BATCH_SIZE       = 50
DV_WINDOW        = 20

# Indices/macro merged into main parquet (for regime computation)
BENCHMARK_TICKERS = {
    "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^NDX": "Nasdaq 100",
    "^IXIC": "Nasdaq Composite", "^RUT": "Russell 2000",
}
MACRO_TICKERS = {"^VIX": "VIX", "HYG": "HYG", "^PCALL": "Put/Call Ratio"}
ALL_INDEX_TICKERS = list(BENCHMARK_TICKERS) + list(MACRO_TICKERS)

EXCHANGE_SOURCES = {
    "NASDAQ": {"file": "nasdaqlisted.txt",  "format": "pipe_delimited"},
    "NYSE":   {"file": "nyselisted.txt",    "format": "simple_list"},
    "TASE":   {"file": "tase_listed.txt",   "format": "simple_list", "suffix": ".TA"},
}

NON_COMMON_KEYWORDS = (
    "warrant", "warrants", "right", "rights", "unit", "units",
    "preferred", "pref", "depositary", "debenture", "note", "notes",
    "bond", "acquisition corp", "blank check", "spac",
    "etf", "fund", "trust", "index",
)
NON_COMMON_SUFFIXES = {"W","R","U","P","WS","WT","WI"}

# v3.0 — full schema including Market_Cap, multi-horizon returns, ATR
FINAL_COLUMNS = [
    "Date","Ticker","Sector","Industry","Exchange","Type",
    "Open","High","Low","Close","Adj_Close","AdjFactor","Volume",
    "Shares_Outstanding","Market_Cap",
    "Dollar_Volume_20d_Avg",
    "Return_20d_Pct","Return_60d_Pct","Return_120d_Pct","Return_252d_Pct",
    "RSI_14","ATR_14",
    "Daily_Volatility_20d","Current_Daily_Return",
    "VIX_Close","SPY_Close","SPY_SMA_200","HYG_Close","HYG_SMA_20",
    "Market_Regime",
]


# ═══════════════════ TICKER LOADING & FILTERING ═════════════════════════════

def _is_common_by_name(name: str) -> bool:
    nl = name.lower()
    if "common stock" in nl:
        return True
    return not any(kw in nl for kw in NON_COMMON_KEYWORDS)

def _is_common_by_ticker(t: str) -> bool:
    t = t.upper().strip()
    for suf in NON_COMMON_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf) + 1:
            return False
    if len(t.split(".")[0]) > 5:
        return False
    if any(c in t for c in ("$","^")):
        return False
    return True


def load_nasdaq_tickers(fp: Path) -> List[dict]:
    if not fp.exists():
        logger.warning(f"  not found: {fp}"); return []
    df = pd.read_csv(fp, sep="|", encoding="utf-8",
                     encoding_errors="ignore", on_bad_lines="skip")
    sym_col, name_col = df.columns[0], df.columns[1]
    test_col = next((c for c in df.columns if "TEST" in c.upper()), None)
    etf_col  = next((c for c in df.columns if "ETF"  in c.upper()), None)
    results, stats = [], {"etf":0,"test":0,"name":0,"ticker":0,"invalid":0}
    for _, row in df.iterrows():
        tk = str(row[sym_col]).strip().upper()
        nm = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
        if not tk or not re.match(r'^[A-Z]{1,5}$', tk):
            stats["invalid"] += 1; continue
        if test_col and str(row.get(test_col,"")).strip().upper() == "Y":
            stats["test"] += 1; continue
        if etf_col and str(row.get(etf_col,"")).strip().upper() == "Y":
            stats["etf"] += 1; continue
        if not _is_common_by_name(nm):
            stats["name"] += 1; continue
        if not _is_common_by_ticker(tk):
            stats["ticker"] += 1; continue
        results.append({"ticker": tk, "name": nm, "exchange": "NASDAQ"})
    logger.info(f"  NASDAQ: {len(results):,} Common Stocks from {len(df):,}")
    for k,v in stats.items():
        if v: logger.info(f"    filtered {v:,} ({k})")
    return results

def load_simple_list(fp: Path, exchange: str, suffix: str = "") -> List[dict]:
    if not fp.exists():
        logger.warning(f"  not found: {fp}"); return []
    lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    results, seen, filt = [], set(), 0
    for line in lines:
        t = line.strip().upper()
        if not t or not re.match(r'^[A-Z]{1,5}$', t): continue
        if t in seen: continue
        seen.add(t)
        if not _is_common_by_ticker(t):
            filt += 1; continue
        results.append({"ticker": t + suffix, "name": "", "exchange": exchange})
    logger.info(f"  {exchange}: {len(results):,} tickers" +
                (f" (filtered {filt})" if filt else ""))
    return results

def load_all_tickers(exchanges: List[str]) -> Tuple[List[dict], Dict[str,str]]:
    logger.info("Step 1: Loading ticker lists")
    all_t = []
    for ex in exchanges:
        if ex not in EXCHANGE_SOURCES:
            logger.warning(f"  unknown exchange: {ex}"); continue
        src = EXCHANGE_SOURCES[ex]
        fp  = Path(src["file"])
        suf = src.get("suffix", "")
        if src["format"] == "pipe_delimited":
            all_t.extend(load_nasdaq_tickers(fp))
        else:
            all_t.extend(load_simple_list(fp, ex, suf))
    # dedup
    seen = {}
    for t in all_t:
        s = t["ticker"]
        if s not in seen or (t["name"] and not seen[s]["name"]):
            seen[s] = t
    deduped = list(seen.values())
    emap = {t["ticker"]: t["exchange"] for t in deduped}
    logger.info(f"  Total: {len(deduped):,} unique Common Stocks")
    for ex in exchanges:
        c = sum(1 for t in deduped if t["exchange"] == ex)
        logger.info(f"    {ex}: {c:,}")
    return deduped, emap


# ═══════════════════ DOWNLOAD ═══════════════════════════════════════════════

def _load_ckpt() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"done": [], "temp": None}

def _save_ckpt(done: Set[str], temp: str):
    CHECKPOINT_FILE.write_text(json.dumps({"done": sorted(done), "temp": temp}))

def _normalize_ohlcv(sym_df: pd.DataFrame, sym: str) -> Optional[pd.DataFrame]:
    df = sym_df.copy()
    df.index.name = "Date"
    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    for col in ("Open","High","Low","Close"):
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce") if col in df.columns else np.nan
    ac = "Adj Close"
    df["Adj_Close"] = pd.to_numeric(df[ac], errors="coerce") if ac in df.columns else df["Close"]
    df["Volume"] = pd.to_numeric(df.get("Volume",0), errors="coerce").fillna(0).astype("int64")
    cs = df["Close"].replace(0, np.nan)
    df["AdjFactor"] = (df["Adj_Close"] / cs).fillna(1.0).clip(lower=1e-6)
    for c in ("Open","High","Low"):
        df[c] = df[c].fillna(df["Close"])
    df["Ticker"] = sym
    df["Type"] = "stock"
    r = df[["Ticker","Date","Open","High","Low","Close","Adj_Close","AdjFactor","Volume","Type"]].dropna(subset=["Date","Close"])
    return r if not r.empty else None

def _extract_sym(raw, sym):
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            for lvl in (0,1):
                if sym in raw.columns.get_level_values(lvl):
                    sd = raw.xs(sym, axis=1, level=lvl).dropna(how="all")
                    return sd if not sd.empty else None
            return None
        return raw
    except Exception:
        return None

def _dl_single(sym, start, end):
    import yfinance as yf
    for a in range(2):
        try:
            raw = yf.download(sym, start=start, end=end, interval="1d",
                              auto_adjust=False, progress=False, threads=False)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                return _normalize_ohlcv(raw, sym)
        except Exception:
            if a < 1: time.sleep(3)
    return None

def _append_temp(new_df, path):
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False, compression="snappy")

def download_stocks(tickers, emap, start, end, temp_out, resume=False, bs=BATCH_SIZE):
    import yfinance as yf
    done = set(_load_ckpt().get("done",[])) if resume else set()
    remaining = [t for t in tickers if t not in done]
    logger.info(f"  To download: {len(remaining):,}" +
                (f" (resuming, {len(done):,} done)" if done else ""))
    if not remaining:
        return pd.read_parquet(temp_out) if temp_out.exists() else pd.DataFrame()
    batches = [remaining[i:i+bs] for i in range(0, len(remaining), bs)]
    t0 = time.time()
    for bi, batch in enumerate(batches, 1):
        bt0 = time.time()
        try:
            raw = yf.download(batch, start=start, end=end, interval="1d",
                              group_by="ticker", auto_adjust=False,
                              progress=False, threads=False)
        except Exception as e:
            logger.warning(f"  Batch {bi} error: {e}"); raw = None
        rows = []
        if raw is not None and not raw.empty:
            for sym in batch:
                sd = _extract_sym(raw, sym)
                if sd is not None:
                    r = _normalize_ohlcv(sd, sym)
                    if r is not None:
                        r["Exchange"] = emap.get(sym, "UNKNOWN")
                        rows.append(r)
        got = {r["Ticker"].iloc[0] for r in rows} if rows else set()
        for sym in batch:
            if sym not in got:
                r = _dl_single(sym, start, end)
                if r is not None:
                    r["Exchange"] = emap.get(sym,"UNKNOWN")
                    rows.append(r)
                time.sleep(random.uniform(0.3,0.7))
        if rows:
            _append_temp(pd.concat(rows, ignore_index=True), temp_out)
        done.update(batch)
        if bi % 5 == 0 or bi == len(batches):
            _save_ckpt(done, str(temp_out))
        el = time.time() - t0
        pr = bi * bs
        eta = str(datetime.timedelta(seconds=int((len(remaining)-min(pr,len(remaining)))/max(pr/max(el,1),0.01))))
        nr = sum(len(r) for r in rows) if rows else 0
        logger.info(f"  [{bi:>4}/{len(batches)}] {min(pr,len(remaining)):>5,}/{len(remaining):>5,} "
                    f"{nr:>5,} rows ETA {eta} ({(time.time()-bt0)*1000:.0f}ms)")
        time.sleep(random.uniform(1.5, 3.0))
    return pd.read_parquet(temp_out) if temp_out.exists() else pd.DataFrame()

def download_indices(start, end):
    import yfinance as yf
    logger.info(f"  Downloading {len(ALL_INDEX_TICKERS)} indices...")
    rows = []
    for sym in ALL_INDEX_TICKERS:
        for a in range(3):
            try:
                raw = yf.download(sym, start=start, end=end, interval="1d",
                                  auto_adjust=False, progress=False, threads=False)
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    r = _normalize_ohlcv(raw, sym)
                    if r is not None:
                        r["Type"] = "index"; r["Exchange"] = "INDEX"
                        rows.append(r)
                        logger.info(f"    OK {sym}: {len(r):,} rows"); break
            except Exception as e:
                if a < 2: time.sleep(3*(a+1))
                else: logger.warning(f"    FAIL {sym}: {e}")
        time.sleep(random.uniform(1.0,2.0))
    return pd.concat(rows, ignore_index=True) if rows else None


# ═══════════════════ SHARES OUTSTANDING (v3.0) ══════════════════════════════

def fetch_shares_outstanding(tickers: List[str], cache_path: Path) -> pd.DataFrame:
    """
    Fetch current sharesOutstanding via yfinance .info (cached).
    Slow (~0.5-1s/ticker) but one-time; subsequent runs reuse cache.
    Returns DataFrame: Ticker | Shares_Outstanding
    """
    import yfinance as yf

    cached: Dict[str, float] = {}
    if cache_path.exists():
        try:
            c = pd.read_parquet(cache_path)
            cached = dict(zip(c["Ticker"], c["Shares_Outstanding"]))
            logger.info(f"  Loaded {len(cached):,} shares from cache")
        except Exception as e:
            logger.warning(f"  Cache read error: {e}")

    to_fetch = [t for t in tickers if t not in cached]
    if not to_fetch:
        logger.info(f"  All {len(tickers):,} tickers already cached")
        return pd.DataFrame({"Ticker": list(cached.keys()),
                             "Shares_Outstanding": list(cached.values())})

    logger.info(f"  Fetching shares for {len(to_fetch):,} tickers (~{len(to_fetch)/60:.0f} min)")
    t0 = time.time()
    fail = 0

    for i, t in enumerate(to_fetch, 1):
        try:
            info = yf.Ticker(t).info
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            cached[t] = float(shares) if shares else np.nan
        except Exception:
            cached[t] = np.nan
            fail += 1

        # Save cache every 200 tickers in case of crash
        if i % 200 == 0:
            tmp = pd.DataFrame({"Ticker": list(cached.keys()),
                                "Shares_Outstanding": list(cached.values())})
            tmp.to_parquet(cache_path, index=False)
            el = time.time() - t0
            eta = (len(to_fetch) - i) * (el / i)
            logger.info(f"    [{i:>5,}/{len(to_fetch):,}] "
                        f"fail={fail} ({fail/i*100:.1f}%) "
                        f"ETA {datetime.timedelta(seconds=int(eta))}")
        time.sleep(random.uniform(0.1, 0.3))

    df = pd.DataFrame({"Ticker": list(cached.keys()),
                       "Shares_Outstanding": list(cached.values())})
    df.to_parquet(cache_path, index=False)
    non_null = df["Shares_Outstanding"].notna().sum()
    logger.info(f"  Shares done: {non_null:,}/{len(df):,} ({non_null/len(df)*100:.1f}%) "
                f"non-null, {fail} errors")
    return df


def compute_market_cap(df: pd.DataFrame, shares_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge shares into df, compute Market_Cap = Adj_Close × Shares_Outstanding"""
    if shares_df is None or shares_df.empty:
        df["Shares_Outstanding"] = np.nan
        df["Market_Cap"] = np.nan
        return df
    shares_map = dict(zip(shares_df["Ticker"], shares_df["Shares_Outstanding"]))
    df["Shares_Outstanding"] = df["Ticker"].map(shares_map).astype("float64")
    df["Market_Cap"] = (df["Adj_Close"].astype("float64") *
                        df["Shares_Outstanding"]).round(0)
    logger.info(f"  Market_Cap computed: {df['Market_Cap'].notna().sum():,} rows non-null")
    return df


# ═══════════════════ INDICATORS ═════════════════════════════════════════════

def _rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d.clip(upper=0))
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators(df):
    logger.info("Step 4: Computing indicators")
    df = df.sort_values(["Ticker","Date"]).copy()
    g_adj = df.groupby("Ticker")["Adj_Close"]

    logger.info("  -> Dollar_Volume_20d_Avg")
    dv = df["Adj_Close"] * df["Volume"].astype("float64")
    df["Dollar_Volume_20d_Avg"] = dv.groupby(df["Ticker"]).transform(
        lambda s: s.rolling(DV_WINDOW, min_periods=5).mean()).round(0)

    logger.info("  -> Return_20d_Pct / 60d / 120d / 252d")
    df["Return_20d_Pct"]  = g_adj.transform(lambda s: s.pct_change(20)*100).round(2)
    df["Return_60d_Pct"]  = g_adj.transform(lambda s: s.pct_change(60)*100).round(2)
    df["Return_120d_Pct"] = g_adj.transform(lambda s: s.pct_change(120)*100).round(2)
    df["Return_252d_Pct"] = g_adj.transform(lambda s: s.pct_change(252)*100).round(2)

    logger.info("  -> Current_Daily_Return")
    df["Current_Daily_Return"] = g_adj.transform(lambda s: s.pct_change()*100).round(4)

    logger.info("  -> RSI_14")
    df["RSI_14"] = g_adj.transform(lambda s: _rsi(s,14)).round(2)

    logger.info("  -> Daily_Volatility_20d")
    dr = g_adj.transform(lambda s: s.pct_change())
    df["Daily_Volatility_20d"] = dr.groupby(df["Ticker"]).transform(
        lambda s: s.rolling(20, min_periods=5).std()).round(6)

    logger.info("  -> ATR_14")
    # True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    prev_close = df.groupby("Ticker")["Close"].shift(1)
    tr_hl = (df["High"] - df["Low"]).astype("float64")
    tr_hc = (df["High"] - prev_close).abs().astype("float64")
    tr_lc = (df["Low"]  - prev_close).abs().astype("float64")
    tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)
    df["ATR_14"] = tr.groupby(df["Ticker"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()).round(4)

    logger.info("  Done")
    return df


# ═══════════════════ MACRO + REGIME ═════════════════════════════════════════

def compute_macro_regime(df):
    logger.info("Step 5: Macro + Market Regime")
    frames = []
    for tk, col in [("^VIX","VIX_Close"),("^GSPC","SPY_Close"),("HYG","HYG_Close")]:
        sub = df[df["Ticker"]==tk][["Date","Adj_Close"]].copy()
        sub = sub.sort_values("Date").drop_duplicates("Date",keep="last")
        sub = sub.rename(columns={"Adj_Close": col})
        if not sub.empty:
            if col == "SPY_Close":
                sub["SPY_SMA_200"] = sub[col].rolling(200, min_periods=50).mean().round(2)
            if col == "HYG_Close":
                sub["HYG_SMA_20"] = sub[col].rolling(20, min_periods=5).mean().round(2)
            frames.append(sub)
            logger.info(f"  {tk}: {len(sub):,} rows")
    if frames:
        m = frames[0]
        for f in frames[1:]: m = m.merge(f, on="Date", how="outer")
        m = m.sort_values("Date")
        for c in m.columns:
            if c != "Date": m[c] = m[c].ffill()
        df = df.merge(m, on="Date", how="left")
        for c in m.columns:
            if c != "Date" and c in df.columns: df[c] = df[c].ffill()
    req = ["VIX_Close","SPY_Close","SPY_SMA_200","HYG_Close","HYG_SMA_20"]
    if all(c in df.columns for c in req):
        bc = ((df["VIX_Close"]>25).astype(int) +
              (df["SPY_Close"]<df["SPY_SMA_200"]).astype(int) +
              (df["HYG_Close"]<df["HYG_SMA_20"]).astype(int))
        bull = (df["VIX_Close"]<18) & (df["SPY_Close"]>df["SPY_SMA_200"]) & (df["HYG_Close"]>df["HYG_SMA_20"])
        bear = bc >= 2
        df["Market_Regime"] = "sideways"
        df.loc[bull, "Market_Regime"] = "bull"
        df.loc[bear, "Market_Regime"] = "bear"
        df.loc[bull & bear, "Market_Regime"] = "bear"
        st = df[df["Type"]=="stock"]
        for r in ["bull","bear","sideways"]:
            c = (st["Market_Regime"]==r).sum()
            logger.info(f"    {r:<10}: {c:>10,} ({c/max(len(st),1)*100:.1f}%)")
    else:
        df["Market_Regime"] = "unknown"
        logger.warning("  Missing macro data -> regime=unknown")
    return df


# ═══════════════════ SECTORS ════════════════════════════════════════════════

def merge_sectors(df, sm_path):
    logger.info("Step 6: Sector mapping")
    if sm_path and sm_path.exists():
        try:
            sm = pd.read_parquet(sm_path)
            df["Sector"]   = df["Ticker"].map(dict(zip(sm["Ticker"],sm.get("Sector","Unknown")))).fillna("Unknown")
            df["Industry"] = df["Ticker"].map(dict(zip(sm["Ticker"],sm.get("Industry","Unknown")))).fillna("Unknown")
            m = (df["Sector"]!="Unknown").sum(); t = len(df[df["Type"]=="stock"])
            logger.info(f"  Mapped {m:,}/{t:,} ({m/max(t,1)*100:.1f}%)")
        except Exception as e:
            logger.warning(f"  {e}"); df["Sector"]="Unknown"; df["Industry"]="Unknown"
    else:
        df["Sector"]="Unknown"; df["Industry"]="Unknown"
    return df


# ═══════════════════ VALIDATION (v3.0) ══════════════════════════════════════

def validate_data(df: pd.DataFrame) -> Dict[str, float]:
    """
    Enhanced validation. Returns stats dict + prints warnings.
    Checks:
      - NaN % per column
      - Per-ticker gap analysis (days between first/last trading day)
      - AdjFactor jump detection (>50% day-over-day = likely split without adjustment)
      - Zero-volume day percentage
    """
    logger.info("Step 7: Validation")
    stats = {}
    st = df[df["Type"]=="stock"]

    # NaN percentages (skip categorical)
    for c in FINAL_COLUMNS:
        if c in st.columns and str(st[c].dtype) != "category":
            na = st[c].isna().mean() * 100
            stats[f"nan_pct_{c}"] = na
            if na > 5:
                tag = "OK" if na < 15 else ("WARN" if na < 50 else "FAIL")
                logger.info(f"  {tag:<4} {c}: {na:.1f}% NaN")

    # Per-ticker coverage
    ticker_days = st.groupby("Ticker").agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        n_rows=("Date", "count"),
    ).reset_index()
    ticker_days["calendar_days"] = (ticker_days["last_date"] - ticker_days["first_date"]).dt.days
    ticker_days["trading_days_expected"] = (ticker_days["calendar_days"] * 252.0 / 365.0).clip(lower=1)
    ticker_days["coverage_pct"] = (ticker_days["n_rows"] / ticker_days["trading_days_expected"] * 100).clip(upper=100)
    low_cov = (ticker_days["coverage_pct"] < 80).sum()
    stats["tickers_low_coverage"] = int(low_cov)
    logger.info(f"  Tickers with <80% coverage: {low_cov:,}/{len(ticker_days):,}")

    # AdjFactor anomalies
    if "AdjFactor" in st.columns:
        adj_chg = st.groupby("Ticker")["AdjFactor"].pct_change().abs()
        big_jumps = (adj_chg > 0.5).sum()
        stats["adjfactor_big_jumps"] = int(big_jumps)
        if big_jumps > 0:
            logger.info(f"  AdjFactor >50% day jumps: {big_jumps:,}")

    # Zero volume
    if "Volume" in st.columns:
        zero_vol = (st["Volume"] == 0).mean() * 100
        stats["zero_volume_pct"] = zero_vol
        if zero_vol > 1:
            logger.info(f"  Zero-volume days: {zero_vol:.2f}%")

    return stats


# ═══════════════════ FINALIZE ═══════════════════════════════════════════════

def finalize(df: pd.DataFrame, output: Path):
    logger.info("Step 8: Finalize + Save")
    for c in FINAL_COLUMNS:
        if c not in df.columns: df[c] = np.nan
    df = df[[c for c in FINAL_COLUMNS if c in df.columns]].copy()

    # Dtype assignment
    f32 = ["Open","High","Low","Close","Adj_Close","AdjFactor","RSI_14",
           "Return_20d_Pct","Return_60d_Pct","Return_120d_Pct","Return_252d_Pct",
           "Current_Daily_Return","ATR_14"]
    f64 = ["Dollar_Volume_20d_Avg","Daily_Volatility_20d",
           "VIX_Close","SPY_Close","SPY_SMA_200","HYG_Close","HYG_SMA_20",
           "Shares_Outstanding","Market_Cap"]
    for c in f32:
        if c in df.columns: df[c] = pd.to_numeric(df[c],errors="coerce").astype("float32")
    for c in f64:
        if c in df.columns: df[c] = pd.to_numeric(df[c],errors="coerce").astype("float64")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"],errors="coerce").fillna(0).astype("int64")
    for c in ("Sector","Industry","Exchange","Type","Market_Regime"):
        if c in df.columns: df[c] = df[c].astype("category")

    df = df.drop_duplicates(subset=["Ticker","Date"],keep="last").sort_values(["Ticker","Date"]).reset_index(drop=True)

    # Validation
    _ = validate_data(df)

    # Save with metadata
    _save_parquet_with_metadata(df, output, schema_version=SCHEMA_VERSION)

    mb = output.stat().st_size / 1_048_576
    ns = df[df["Type"]=="stock"]["Ticker"].nunique()
    ni = df[df["Type"]=="index"]["Ticker"].nunique()
    logger.info("="*60)
    logger.info(f"  simulation_db: {output}")
    logger.info(f"    Schema:    {SCHEMA_VERSION}")
    logger.info(f"    Rows:      {len(df):>12,}")
    logger.info(f"    Stocks:    {ns:>12,}")
    logger.info(f"    Indices:   {ni:>12,}")
    logger.info(f"    Range:     {df['Date'].min().date()} -> {df['Date'].max().date()}")
    logger.info(f"    Size:      {mb:>10.1f} MB")
    logger.info(f"    Columns:   {list(df.columns)}")
    logger.info("="*60)
    for f in [CHECKPOINT_FILE, Path("sim_db_temp.parquet")]:
        if f.exists(): f.unlink(); logger.info(f"  Cleaned {f}")


def _save_parquet_with_metadata(df: pd.DataFrame, path: Path, schema_version: str):
    """Save parquet with embedded schema_version and build_timestamp metadata."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df)
        existing_meta = dict(table.schema.metadata) if table.schema.metadata else {}
        new_meta = {
            b"schema_version": schema_version.encode(),
            b"build_timestamp": datetime.datetime.now().isoformat().encode(),
            b"n_rows": str(len(df)).encode(),
            b"n_tickers": str(df["Ticker"].nunique()).encode() if "Ticker" in df.columns else b"0",
            b"columns": ",".join(df.columns).encode(),
        }
        merged_meta = {**existing_meta, **new_meta}
        table = table.replace_schema_metadata(merged_meta)
        pq.write_table(table, path, compression="snappy")
    except Exception as e:
        logger.warning(f"  Metadata save failed ({e}); falling back to plain write")
        df.to_parquet(path, index=False, compression="snappy")


def read_parquet_metadata(path: Path) -> Dict[str, str]:
    """Read schema_version and other metadata from parquet file."""
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path).metadata or {}
        out = {}
        for k, v in meta.items():
            kd = k.decode(errors="replace")
            if kd.startswith("pandas") or kd.startswith("ARROW:"):
                continue
            try:
                out[kd] = v.decode(errors="replace")
            except Exception:
                out[kd] = str(v)
        return out
    except Exception as e:
        logger.warning(f"  Metadata read failed: {e}")
        return {}


# ═══════════════════ BENCHMARKS.PARQUET (v3.0) ══════════════════════════════

def build_benchmarks_parquet(start: str, end: str, output_path: Path):
    """
    Download SPY + QQQ + ^VIX and save to separate benchmarks.parquet.
    SPY/QQQ are ETFs — their Adj_Close includes dividend reinvestment,
    which is the fair comparison for a buy-and-hold benchmark.
    """
    import yfinance as yf

    logger.info(f"Step 9: Build benchmarks parquet")
    frames: List[pd.DataFrame] = []

    for sym in ["SPY", "QQQ"]:
        for attempt in range(3):
            try:
                raw = yf.download(sym, start=start, end=end, interval="1d",
                                  auto_adjust=False, progress=False, threads=False)
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    raw = raw.reset_index()
                    raw["Date"] = pd.to_datetime(raw["Date"])
                    if hasattr(raw["Date"].dt, "tz") and raw["Date"].dt.tz is not None:
                        raw["Date"] = raw["Date"].dt.tz_localize(None)
                    sub = raw[["Date", "Close", "Adj Close"]].rename(
                        columns={"Close": f"{sym}_Close",
                                 "Adj Close": f"{sym}_Adj_Close"}
                    )
                    frames.append(sub)
                    logger.info(f"  {sym}: {len(sub):,} rows")
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    logger.warning(f"  FAIL {sym}: {e}")
        time.sleep(random.uniform(0.5, 1.5))

    # VIX - no dividends, just Close
    for attempt in range(3):
        try:
            raw = yf.download("^VIX", start=start, end=end, interval="1d",
                              auto_adjust=False, progress=False, threads=False)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.reset_index()
                raw["Date"] = pd.to_datetime(raw["Date"])
                if hasattr(raw["Date"].dt, "tz") and raw["Date"].dt.tz is not None:
                    raw["Date"] = raw["Date"].dt.tz_localize(None)
                sub = raw[["Date", "Close"]].rename(columns={"Close": "VIX_Close"})
                frames.append(sub)
                logger.info(f"  ^VIX: {len(sub):,} rows")
                break
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                logger.warning(f"  FAIL ^VIX: {e}")

    if not frames:
        logger.error("  No benchmark data downloaded — skipping benchmarks.parquet")
        return

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="Date", how="outer")
    merged = merged.sort_values("Date").reset_index(drop=True)
    for c in merged.columns:
        if c != "Date":
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float64")

    _save_parquet_with_metadata(merged, output_path, schema_version=SCHEMA_VERSION)
    mb = output_path.stat().st_size / 1_048_576
    logger.info(f"  benchmarks: {output_path} | {len(merged):,} rows | {mb:.2f} MB | "
                f"{merged['Date'].min().date()} -> {merged['Date'].max().date()}")


# ═══════════════════ FROM-PARQUET ═══════════════════════════════════════════

def build_from_parquet(p):
    logger.info(f"Step 2: Loading {p}")
    if not p.exists(): logger.error(f"Not found: {p}"); sys.exit(1)
    df = pd.read_parquet(p); df["Date"] = pd.to_datetime(df["Date"])
    v3 = "Close" in df.columns and "AdjFactor" in df.columns
    if v3:
        df["Close"] = pd.to_numeric(df["Close"],errors="coerce").astype("float64")
        df["AdjFactor"] = pd.to_numeric(df["AdjFactor"],errors="coerce").fillna(1.0).astype("float64")
        df["Adj_Close"] = (df["Close"]*df["AdjFactor"]).round(6)
    elif "Adj Close" in df.columns:
        df["Adj_Close"] = pd.to_numeric(df["Adj Close"],errors="coerce").astype("float64")
        if "Close" not in df.columns: df["Close"] = df["Adj_Close"]
        df["AdjFactor"] = (df["Adj_Close"]/df["Close"].replace(0,np.nan)).fillna(1.0)
    else:
        logger.error("Unknown schema"); sys.exit(1)
    df["Volume"] = pd.to_numeric(df.get("Volume",0),errors="coerce").fillna(0).astype("int64")
    for c in ("Open","High","Low"):
        if c not in df.columns: df[c] = df["Close"]
        else: df[c] = pd.to_numeric(df[c],errors="coerce").fillna(df["Close"])
    if "Type" not in df.columns: df["Type"] = "stock"
    if "Exchange" not in df.columns: df["Exchange"] = "UNKNOWN"
    logger.info(f"  {len(df):,} rows | {df['Ticker'].nunique():,} tickers")
    return df


# ═══════════════════ INCREMENTAL UPDATE (v3.0) ══════════════════════════════

def incremental_update(existing_path: Path, output_path: Path,
                       sector_map: Optional[Path],
                       with_market_cap: bool,
                       skip_benchmarks: bool,
                       benchmark_output: Path):
    """
    Load existing parquet, download only delta from last date,
    recompute indicators on combined data, save.
    """
    if not existing_path.exists():
        logger.error(f"Cannot incremental update: {existing_path} not found")
        sys.exit(1)

    logger.info(f"Incremental update from {existing_path}")
    meta = read_parquet_metadata(existing_path)
    prev_version = meta.get("schema_version", "unknown")
    logger.info(f"  Existing schema: {prev_version} (target: {SCHEMA_VERSION})")

    existing = pd.read_parquet(existing_path)
    existing["Date"] = pd.to_datetime(existing["Date"])

    # Ensure string categories (can't concat if categoricals have different categories)
    for c in ("Sector","Industry","Exchange","Type","Market_Regime"):
        if c in existing.columns and str(existing[c].dtype) == "category":
            existing[c] = existing[c].astype(str)

    # Strip computed columns — will be recomputed fresh from OHLCV
    stripped_cols = [
        "Dollar_Volume_20d_Avg",
        "Return_20d_Pct", "Return_60d_Pct", "Return_120d_Pct", "Return_252d_Pct",
        "RSI_14", "ATR_14",
        "Daily_Volatility_20d", "Current_Daily_Return",
        "VIX_Close", "SPY_Close", "SPY_SMA_200", "HYG_Close", "HYG_SMA_20",
        "Market_Regime", "Shares_Outstanding", "Market_Cap",
    ]
    base = existing.drop(columns=[c for c in stripped_cols if c in existing.columns])

    # Find last date (stocks only — indices may lag)
    stock_mask = base["Type"] == "stock"
    last_stock_date = base.loc[stock_mask, "Date"].max()
    logger.info(f"  Existing data through {last_stock_date.date()} (stocks)")

    start_new = (last_stock_date + pd.Timedelta(days=1)).date()
    end_new = datetime.date.today() + datetime.timedelta(days=1)

    if start_new >= end_new:
        logger.info("  Already up to date — recomputing indicators only")
        combined = base
    else:
        logger.info(f"  Delta window: {start_new} -> {end_new}")

        stock_tickers = base.loc[stock_mask, "Ticker"].unique().tolist()
        emap = dict(zip(
            base.loc[stock_mask, "Ticker"].astype(str),
            base.loc[stock_mask, "Exchange"].astype(str)
        ))

        tmp = Path("sim_db_incr_temp.parquet")
        if tmp.exists(): tmp.unlink()

        new_stocks = download_stocks(stock_tickers, emap, str(start_new), str(end_new),
                                      tmp, resume=False)
        new_idx = download_indices(str(start_new), str(end_new))

        new_frames = []
        if not new_stocks.empty:
            new_frames.append(new_stocks)
        if new_idx is not None and not new_idx.empty:
            new_frames.append(new_idx)

        if new_frames:
            delta = pd.concat(new_frames, ignore_index=True)
            # Bring Sector/Industry from base via ticker map (for new rows that lack them)
            sec_map_d = dict(zip(base["Ticker"].astype(str),
                                 base["Sector"].astype(str))) if "Sector" in base.columns else {}
            ind_map_d = dict(zip(base["Ticker"].astype(str),
                                 base["Industry"].astype(str))) if "Industry" in base.columns else {}
            delta["Sector"]   = delta["Ticker"].map(sec_map_d).fillna("Unknown")
            delta["Industry"] = delta["Ticker"].map(ind_map_d).fillna("Unknown")

            combined = pd.concat([base, delta], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Ticker","Date"], keep="last")
        else:
            logger.warning("  No delta downloaded")
            combined = base

        if tmp.exists(): tmp.unlink()

    # Apply sector map if provided (overrides cached)
    if sector_map:
        combined = merge_sectors(combined, sector_map)

    # Recompute all indicators
    combined = compute_indicators(combined)
    combined = compute_macro_regime(combined)

    # Shares / Market Cap
    if with_market_cap:
        stock_tickers = combined[combined["Type"]=="stock"]["Ticker"].unique().tolist()
        shares_df = fetch_shares_outstanding(stock_tickers, SHARES_CACHE)
        combined = compute_market_cap(combined, shares_df)
    else:
        combined["Shares_Outstanding"] = np.nan
        combined["Market_Cap"] = np.nan

    # Save
    finalize(combined, output_path)

    # Benchmarks
    if not skip_benchmarks:
        end_date = datetime.date.today() + datetime.timedelta(days=1)
        start_date = combined["Date"].min().date()
        build_benchmarks_parquet(str(start_date), str(end_date), benchmark_output)


# ═══════════════════ MAIN ═══════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=f"Simulation DB Builder {SCHEMA_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output",           type=Path, default=OUTPUT_DEFAULT,
                   help="Main simulation_db parquet path")
    p.add_argument("--benchmark-output", type=Path, default=BENCHMARK_OUTPUT,
                   help="Benchmarks parquet path (SPY + QQQ + VIX)")
    p.add_argument("--skip-benchmarks",  action="store_true",
                   help="Don't build benchmarks.parquet")
    p.add_argument("--with-market-cap",  action="store_true",
                   help="Fetch Shares_Outstanding + compute Market_Cap (slow, ~1min/100 tickers)")
    p.add_argument("--incremental",      action="store_true",
                   help="Update existing DB with delta from last date (requires --output exists)")
    p.add_argument("--from-parquet",     type=Path, default=None,
                   help="Skip download, use existing OHLCV parquet")
    p.add_argument("--exchanges",        nargs="+", default=["NASDAQ"],
                   help="Exchanges to include (default: NASDAQ)")
    p.add_argument("--years",            type=int,  default=YEARS_BACK,
                   help=f"Years of history (default: {YEARS_BACK})")
    p.add_argument("--sector-map",       type=Path, default=None,
                   help="Parquet file with Ticker -> Sector/Industry mapping")
    p.add_argument("--batch-size",       type=int,  default=BATCH_SIZE)
    p.add_argument("--resume",           action="store_true",
                   help="Resume interrupted download from checkpoint")
    a = p.parse_args()

    logger.info("="*60)
    logger.info(f"  Simulation DB Builder {SCHEMA_VERSION}")
    logger.info(f"  Common Stocks Only | Multi-Exchange | Pre-computed")
    logger.info("="*60)

    # Incremental mode — separate code path
    if a.incremental:
        incremental_update(
            existing_path    = a.output,
            output_path      = a.output,
            sector_map       = a.sector_map,
            with_market_cap  = a.with_market_cap,
            skip_benchmarks  = a.skip_benchmarks,
            benchmark_output = a.benchmark_output,
        )
        logger.info("Incremental update done!")
        return

    # Full build
    if a.from_parquet:
        df = build_from_parquet(a.from_parquet)
        date_min = df["Date"].min()
        date_max = df["Date"].max()
        start_str = str(date_min.date())
        end_str   = str((date_max + pd.Timedelta(days=1)).date())
    else:
        tickers, emap = load_all_tickers(a.exchanges)
        syms = [t["ticker"] for t in tickers]
        end = datetime.date.today() + datetime.timedelta(days=1)
        try: start = end.replace(year=end.year - a.years)
        except ValueError: start = end.replace(year=end.year - a.years, day=28)
        logger.info(f"Step 2: Download ({start} -> {end})")
        tmp = Path("sim_db_temp.parquet")
        df = download_stocks(syms, emap, str(start), str(end), tmp, a.resume, a.batch_size)
        if df.empty: logger.error("No data!"); sys.exit(1)
        logger.info("Step 3: Download indices")
        idx = download_indices(str(start), str(end))
        if idx is not None and not idx.empty:
            df = pd.concat([df, idx], ignore_index=True)
            logger.info(f"  Added {idx['Ticker'].nunique()} indices")
        start_str, end_str = str(start), str(end)

    df = merge_sectors(df, a.sector_map)
    df = compute_indicators(df)
    df = compute_macro_regime(df)

    # Shares Outstanding / Market Cap (opt-in)
    if a.with_market_cap:
        stock_tickers = df[df["Type"]=="stock"]["Ticker"].unique().tolist()
        shares_df = fetch_shares_outstanding(stock_tickers, SHARES_CACHE)
        df = compute_market_cap(df, shares_df)
    else:
        df["Shares_Outstanding"] = np.nan
        df["Market_Cap"] = np.nan
        logger.info("Step [skip]: Market_Cap skipped (pass --with-market-cap to enable)")

    finalize(df, a.output)

    # Benchmarks file (SPY + QQQ + VIX)
    if not a.skip_benchmarks:
        build_benchmarks_parquet(start_str, end_str, a.benchmark_output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
