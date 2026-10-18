"""
HalgOM Screener — a local, Finviz-style stock screener.

Layout:
  fields.py        column registry (source field, label, format, Hebrew tooltip) + table views
  filters.py       Finviz-style dropdown filters built on the column registry
  derive.py        provider-independent derived columns (SMA distance, 52W range, ...)
  scores.py        percentile scores: Value / Quality / Momentum / Composite
  signals.py       Finviz-style signals (candles, breakouts, crosses, volume) + the "Signals" summary column
  history.py       "New since yesterday" against the previous daily snapshot
  groups.py        sector / industry performance and the market heatmap (plotly treemap)
  presets.py       built-in and user-saved screens
  export.py        Excel / CSV / TradingView watchlist export
  market.py        market regime (same rules as DB/build_simulation_db.py)
  data/            data providers + local cache / daily snapshots / offline fallback
  app.py           Streamlit UI  (run via run_screener.bat)
"""
