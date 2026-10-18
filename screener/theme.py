"""
Colors and page CSS of the screener ("ledger" look). The Streamlit widget theme is in .streamlit/config.toml;
the colors here are the same ones, for our own HTML, table styling and the Plotly heatmap.
"""
from __future__ import annotations

INK = "#1B2430"         # text
MUTED = "#6B6A63"       # secondary text, notes
PAPER = "#F7F5EF"       # page background
CARD = "#FFFFFF"        # panels: filters, detail card
RULE = "#D8D3C6"        # borders
HAIRLINE = "#ECE8DE"    # row separators
ACCENT = "#1F4E8C"      # buttons, selection, links (never red/green: those mean loss/gain)
GAIN = "#147A43"
LOSS = "#C2352B"
FLAT = "#3B4048"        # heatmap middle (no change)

CSS = f"""
<style>
  :root {{
    --hs-ink: {INK}; --hs-muted: {MUTED}; --hs-paper: {PAPER}; --hs-card: {CARD};
    --hs-rule: {RULE}; --hs-hairline: {HAIRLINE}; --hs-accent: {ACCENT};
  }}
  .block-container {{padding-top: 3.25rem; padding-bottom: 2rem; max-width: 1800px;}}  /* clear the fixed top bar */

  /* numbers line up in columns: prices, deltas, counts */
  [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stCaptionContainer"],
  .hs-count, .hs-snapshot {{font-variant-numeric: tabular-nums;}}

  /* masthead: brand, market tape, data status — closed by a ledger double rule */
  .st-key-masthead {{
    border-bottom: 3px double var(--hs-rule); padding-bottom: 0.6rem; margin-bottom: 0.4rem;
  }}
  .hs-brand {{display: flex; align-items: baseline; gap: 0.45rem; line-height: 1.1;}}
  .hs-brand b {{font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; color: var(--hs-ink);}}
  .hs-brand span {{font-size: 1.7rem; font-weight: 300; color: var(--hs-accent);}}
  .hs-brand i {{
    width: 0.62rem; height: 0.62rem; background: var(--hs-accent); align-self: center;
    transform: rotate(45deg); flex: none;
  }}
  .st-key-tape [data-testid="stColumn"] + [data-testid="stColumn"] {{
    border-inline-start: 1px solid var(--hs-hairline); padding-inline-start: 0.9rem;
  }}
  .st-key-tape [data-testid="stMetricLabel"] p {{
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; color: var(--hs-muted);
  }}

  /* filters panel and the stock detail card sit on white paper */
  [data-testid="stExpander"] details, .st-key-detail {{
    background: var(--hs-card); border-color: var(--hs-rule);
  }}
  .st-key-detail {{border: 1px solid var(--hs-rule); border-radius: 0.5rem; padding: 0.9rem 1rem;}}
  [data-testid="stExpander"] summary p {{font-weight: 600;}}
  [data-baseweb="tab-list"] button p {{font-weight: 500;}}

  /* results header */
  .hs-count {{font-size: 1.55rem; font-weight: 700; color: var(--hs-ink); letter-spacing: -0.01em;
             text-align: start;}}
  .hs-count small {{font-size: 0.95rem; font-weight: 400; color: var(--hs-muted); margin-inline-start: 0.35rem;}}

  .hs-note {{font-size: 0.85rem; color: var(--hs-muted);}}
  [data-testid="stDataFrame"] {{border: 1px solid var(--hs-rule); border-radius: 0.4rem; background: var(--hs-card);}}
  hr {{border-color: var(--hs-hairline) !important;}}
</style>
"""

# Hebrew: the interface runs right-to-left; tables, the heatmap, the chart and code stay left-to-right
# (Streamlit's table can only pin columns on the left, and numbers/tickers read left-to-right).
RTL_CSS = """
<style>
  .block-container, [data-baseweb="popover"], [data-testid="stTooltipContent"],
  [data-testid="stToast"] {direction: rtl; text-align: right;}
  [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stIFrame"],
  [data-testid="stCode"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {direction: ltr;}
  [data-testid="stMetricValue"] {text-align: right;}
  .hs-brand {direction: ltr; justify-content: flex-end;}
</style>
"""
