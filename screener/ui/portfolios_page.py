"""Portfolios page: watchlists and portfolios, import, trades.

Moved verbatim from screener/app.py; behavior unchanged."""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import streamlit as st

from screener import export, i18n as I, portfolio as PF

from screener.ui.common import _sign_color, col_label, hebrew_note, log, on_reset, ss, t


# ─────────────────────────── Portfolios page ───────────────────────────────

NEW_WATCHLIST = "— New watchlist —"


def _by_ticker(df: pd.DataFrame, col: str = "price") -> pd.Series:
    """ticker -> a universe column (prices come from the screener's data: no extra requests)."""
    return df.drop_duplicates("ticker").set_index("ticker")[col]


def _money(v: float) -> str:
    return "—" if pd.isna(v) else f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def on_pf_import():
    imp, name = ss.get("pf_pending"), (ss.get("pf_new_name") or "").strip()
    if imp is None or not imp.ok or not name:
        return
    try:
        PF.save(PF.from_imported(name, imp, _by_ticker(ss.get("df"))))
    except Exception as e:
        log.warning("cannot save portfolio %r: %s", name, e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    ss["pf_name"], ss["pf_new_name"] = name, ""
    ss["pf_upload"] = ss.get("pf_upload", 0) + 1          # a new uploader widget: the file is cleared
    st.toast(f"{t('List saved')}: “{name}”", icon="💾")


def on_pf_new_watchlist():
    name = (ss.get("pf_new_name") or "").strip()
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load lists: %s", e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    if not name or name in lists:
        return
    try:
        PF.save(PF.PList(name, PF.WATCHLIST))
    except Exception as e:
        log.warning("cannot save watchlist %r: %s", name, e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    ss["pf_name"], ss["pf_new_name"] = name, ""


def add_tickers(plist: PF.PList, text: str, prices: pd.Series) -> tuple[list[str], list[str]]:
    """'aapl, msft nvda' -> (added, not in the screener). Tickers already in the list are skipped."""
    added, unknown = [], []
    for tk in filter(None, (PF.normalize_ticker(x) for x in re.split(r"[,;\s]+", text))):
        if tk not in prices.index:
            unknown.append(tk)
        elif PF.add_to_watchlist(plist, tk, prices[tk]):
            added.append(tk)
    return added, unknown


def _add_and_report(plist: PF.PList, text: str) -> None:
    added, unknown = add_tickers(plist, text, _by_ticker(ss.get("df")))
    try:
        PF.save(plist)
    except Exception as e:
        log.warning("cannot save list %r: %s", plist.name, e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    if added:
        st.toast(f"{t('Added to')} “{plist.name}”: {', '.join(added)}", icon="✅")
    if unknown:
        st.toast(f"{t('Not found in screener')}: {', '.join(unknown)}", icon="⚠️")


def on_pf_add_tickers():
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load list: %s", e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    name = ss.get("pf_name")
    if not name or name not in lists:
        st.toast(t("The list was deleted in another window"), icon="⚠️")
        return
    plist = lists[name]
    if plist is not None:
        _add_and_report(plist, ss.get("pf_add") or "")
    ss["pf_add"] = ""


def on_add_to_watchlist(ticker: str):
    """From the stock details panel."""
    name = ss.get("add_wl")
    if name == NEW_WATCHLIST:
        name = (ss.get("add_wl_name") or "").strip()
        if not name:
            return
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load lists: %s", e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    plist = lists.get(name) or PF.PList(name, PF.WATCHLIST)
    if plist.kind != PF.WATCHLIST:
        st.toast(f"“{name}” {t('is a portfolio, not a watchlist')}", icon="⚠️")
        return
    _add_and_report(plist, ticker)
    ss["add_wl"], ss["add_wl_name"] = name, ""


def on_pf_remove():
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load list: %s", e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    name = ss.get("pf_name")
    if not name or name not in lists:
        st.toast(t("The list was deleted in another window"), icon="⚠️")
        return
    plist = lists[name]
    if plist is not None:
        plist.rows = [r for r in plist.rows if r["ticker"] != ss.get("pf_remove")]
        try:
            PF.save(plist)
        except Exception as e:
            log.warning("cannot save list: %s", e)
            st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
            return


def on_pf_trade():
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load list: %s", e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    name = ss.get("pf_name")
    if not name or name not in lists:
        st.toast(t("The list was deleted in another window"), icon="⚠️")
        return
    plist = lists[name]
    if plist is None:
        return
    raw = pd.DataFrame([{"ticker": ss.get("pf_t_ticker") or "", "date": ss.get("pf_t_date"),
                         "price": ss.get("pf_t_price"), "quantity": ss.get("pf_t_qty"),
                         "side": ss.get("pf_t_side"), "fees": ss.get("pf_t_fees") or 0.0,
                         "note": ss.get("pf_t_note") or ""}])
    imp = PF.parse(raw)
    errors = [I.problem(PF.Problem(0, p.code, p.value)) for p in imp.problems]
    if not errors:
        errors = [I.problem(PF.Problem(0, "oversold", tk)) for tk in PF.add_trades(plist, imp.rows)]
    if errors:
        ss["pf_trade_errors"] = errors
        return
    try:
        PF.save(plist)
    except Exception as e:
        log.warning("cannot save trade: %s", e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    ss["pf_trade_errors"] = []
    for k in ("pf_t_ticker", "pf_t_price", "pf_t_qty", "pf_t_fees", "pf_t_note"):
        ss.pop(k, None)
    st.toast(f"{t('Trade added')}: {imp.rows['ticker'].iloc[0]}", icon="✅")


def on_pf_delete():
    name = ss.get("pf_name")
    if name:
        try:
            PF.delete(name)
        except Exception as e:
            log.warning("cannot delete portfolio %r: %s", name, e)
            st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
            return
        ss.pop("pf_name", None)


def on_show_in_screener(name: str, tickers: list[str]):
    on_reset()                                    # filters would hide part of the list
    ss["hide_junk"] = False                          # ... and so would Hide junk ("Only new" is off after the reset)
    query = ", ".join(tickers) + ("," if len(tickers) == 1 else "")    # a comma = exact tickers, even one
    ss["query"] = query
    ss["list_view"] = (name, query)                  # the banner shows while this search is on
    ss["page"] = "Screener"


def import_panel(lists: dict[str, PF.PList], df: pd.DataFrame) -> None:
    with st.expander(t("New list / import a file"), expanded=not lists, icon=":material/upload_file:"):
        hebrew_note("קובץ CSV או אקסל, שורה לכל עסקה: סימול, תאריך, שער, כמות, ואפשר גם פעולה (קנייה/מכירה), "
                    "עמלות והערה. קובץ עם עמודת סימולים בלבד נשמר כרשימת מעקב. הכותרות יכולות להיות בעברית. "
                    "כדי לערוך רשימה קיימת: מייצאים אותה, משנים באקסל ומייבאים שוב באותו שם.")
        c1, c2 = st.columns([1, 1.4], vertical_alignment="bottom")
        name = c1.text_input(t("List name"), key="pf_new_name").strip()
        up = c2.file_uploader(t("CSV or Excel file"), type=["csv", "xlsx", "xls"],
                              key=f"pf_file_{ss.get('pf_upload', 0)}")
        ss["pf_pending"] = None
        if up is not None:
            try:
                imp = PF.parse(PF.read_table(up.getvalue(), up.name))
            except Exception as e:
                log.warning("cannot read %s: %s", up.name, e)
                st.error(t("Cannot read file. Save it as CSV or xlsx in Excel and try again."))
                imp = None
            if imp is not None and imp.problems:
                st.error(t("Fix errors in the file and load again:") + "\n\n" + "\n".join(f"- {I.problem(p)}" for p in imp.problems[:15])
                         + (f"\n- ועוד {len(imp.problems) - 15}" if len(imp.problems) > 15 else ""))
            elif imp is not None and imp.rows.empty:
                st.error(t("File is empty"))
            elif imp is not None:
                ss["pf_pending"] = imp
                n_tk = imp.rows["ticker"].nunique()
                kind_str = t("Portfolio") if imp.kind == PF.PORTFOLIO else t("Watchlist")
                details = [f"סוג: **{kind_str}**"]
                if imp.kind == PF.PORTFOLIO:
                    details.append(f"פורמט תאריך: **{imp.date_format}**")
                    dates = imp.rows["date"].dropna()
                    if not dates.empty:
                        d_min, d_max = dates.min(), dates.max()
                        details.append(f"טווח תאריכים: **{d_min:%d/%m/%Y} עד {d_max:%d/%m/%Y}**")
                    details.append(f"עסקאות: **{len(imp.rows)}** ב-**{n_tk}** מניות")
                else:
                    details.append(f"מניות: **{n_tk}**")

                st.success(" · ".join(details))
                st.caption("תצוגה מקדימה (5 שורות ראשונות):")
                preview_cols = [c for c in PF.COLUMNS if c in imp.rows.columns and not imp.rows[c].isna().all()]
                st.dataframe(imp.rows[preview_cols].head(5), hide_index=True)

                unknown = sorted(set(imp.rows["ticker"]) - set(df["ticker"]))
                if unknown:
                    hebrew_note(f"אין בסקרינר מחיר ל: {', '.join(unknown)} (למשל קרנות סל). "
                                "הן יישמרו, אבל בלי שווי נוכחי.")
        if name and name in lists:
            hebrew_note(f"{t('There is already a list named')} “{name}”. {t('Saving with a file will replace it.')}")
        b1, b2, b3 = st.columns([1, 1, 1])
        b1.button(t("Save list"), on_click=on_pf_import, type="primary", use_container_width=True,
                  disabled=not (name and ss.get("pf_pending") is not None), key="pf_save")
        b2.button(t("Empty watchlist"), on_click=on_pf_new_watchlist, use_container_width=True,
                  disabled=not name or name in lists, key="pf_empty", icon=":material/add:")
        b3.download_button(t("Sample file"), PF.template_csv(), "halgom_portfolio_sample.csv", mime="text/csv",
                           on_click="ignore", use_container_width=True, icon=":material/download:")


def watchlist_view(plist: PF.PList, df: pd.DataFrame) -> None:
    w = PF.watch_performance(plist.frame(), _by_ticker(df))
    if w.empty:
        hebrew_note(t("The list is empty. Add tickers in the box below, or from the \"Add to watchlist\" button in the stock details in the screener."))
    else:
        w["company"] = w["ticker"].map(_by_ticker(df, "company"))
        w["change"] = w["ticker"].map(_by_ticker(df, "change"))
        cols = ["ticker", "company", "date", "price", "current", "change", "since_added_pct", "note"]
        cc = st.column_config
        config = {"ticker": cc.TextColumn(t("Ticker"), pinned=True, width="small"),
                  "company": cc.TextColumn(col_label("company"), width="medium"),
                  "date": cc.DateColumn(t("Added"), format="DD/MM/YYYY"),
                  "price": cc.NumberColumn(t("Price then"), format="%.2f"),
                  "current": cc.NumberColumn(t("Price"), format="%.2f"),
                  "change": cc.NumberColumn(col_label("change"), format="%.2f%%"),
                  "since_added_pct": cc.NumberColumn(t("Since added"), format="%.2f%%"),
                  "note": cc.TextColumn(t("Notes"))}
        st.dataframe(w[cols].style.map(_sign_color, subset=["change", "since_added_pct"]), hide_index=True,
                     column_config=config, placeholder="—", key="pf_watch_table", height=min(38 + 35 * len(w), 600))
    c1, c2, c3 = st.columns([2.4, 1, 1.4], vertical_alignment="bottom")
    c1.text_input(t("Add tickers"), key="pf_add", placeholder="AAPL, MSFT")
    c2.button(t("Add"), on_click=on_pf_add_tickers, use_container_width=True, key="pf_add_btn")
    if not w.empty:
        with c3.popover(t("Remove ticker"), icon=":material/remove:", use_container_width=True):
            st.selectbox(t("Ticker"), list(w["ticker"]), key="pf_remove")
            st.button(t("Remove"), on_click=on_pf_remove, key="pf_remove_btn")


def portfolio_view(plist: PF.PList, df: pd.DataFrame) -> None:
    trades = plist.frame()
    val = PF.valuation(PF.positions(trades), _by_ticker(df))
    val["company"] = val["ticker"].map(_by_ticker(df, "company"))
    val["change"] = val["ticker"].map(_by_ticker(df, "change"))
    held = val[val["quantity"] > 0]
    priced = held[held["price"].notna()]
    value, cost = priced["value"].sum(), priced["cost"].sum()
    today = (priced["value"] * priced["change"] / (100 + priced["change"])).sum()
    cc = st.column_config
    m = st.columns(4)
    m[0].metric(t("Market value"), _money(value), f"{_money(today)} {t('today')}" if len(priced) else None)
    m[1].metric(t("Cost"), _money(cost))
    m[2].metric(t("Unrealized P&L"), _money(value - cost),
                f"{(value - cost) / cost * 100:+.2f}%" if cost > 0 else None)
    m[3].metric(t("Realized P&L"), _money(val["realized"].sum()))
    missing = held.loc[held["price"].isna(), "ticker"].tolist()
    if missing:
        hebrew_note(f"{t('No price in screener for:')} {', '.join(missing)} {t('(e.g. ETFs). They are not included in the value and summaries.')}")
    if held.empty:
        hebrew_note(t("There are currently no open positions in the portfolio."))
    else:
        cols = ["ticker", "company", "quantity", "avg_cost", "price", "change", "value", "pnl", "pnl_pct",
                "weight", "annual_pct", "first_buy"]
        config = {"ticker": cc.TextColumn(t("Ticker"), pinned=True, width="small"),
                  "company": cc.TextColumn(col_label("company"), width="medium"),
                  "quantity": cc.NumberColumn(t("Quantity"), format="localized"),
                  "avg_cost": cc.NumberColumn(t("Avg cost"), format="%.2f"),
                  "price": cc.NumberColumn(t("Price"), format="%.2f"),
                  "change": cc.NumberColumn(col_label("change"), format="%.2f%%"),
                  "value": cc.NumberColumn(t("Market value"), format="dollar"),
                  "pnl": cc.NumberColumn(t("P&L"), format="dollar"),
                  "pnl_pct": cc.NumberColumn(t("P&L %"), format="%.2f%%"),
                  "weight": cc.ProgressColumn(t("Weight"), min_value=0, max_value=100, format="%.1f%%"),
                  "annual_pct": cc.NumberColumn(t("Annual return"), format="%.2f%%",
                                                help="תשואה שנתית ממוצעת לפי תאריך הקנייה הממוצע. מוצגת רק "
                                                     "אחרי 90 יום, כי לפני כן המספר מטעה"),
                  "first_buy": cc.DateColumn(t("First buy"), format="DD/MM/YYYY")}
        st.dataframe(held[cols].style.map(_sign_color, subset=["change", "pnl", "pnl_pct", "annual_pct"]),
                     hide_index=True, column_config=config, placeholder="—", key="pf_positions",
                     height=min(38 + 35 * len(held), 600))
    hebrew_note(t("Average cost includes fees. Splits and dividends after purchase are not yet calculated."))
    with st.expander(f"{t('Trades')} ({len(trades)})"):
        shown = trades.assign(side=trades["side"].map(t))
        st.dataframe(shown, hide_index=True, placeholder="—", key="pf_trades", column_config={
            "ticker": t("Ticker"), "date": cc.DateColumn(t("Date"), format="DD/MM/YYYY"),
            "price": cc.NumberColumn(t("Price"), format="%.2f"), "quantity": t("Quantity"),
            "side": t("Side"), "fees": cc.NumberColumn(t("Fees"), format="%.2f"), "note": t("Notes")})
    with st.popover(t("Add trade"), icon=":material/add:"):
        with st.form("pf_trade", border=False):
            c1, c2 = st.columns(2)
            c1.text_input(t("Ticker"), key="pf_t_ticker")
            c2.date_input(t("Date"), key="pf_t_date", value=dt.date.today(), max_value=dt.date.today(),
                          format="DD/MM/YYYY")
            c1.number_input(t("Price"), key="pf_t_price", min_value=0.0, value=None, step=0.01, format="%.2f")
            c2.number_input(t("Quantity"), key="pf_t_qty", min_value=0.0, value=None, step=1.0)
            c1.radio(t("Side"), ["buy", "sell"], key="pf_t_side", format_func=t, horizontal=True)
            c2.number_input(t("Fees"), key="pf_t_fees", min_value=0.0, value=0.0, step=0.5)
            st.text_input(t("Notes"), key="pf_t_note")
            st.form_submit_button(t("Add trade"), on_click=on_pf_trade, type="primary")
        for e in ss.get("pf_trade_errors") or []:
            st.error(e)


def portfolios_page(df: pd.DataFrame) -> None:
    lists = PF.load_all()
    import_panel(lists, df)
    if not lists:
        hebrew_note("עוד אין רשימות. אפשר לייבא קובץ מהברוקר או מאקסל, ליצור רשימת מעקב ריקה, "
                    "או להוסיף מניה מפרטי המניה בסורק.")
        return
    names = list(lists)
    if ss.get("pf_name") not in names:
        ss["pf_name"] = names[0]
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2], vertical_alignment="bottom")
    c1.selectbox(t("List"), names, key="pf_name",
                 format_func=lambda n: f"{n} · {t('Portfolio' if lists[n].kind == PF.PORTFOLIO else 'Watchlist')}")
    plist = lists[ss.get("pf_name")]
    tickers = list(dict.fromkeys(r["ticker"] for r in plist.rows))
    c2.button(t("Show in screener"), on_click=on_show_in_screener, args=(plist.name, tickers), disabled=not tickers,
              use_container_width=True, icon=":material/search:", key="pf_show",
              help="פותח את הסורק עם המניות של הרשימה בלבד (הפילטרים מתאפסים)")
    c3.download_button(t("Export list"), PF.to_csv(plist), f"{export.file_stem(plist.name)}.csv", mime="text/csv",
                       on_click="ignore", use_container_width=True, icon=":material/download:",
                       help="קובץ CSV באותו מבנה של הייבוא: גיבוי, עריכה באקסל או העברה למחשב אחר")
    with c4.popover(t("Delete list"), icon=":material/delete:", use_container_width=True):
        hebrew_note(f"{t('Delete the list')} “{plist.name}”? {t('This cannot be undone. Consider exporting first.')}")
        st.button(t("Delete"), on_click=on_pf_delete, type="primary", key="pf_delete")
    if plist.kind == PF.PORTFOLIO:
        portfolio_view(plist, df)
    else:
        watchlist_view(plist, df)
    hebrew_note(t("Lists are saved on this computer only. Prices are from the screener (delayed ~15 min)."))
