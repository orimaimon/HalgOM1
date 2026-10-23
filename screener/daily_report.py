"""
Daily report module.

Composes the daily Hebrew report (text and Telegram MarkdownV2) from Morning, Regime,
and Fired alert data.

No Streamlit, no network, no file I/O.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from screener import i18n
from screener.alerts import Fired
from screener.home import Morning
from screener.market import Regime

TELEGRAM_LIMIT = 4096       # characters per message
SPECIAL_CHARS = set(r"_*[]()~`>#+-=|{}.!\\")


def escape_markdown_v2(s: str) -> str:
    """Escape special characters for Telegram MarkdownV2 format."""
    if not s:
        return ""
    res = []
    for ch in str(s):
        if ch in SPECIAL_CHARS:
            res.append("\\" + ch)
        else:
            res.append(ch)
    return "".join(res)


@dataclass
class Report:
    title: str
    sections: list[tuple[str, list[str]]]
    text: str
    telegram: str


def compose(
    morning: Morning,
    regime: Regime | None,
    fired: list[Fired] | None,
    day: dt.date,
    top_screens: int = 5,
    top_movers: int = 5,
) -> Report:
    """
    Compose the daily Hebrew report (plain text and Telegram MarkdownV2).

    Movers with NaN or missing change values are skipped.
    """
    title = f"הבוקר שלי · {day.strftime('%d/%m/%Y')}"
    footer = "נתונים בעיכוב של כ-15 דקות. אינו ייעוץ השקעות."

    sections: list[tuple[str, list[str]]] = []

    # 1. Market section
    if regime is not None:
        label_raw = (regime.label or "").upper()
        label_he = i18n.ui(label_raw, i18n.HE)
        reasons_he = []
        if regime.reasons:
            reasons_he = [i18n.regime_reason(r, i18n.HE) for r in regime.reasons[:2] if r]

        if reasons_he:
            market_item = f"{label_he} ({', '.join(reasons_he)})"
        else:
            market_item = label_he

        sections.append(("מצב השוק", [market_item]))

    # 2. New in screens
    df_screens = morning.new_in_screens if morning is not None and hasattr(morning, "new_in_screens") else None

    if df_screens is not None and not df_screens.empty:
        screen_items: list[str] = []
        valid_rows = []
        for _, row in df_screens.iterrows():
            new_val = row.get("new")
            if pd.notna(new_val) and new_val is not None:
                try:
                    new_int = int(new_val)
                    if new_int > 0:
                        total_val = row.get("total", 0)
                        total_int = int(total_val) if pd.notna(total_val) and total_val is not None else 0
                        valid_rows.append((row.get("screen", ""), new_int, total_int))
                except (ValueError, TypeError):
                    pass

        valid_rows.sort(key=lambda x: x[1], reverse=True)
        top_rows = valid_rows[:top_screens]

        for name, new_cnt, total_cnt in top_rows:
            if name.startswith("★ "):
                disp_name = name
            else:
                disp_name = i18n.preset(name, i18n.HE)
            screen_items.append(f"{disp_name}: {new_cnt} חדשות (מתוך {total_cnt})")

        if not screen_items:
            screen_items = ["אין מניות חדשות בסריקות"]

        sections.append(("חדשות בסריקות", screen_items))

    # 3. Movers
    mover_items: list[str] = []
    df_movers = morning.movers if morning is not None and hasattr(morning, "movers") else None
    if df_movers is not None and not df_movers.empty:
        count = 0
        for _, row in df_movers.iterrows():
            if count >= top_movers:
                break
            chg = row.get("change")
            if pd.isna(chg) or chg is None:
                continue
            try:
                chg_float = float(chg)
                if math.isnan(chg_float):
                    continue
            except (ValueError, TypeError):
                continue

            ticker = str(row.get("ticker", "")).strip()
            if not ticker:
                continue

            lists = row.get("lists")
            lists_str = ""
            if isinstance(lists, (list, tuple)) and lists:
                valid_lists = [str(l) for l in lists if l]
                if valid_lists:
                    lists_str = f" ({', '.join(valid_lists)})"

            chg_str = f"{chg_float:+.1f}%"
            mover_items.append(f"{ticker} {chg_str}{lists_str}")
            count += 1

    if mover_items:
        sections.append(("תנועות בולטות", mover_items))

    # 4. Earnings
    earnings_items: list[str] = []
    df_earnings = morning.earnings if morning is not None and hasattr(morning, "earnings") else None
    if df_earnings is not None and not df_earnings.empty:
        for _, row in df_earnings.iterrows():
            ticker = str(row.get("ticker", "")).strip()
            if not ticker:
                continue
            days_raw = row.get("days")
            if pd.isna(days_raw) or days_raw is None:
                continue
            try:
                days = int(days_raw)
            except (ValueError, TypeError):
                continue

            edate_raw = row.get("earnings_date")
            date_str = ""
            if pd.notna(edate_raw) and edate_raw is not None:
                if isinstance(edate_raw, (dt.date, dt.datetime, pd.Timestamp)):
                    date_str = edate_raw.strftime("%d/%m")
                else:
                    try:
                        date_str = pd.to_datetime(edate_raw).strftime("%d/%m")
                    except Exception:
                        date_str = str(edate_raw)

            if days == 0:
                days_label = "היום"
            elif days == 1:
                days_label = "מחר"
            else:
                days_label = f"בעוד {days} ימים"

            if date_str:
                item_str = f"{ticker} — דוח {days_label} ({date_str})"
            else:
                item_str = f"{ticker} — דוח {days_label}"
            earnings_items.append(item_str)

    if earnings_items:
        sections.append(("דוחות קרובים", earnings_items))

    # 5. Alerts
    alert_items: list[str] = []
    fired_list = fired if fired is not None else (morning.alerts if morning is not None and hasattr(morning, "alerts") else [])
    if fired_list:
        for f in fired_list:
            msg = getattr(f, "message", None)
            if not msg and isinstance(f, str):
                msg = f
            if msg:
                alert_items.append(str(msg))

    if alert_items:
        sections.append(("התראות", alert_items))

    text = _build_text(title, sections, footer)
    telegram = _build_telegram(title, sections, footer)

    return Report(
        title=title,
        sections=sections,
        text=text,
        telegram=telegram,
    )


def _build_text(title: str, sections: list[tuple[str, list[str]]], footer: str) -> str:
    lines = [title]
    if not sections:
        lines.append("")
        lines.append("אין חדש היום")
        lines.append("")
        lines.append(footer)
        return "\n".join(lines)

    for heading, items in sections:
        lines.append("")
        lines.append(heading)
        for item in items:
            lines.append(f"• {item}")

    lines.append("")
    lines.append(footer)
    return "\n".join(lines)


def _build_telegram(title: str, sections: list[tuple[str, list[str]]], footer: str) -> str:
    if not sections:
        lines = [
            f"*{escape_markdown_v2(title)}*",
            "",
            "אין חדש היום",
            "",
            escape_markdown_v2(footer),
        ]
        return "\n".join(lines)

    sec_items = [list(items) for _, items in sections]
    dropped_counts = [0] * len(sections)

    while True:
        rendered = _render_telegram_msg(title, sections, sec_items, dropped_counts, footer)
        if len(rendered) <= TELEGRAM_LIMIT:
            return rendered

        max_idx = -1
        max_len = -1
        for i, items_list in enumerate(sec_items):
            if len(items_list) > max_len and len(items_list) > 0:
                max_len = len(items_list)
                max_idx = i

        if max_idx == -1 or max_len <= 0:
            return _hard_cut(rendered)

        sec_items[max_idx].pop()
        dropped_counts[max_idx] += 1


def _hard_cut(text: str, limit: int = TELEGRAM_LIMIT) -> str:
    """Last resort when one item alone is too long: cut, never leaving a lone escaping backslash."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    trailing = len(cut) - len(cut.rstrip("\\"))
    if trailing % 2:
        cut = cut[:-1]
    return cut + "…"


def _render_telegram_msg(
    title: str,
    sections: list[tuple[str, list[str]]],
    sec_items: list[list[str]],
    dropped_counts: list[int],
    footer: str,
) -> str:
    lines = [f"*{escape_markdown_v2(title)}*"]
    for i, (heading, _) in enumerate(sections):
        lines.append("")
        lines.append(f"*{escape_markdown_v2(heading)}*")
        items_to_show = sec_items[i]
        for item in items_to_show:
            lines.append(f"• {escape_markdown_v2(item)}")
        if dropped_counts[i] > 0:
            drop_msg = f"…ועוד {dropped_counts[i]}"
            lines.append(f"• {escape_markdown_v2(drop_msg)}")

    lines.append("")
    lines.append(escape_markdown_v2(footer))
    return "\n".join(lines)
