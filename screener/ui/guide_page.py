"""Guide page: the Hebrew user guide (screener/guide.html)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from screener.ui import maintenance_view


GUIDE = Path(__file__).resolve().parent.parent / "guide.html"


def guide_page() -> None:
    st.html(GUIDE.read_text(encoding="utf-8"))      # Hebrew only, like all end-user explanations
    maintenance_view.render()
