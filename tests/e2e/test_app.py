from __future__ import annotations

from playwright.sync_api import Page, expect


def test_app_loads(page: Page, streamlit_app: str):
    page.goto(streamlit_app)
    app_container = page.locator("[data-testid='stAppViewContainer']")
    expect(app_container).to_be_visible(timeout=15000)

    brand = page.locator(".hs-brand")
    expect(brand).to_be_visible()
    expect(brand).to_contain_text("HalgOM")
    expect(brand).to_contain_text("Screener")


def test_app_navigation(page: Page, streamlit_app: str):
    page.goto(streamlit_app)
    app_container = page.locator("[data-testid='stAppViewContainer']")
    expect(app_container).to_be_visible(timeout=15000)

    nav_radio = page.get_by_role("radiogroup", name="Page")
    expect(nav_radio).to_be_visible()

    screener_option = page.get_by_role("radio", name="סורק")
    screener_option.click()

    expect(page.get_by_text("איפוס").first).to_be_visible()
