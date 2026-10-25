from playwright.sync_api import Page, expect


def test_app_boots(page: Page, streamlit_app: str):
    page.goto(streamlit_app)
    page.wait_for_selector(".hs-brand", timeout=15000)
    expect(page.locator(".hs-brand")).to_be_visible()
