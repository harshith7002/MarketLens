"""Capture one screenshot per dashboard tab for the README.

Usage:
    streamlit run app.py            # in one terminal
    python scripts/take_screenshots.py [--url http://localhost:8501]

Uses Playwright with the Microsoft Edge (or Chrome) already installed on the
machine, so no browser download is needed. Writes docs/screenshots/*.png.
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
TABS = [  # (tab label, text that appears once the tab has rendered, file name)
    ("Risk", "Growth and drawdowns", "dashboard_risk.png"),
    ("VaR", "Backtest: did the model", "dashboard_var.png"),
    ("Options", "Delta-hedging a short call", "dashboard_options.png"),
    ("Pairs Trading", "Engle-Granger on the formation period", "dashboard_pairs.png"),
    ("Rates & Credit", "Credit: spreads, default probabilities", "dashboard_rates_credit.png"),
]


def main(url: str, channel: str):
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(url)
        page.get_by_text("MarketLens — Markets Risk").wait_for(timeout=60_000)
        for label, ready_text, fname in TABS:
            page.get_by_role("tab", name=label).click()
            page.get_by_text(ready_text).first.wait_for(timeout=180_000)
            # wait until Streamlit has finished running the script ("Running..." widget gone)
            page.wait_for_function(
                "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180_000)
            page.wait_for_timeout(2500)  # let plotly finish drawing
            # Streamlit scrolls an inner container, so grow the viewport to the content height
            h = page.evaluate("() => document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
            page.set_viewport_size({"width": 1400, "height": min(int(h) + 40, 4000)})
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / fname))
            page.set_viewport_size({"width": 1400, "height": 900})
            print("saved", OUT / fname, f"({h}px)")
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge", help="msedge or chrome")
    a = ap.parse_args()
    main(a.url, a.channel)
