"""
AppDriver — a deep Page-Object adapter over the Playwright Page.

This is the canonical interaction layer for UI (E2E) tests. It encapsulates
the knowledge of how the Gradio DOM renders our app and how to synchronize
with its reactive updates, so that test files read as intent + assertion
rather than locator chains and timeout arithmetic.

Design rules (see the plan in the PR that introduced this file):
- **ui_locators.py is the single source of locator truth.** This module
  *consumes* ``Selectors`` / ``Labels``; it never re-encodes them.
- **Poll, don't snapshot.** LogViewer refreshes asynchronously via a 0.5s
  ``gr.Timer`` (ui/components/log_viewer.py). ``expect_log`` therefore uses
  ``to_contain_text`` (a polling assertion) and auto-opens the logs accordion.
  ``expect_log_equals`` is the lone exact-value variant (clear-logs).
- **The proven free-function bodies from conftest.py live here now.**
  ``goto_app`` / ``navigate`` / ``open_accordion`` were lifted from
  ``wait_for_app_ready`` / ``switch_to_tab`` / ``open_accordion`` rather than
  rewritten; they already handled Gradio 5's DOM quirks.

Escape hatch: tests that need raw Playwright (console listeners, screenshot
captures, slider keyboard input, gallery img-counts) use ``driver.page`` /
``driver.locator()``. Anything that touches the pipeline or status surface
goes through the driver methods.
"""

from __future__ import annotations

from os import environ
from typing import Pattern, Union

from playwright.sync_api import Locator, Page, expect

from .ui_locators import Labels, Selectors

# A status/log matcher may be plain text or a compiled regex.
TextMatch = Union[str, Pattern[str]]

# ---------------------------------------------------------------------------
# Server location — the driver owns "where the app lives".
# (Previously free-floating constants in conftest.py; centralized here so
# tests import one symbol — the driver — instead of a grab-bag of helpers.)
# ---------------------------------------------------------------------------

# Isolated port range to avoid collisions with the real app (7860).
# xdist workers get distinct ports: gw0 -> 8765, gw1 -> 8766, ...
BASE_PORT = 8765


def get_test_port() -> int:
    """Resolve the per-worker mock-server port for parallel runs."""
    worker_id = environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        try:
            worker_num = int("".join(filter(str.isdigit, worker_id)))
            return BASE_PORT + worker_num
        except ValueError:
            pass
    return BASE_PORT


PORT = get_test_port()
BASE_URL = f"http://127.0.0.1:{PORT}"


class AppDriver:
    """Adapter wrapping a Playwright ``Page`` bound to the mock Gradio app.

    Construct one per test (see the ``app_driver`` fixture in conftest.py) and
    drive the UI through its verbs. The driver is intentionally stateless
    beyond holding ``page``; all Gradio-side state lives in the app.
    """

    def __init__(self, page: Page):
        self.page = page

    # -- locator passthrough -------------------------------------------------

    def locator(self, selector: str) -> Locator:
        """Raw locator escape hatch for assertions the driver doesn't model."""
        return self.page.locator(selector)

    # -- navigation & app lifecycle -----------------------------------------

    def goto_app(self) -> "AppDriver":
        """Load the app and block until it is fully hydrated and reset to idle.

        Lifted from the former ``wait_for_app_ready`` free function. Sequence:
        1. initial Gradio loading indicators hidden
        2. main heading visible (app-shell hydration proxy)
        3. unified status attached
        4. mock-only Reset button clicked, idleness confirmed

        Reset is mandatory: the mock server is session-scoped across every UI
        test in a worker, so each test must start from a clean app state.
        """
        self.page.goto(BASE_URL)

        # 1. Wait out the initial JS bundle load + hydration gap.
        self.page.wait_for_selector(".generating, .loading, [data-testid='loading']", state="hidden", timeout=20000)

        # 2. Heading visible => shell hydrated.
        expect(self.page.get_by_text("Frame Extractor & Analyzer")).to_be_visible(timeout=30000)

        # 3. Status area attached.
        expect(self.page.locator(Selectors.UNIFIED_STATUS)).to_be_attached(timeout=5000)

        # 4. Reset to clean state (mock-only control surface).
        reset_btn = self.page.locator(Selectors.RESET_STATE_BUTTON)
        if not reset_btn.is_visible():
            try:
                self.open_accordion("Tests (Experimental)")
            except Exception:
                pass

        if reset_btn.is_visible():
            reset_btn.click()
            expect(self.page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(Selectors.STATUS_READY, timeout=15000)
            expect(self.page.get_by_text("System Reset Ready.")).to_be_visible(timeout=5000)

        # Final settle for event-loop binding stability.
        self.page.wait_for_timeout(500)
        return self

    def navigate(self, tab_name: str) -> "AppDriver":
        """Switch to a main tab by label, idempotently.

        Lifted from ``switch_to_tab``. Uses ``aria-selected`` to avoid a
        redundant click when already on the target tab, then waits for the
        post-switch loading indicator to clear.
        """
        tab_btn = self.page.get_by_role("tab", name=tab_name, exact=False)
        expect(tab_btn).to_be_visible(timeout=10000)
        if tab_btn.get_attribute("aria-selected") != "true":
            tab_btn.click(force=True)
            expect(tab_btn).to_have_attribute("aria-selected", "true", timeout=5000)
            self.page.wait_for_selector(".generating, [data-testid='loading']", state="hidden", timeout=2000)
        return self

    def open_accordion(self, label: str) -> "AppDriver":
        """Open a Gradio accordion if it isn't already, by label or elem_id.

        Lifted from ``open_accordion``. Resolves a known elem_id where possible
        (more resilient than text), then checks ``<details open>`` and
        ``aria-expanded`` before toggling so a re-click never collapses it.
        """
        elem_id = _ACCORDION_ELEM_IDS.get(label)
        if elem_id:
            accordion = self.page.locator(elem_id)
        else:
            accordion = self.page.get_by_text(label, exact=False).first

        expect(accordion).to_be_visible(timeout=5000)

        is_open = False
        try:
            if accordion.evaluate("el => el.tagName === 'DETAILS' && el.open"):
                is_open = True
            elif accordion.get_attribute("aria-expanded") == "true":
                is_open = True
            elif accordion.locator("button[aria-expanded='true']").count() > 0:
                is_open = True
        except Exception:
            pass

        if not is_open:
            accordion.click()
            self.page.wait_for_timeout(500)
        return self

    # -- pipeline verbs ------------------------------------------------------
    # Each verb captures one repeated E2E idiom. They all assume the driver is
    # already on the relevant tab and (except for extraction) that the prior
    # pipeline stage completed; the caller asserts completion between stages.

    def extract(self, source: str | None = None, *, method: str | None = None, force: bool = False) -> "AppDriver":
        """Optionally pick an extraction method, fill the source, start extraction.

        ``method`` selects from the ``#method_input`` dropdown (Gradio 5's
        custom list component — requires ``force=True`` on the option click).
        When omitted, the app default is used.
        """
        if method is not None:
            self.select_method(method)
        if source is not None:
            self.page.locator(Selectors.SOURCE_INPUT).fill(source)
        self.page.locator(Selectors.START_EXTRACTION).click(force=force)
        return self

    def select_method(self, method: str) -> "AppDriver":
        """Choose an option from the extraction-method dropdown.

        Encapsulates the Gradio 6 dropdown quirk documented in tests/README.md:
        the list renders as ``role=option`` list items, not a native ``<select>``,
        and the transparent overlay forces a ``force=True`` click.
        """
        self.page.locator(f"{Selectors.EXTRACTION_METHOD} input").click()
        self.page.get_by_role("option", name=method, exact=False).click(force=True)
        return self

    def pre_analyze(self, *, force: bool = True) -> "AppDriver":
        """Click 'Confirm Subject' on the Subject tab."""
        btn = self.page.locator(Selectors.START_PRE_ANALYSIS)
        expect(btn).to_be_visible(timeout=10000)
        btn.click(force=force)
        return self

    def propagate(self, *, force: bool = True) -> "AppDriver":
        """Click 'Propagate Masks' on the Scenes tab."""
        btn = self.page.locator(Selectors.PROPAGATE_MASKS)
        expect(btn).to_be_visible(timeout=10000)
        btn.click(force=force)
        return self

    def analyze(self, *, force: bool = True) -> "AppDriver":
        """Click 'Start Analysis' on the Metrics tab."""
        btn = self.page.locator(Selectors.START_ANALYSIS)
        expect(btn).to_be_visible(timeout=10000)
        btn.click(force=force)
        return self

    def export(self, *, force: bool = True) -> "AppDriver":
        """Click the Export button on the Export tab."""
        btn = self.page.locator(Selectors.EXPORT_BUTTON)
        expect(btn).to_be_visible(timeout=10000)
        btn.click(force=force)
        return self

    def dry_run(self) -> "AppDriver":
        """Click the Dry Run button on the Export tab."""
        btn = self.page.locator(Selectors.DRY_RUN_BUTTON)
        expect(btn).to_be_visible(timeout=10000)
        btn.click()
        return self

    def click_cancel(self) -> "AppDriver":
        """Click the global Cancel button (only enabled while a pipeline runs)."""
        self.page.locator(Selectors.CANCEL_BUTTON).click()
        return self

    def select_strategy(self, strategy_label: str) -> "AppDriver":
        """Select a subject-detection strategy radio (Auto/Face/Text)."""
        self.page.get_by_label(strategy_label, exact=False).click()
        return self

    def load_session(self, path: str) -> "AppDriver":
        """Open the session accordion, fill the path, and click Load Session.

        ``force=True`` bypasses the Gradio 5 transparent overlay that
        intermittently intercepts the button click.
        """
        self.open_accordion(Labels.SESSION_ACCORDION)
        self.page.locator(Selectors.SESSION_INPUT).fill(path)
        self.page.get_by_role("button", name="Load Session").click(force=True)
        return self

    # -- assertions ----------------------------------------------------------

    def expect_status(self, match: TextMatch, *, timeout: int = 30000) -> "AppDriver":
        """Assert the unified status area contains ``match`` (text or regex)."""
        expect(self.page.locator(Selectors.UNIFIED_STATUS)).to_contain_text(match, timeout=timeout)
        return self

    def expect_log(self, match: TextMatch, *, timeout: int = 10000) -> "AppDriver":
        """Assert the system log textarea contains ``match``.

        Canonical log assertion. Auto-opens the System Logs accordion (it is
        collapsed by default and its contents are lazy-rendered), then uses
        ``to_contain_text``, a *polling* assertion that tolerates the 0.5s
        ``gr.Timer`` refresh in LogViewer.

        If the textarea appears empty after opening the accordion (the timer
        hasn't fired yet), clicks the Refresh button to force an immediate
        queue drain before polling.
        """
        self.open_accordion(Labels.SYSTEM_LOGS)
        textarea = self.page.locator(Selectors.LOG_TEXTAREA)
        # If the textarea is empty, the 0.5s gr.Timer may not have fired yet.
        # Click Refresh to force an immediate queue drain. Skip if the textarea
        # already has content (e.g. from a direct Gradio yield) — clicking
        # Refresh would replace it with a queue snapshot that may be stale.
        try:
            if textarea.input_value() == "":
                refresh_btn = self.page.locator(Selectors.REFRESH_LOGS)
                if refresh_btn.is_visible():
                    refresh_btn.click()
                    self.page.wait_for_timeout(500)
        except Exception:
            pass
        import re

        if isinstance(match, str):
            pattern = re.compile(re.escape(match))
        else:
            pattern = match
        expect(textarea).to_have_value(pattern, timeout=timeout)
        return self

    def expect_log_equals(self, text: str, *, timeout: int = 5000) -> "AppDriver":
        """Assert the log textarea's value is exactly ``text``.

        Sole exact-value log assertion, for the clear-logs case that asserts an
        empty string. Opens the accordion first for the same reason as
        ``expect_log``. Uses ``to_have_value`` (a snapshot read) intentionally:
        here we want an exact value, not a substring poll.
        """
        self.open_accordion(Labels.SYSTEM_LOGS)
        expect(self.page.locator(Selectors.LOG_TEXTAREA)).to_have_value(text, timeout=timeout)
        return self

    def expect_disabled(self, selector: str, *, timeout: int = 5000) -> "AppDriver":
        """Assert a control is present and disabled."""
        expect(self.page.locator(selector)).to_be_disabled(timeout=timeout)
        return self

    def expect_enabled(self, selector: str, *, timeout: int = 5000) -> "AppDriver":
        """Assert a control is present and enabled."""
        expect(self.page.locator(selector)).to_be_enabled(timeout=timeout)
        return self

    def expect_visible(self, selector: str, *, timeout: int = 5000) -> "AppDriver":
        """Assert a control is visible."""
        expect(self.page.locator(selector)).to_be_visible(timeout=timeout)
        return self

    def expect_no_error_toast(self) -> "AppDriver":
        """Assert no Gradio error toast is visible."""
        expect(self.page.locator(".toast-wrap")).not_to_be_visible()
        return self

    def select_preset(self, preset_name: str) -> "AppDriver":
        """Choose an option from the filter preset dropdown."""
        self.page.get_by_label("Use a Preset", exact=False).click()
        self.page.get_by_role("option", name=preset_name, exact=True).click(force=True)
        return self


# ---------------------------------------------------------------------------
# Known accordion label -> elem_id mapping.
# Mirrors the old ``open_accordion`` lookup; kept here so the driver is the
# single owner of "how to find an accordion by human label".
# ---------------------------------------------------------------------------
_ACCORDION_ELEM_IDS: dict[str, str] = {
    "Log": "#system_logs_accordion",
    "Help": "#help_accordion",
    "Deduplication": "#dedup_accordion",
    "Advanced Model Configuration": "#subject_advanced_config_accordion",
    "Quality Score": "#accordion_quality_score",
    "Sharpness": "#accordion_sharpness",
}
