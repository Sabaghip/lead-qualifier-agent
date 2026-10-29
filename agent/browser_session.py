from __future__ import annotations

from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

MAX_CHARS_PER_PAGE = 6000
NAV_TIMEOUT_MS = 15_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def _same_domain(base: str, candidate: str) -> bool:
    return urlparse(base).netloc == urlparse(candidate).netloc


def _normalize(url: str) -> str:
    """Strip the fragment and trailing slash so #anchor links and the
    trailing-slash variant of a URL aren't treated as distinct pages."""
    parsed = urlparse(url)._replace(fragment="")
    normalized = parsed.geturl()
    return normalized[:-1] if normalized.endswith("/") and len(normalized) > 1 else normalized


class BrowserSession:
    """A single reusable Playwright page, kept open across multiple fetch() calls.

    Unlike scraper.scrape_site (which launches a fresh browser per run and
    follows a fixed keyword-scored list of links), this is meant to be driven
    one fetch at a time by an agentic loop that decides the next URL itself —
    so the browser stays open between decisions instead of relaunching.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(user_agent=USER_AGENT)
        self._page = self._context.new_page()
        self._page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

    def fetch(self, url: str) -> tuple[str, list[str], str | None]:
        """Returns (text, same-domain links found on the page, error-or-None).

        Links are normalized (fragment and trailing slash stripped) and
        deduped, so e.g. a "#pricing" anchor on the homepage isn't treated
        as a separate page from the homepage itself.
        """
        try:
            self._page.goto(url, wait_until="domcontentloaded")
            self._page.wait_for_timeout(500)
            text = self._page.inner_text("body")[:MAX_CHARS_PER_PAGE]
            hrefs = self._page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
            own_url = _normalize(url)
            links = sorted(
                {
                    _normalize(h)
                    for h in hrefs
                    if h and _same_domain(self.base_url, h) and _normalize(h) != own_url
                }
            )
            return text, links, None
        except PlaywrightTimeoutError:
            return "", [], f"timeout loading {url}"
        except PlaywrightError as exc:
            return "", [], f"error loading {url}: {exc}"

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()

    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
