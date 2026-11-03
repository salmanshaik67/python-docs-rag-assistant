"""Download a handful of Python documentation pages and save them as plain text."""

from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://docs.python.org/3/"

PAGES = [
    "tutorial/introduction.html",
    "tutorial/controlflow.html",
    "tutorial/datastructures.html",
    "tutorial/modules.html",
    "tutorial/inputoutput.html",
    "tutorial/errors.html",
    "tutorial/classes.html",
    "tutorial/stdlib.html",
    "tutorial/venv.html",
    "library/functions.html",
    "library/stdtypes.html",
    "library/os.path.html",
    "library/json.html",
    "library/datetime.html",
    "library/collections.html",
    "library/itertools.html",
    "library/pathlib.html",
    "library/re.html",
]

DOCS_DIR = Path(__file__).parent / "data" / "docs"


BLOCK_TAGS = ["p", "pre", "li", "dt", "dd", "h1", "h2", "h3", "h4", "tr", "div"]


def page_to_text(html: bytes) -> tuple[str, str]:
    """Return (title, readable body text) for a docs page."""
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    body = soup.select_one("div.body") or soup.body
    for tag in body.select("a.headerlink, script, style"):
        tag.decompose()

    h1 = body.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    # Break lines only at block elements so inline code stays inside its sentence.
    for tag in body.find_all(BLOCK_TAGS):
        tag.append("\n")
    lines = (" ".join(line.split()) for line in body.get_text().splitlines())
    return title, "\n".join(line for line in lines if line)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "python-docs-rag-assistant/1.0"

    for page in PAGES:
        url = BASE_URL + page
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"skip  {url}: {exc}")
            continue

        name = page.replace("/", "_").removesuffix(".html") + ".txt"
        title, text = page_to_text(resp.content)
        # First two lines hold the source URL and page title so the app can cite them.
        (DOCS_DIR / name).write_text(f"{url}\n{title}\n{text}", encoding="utf-8")
        print(f"saved {name}")


if __name__ == "__main__":
    main()
