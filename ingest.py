"""Step 1: LOAD. Turn the BlogChain RSS feed into clean plain-text posts.

Run: python3 ingest.py   -> writes data/posts.json
"""
import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

FEED_URL = "https://paragraph.com/@kazani/rss"
FEED_FILE = Path("data/feed.xml")
OUT_FILE = Path("data/posts.json")
CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"

# Block-level tags become paragraph breaks, so chunking can split on them later.
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "br", "div"}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0  # inside <script>/<style>

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(raw):
    p = TextExtractor()
    p.feed(raw)
    text = html.unescape("".join(p.parts))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*\n\s*", "\n\n", text).strip()


def main():
    if not FEED_FILE.exists():
        FEED_FILE.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(FEED_URL, FEED_FILE)
    root = ET.parse(FEED_FILE).getroot()
    posts = []
    for item in root.findall("./channel/item"):
        link = item.findtext("link")
        posts.append({
            "id": link.rstrip("/").rsplit("/", 1)[-1],  # URL slug, stable across runs
            "title": item.findtext("title"),
            "url": link,
            "date": item.findtext("pubDate"),
            "text": html_to_text(item.findtext(CONTENT) or ""),
        })
    OUT_FILE.write_text(json.dumps(posts, indent=2, ensure_ascii=False))
    words = sum(len(p["text"].split()) for p in posts)
    print(f"{len(posts)} posts, {words:,} words -> {OUT_FILE}")


if __name__ == "__main__":
    main()
