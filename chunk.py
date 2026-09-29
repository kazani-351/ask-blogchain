"""Step 2: CHUNK. Split each post into overlapping pieces small enough to search.

Why chunk at all: a whole post mixes many topics, so it matches everything a little.
A ~300-word piece is usually about one idea, so a match means something.
Why overlap: an answer that straddles a boundary still lands whole in one chunk.

Run: python3 chunk.py [chunk_words] [overlap_words]   -> writes data/chunks.json
"""
import json
import sys
from pathlib import Path

POSTS_FILE = Path("data/posts.json")
OUT_FILE = Path("data/chunks.json")


def chunk_post(post, size, overlap):
    """Pack whole paragraphs into chunks of about `size` words.

    Splitting on paragraphs (not every N words) keeps sentences intact.
    Each new chunk starts with the last `overlap` words of the previous one.
    """
    paras = [p.split() for p in post["text"].split("\n\n") if p.strip()]
    chunks, current = [], []
    for words in paras:
        if current and len(current) + len(words) > size:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
        current = current + words
    if current:
        chunks.append(current)
    return [
        {
            "id": f"{post['id']}#{i}",
            "post_id": post["id"],
            "title": post["title"],
            "url": post["url"],
            "text": " ".join(words),
        }
        for i, words in enumerate(chunks)
    ]


def build_chunks(size=300, overlap=50):
    posts = json.loads(POSTS_FILE.read_text())
    return [c for post in posts for c in chunk_post(post, size, overlap)]


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    chunks = build_chunks(size, overlap)
    OUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"{len(chunks)} chunks (size={size}, overlap={overlap}) -> {OUT_FILE}")


if __name__ == "__main__":
    main()
