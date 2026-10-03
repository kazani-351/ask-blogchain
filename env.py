"""Load .env into the process once. Values are never printed."""
import os
from pathlib import Path


def load():
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() and value.strip() and not name.strip().startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


load()
