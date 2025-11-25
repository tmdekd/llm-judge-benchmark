from pathlib import Path


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()
