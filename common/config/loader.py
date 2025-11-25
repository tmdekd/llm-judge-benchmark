from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_config():
    with (BASE_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
