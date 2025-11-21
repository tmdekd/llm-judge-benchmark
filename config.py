from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROMPT_DIR = BASE_DIR / "prompt"
DATASET_DIR = BASE_DIR / "data" / "test_dataset"
OUTPUT_DIR = DATASET_DIR / "output"
SCORE_DIR = OUTPUT_DIR / "scores"

ENV_CSV_PATH = DATASET_DIR / "environment_50.csv"
HEALTH_CSV_PATH = DATASET_DIR / "health_50.csv"

LOG_DIR = BASE_DIR / "log"

MAX_LIMIT = 25
REPETITION_NUM = 10
