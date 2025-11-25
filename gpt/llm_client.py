import os
from pathlib import Path
import dotenv
from openai import AsyncOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(BASE_DIR / "settings.env")

api_key = os.environ.get("OPENAI_KEY")
client = AsyncOpenAI(api_key=api_key)


def get_client() -> AsyncOpenAI:
    return client
