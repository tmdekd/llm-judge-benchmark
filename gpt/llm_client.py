import os  # OS 환경 변수 접근을 위한 모듈
from pathlib import Path  # 경로 처리를 편하게 해주는 Path 객체
import dotenv  # .env 파일 로딩용
from openai import AsyncOpenAI  # 비동기 OpenAI 클라이언트

# 프로젝트 루트 디렉터리 (현재 파일 기준으로 상위 폴더)
BASE_DIR = Path(__file__).resolve().parent.parent

# settings.env 로드
dotenv.load_dotenv(BASE_DIR / "settings.env")

# 환경 변수에서 API 키 읽기
api_key = os.environ.get("OPENAI_KEY")

# 전역 AsyncOpenAI 클라이언트 생성
client = AsyncOpenAI(api_key=api_key)


def get_client() -> AsyncOpenAI:
    """
    전역 AsyncOpenAI 클라이언트를 반환.
    여러 모듈에서 공통으로 사용하기 위해 분리.
    """
    return client


def load_text(path: Path) -> str:
    """
    주어진 경로에서 텍스트 파일을 읽어 문자열로 반환.
    Path 객체를 인자로 받도록 통일.
    """
    with path.open("r", encoding="utf-8") as f:
        return f.read()
