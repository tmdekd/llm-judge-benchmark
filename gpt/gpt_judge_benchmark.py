# -------------------------
# Python Standard Libraries
# -------------------------
import os
from pathlib import Path
import traceback

# import sys
# import json

# -------------------------
# Third-Party Libraries
# -------------------------
import dotenv
from openai import OpenAI


# project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(BASE_DIR / "settings.env")
api_key = os.environ.get("OPENAI_KEY")
client = OpenAI(api_key=api_key)


def main():
    system_prompt = """\
    당신은 Rock 전문가이다.
    
    목표:
    - 입력으로 주어지는 rocker 이름으로 그 사람의 대표적인 음악에 대해서 서술한다.
    
    입력 형식:
    - Rocker 이름
    
    출력 형식 (중요):
    - 오래된 것부터 최근 것까지 시대별로 주요 음악 차트 순위를 포함하여 서술하라.
    - 작품의 제목, 고유명사같은 경우는 가급적 원래의 언어로 표현하라.
    """

    user_prompt = """\
    다음 rocker 이름을 입력으로 system 메시지의 규칙에 따라 500자 이내로 서술하라.
    
    {{이름}}
    """

    completion = client.chat.completions.create(
        model="gpt-5",
        # temperature = ai_setting['temperature'],
        # max_completion_tokens = ai_setting['max_tokens'],
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt.replace("{{이름}}", "Jimi Hendrix"),
            },
        ],
        # response_format=response_format,
        # request_timeout=request_timeout
    )

    output = completion.choices[0].message.content
    print(output)


if __name__ == "__main__":
    try:
        print(f"BASE_DIR : {BASE_DIR}")
        main()
    except:
        traceback.print_exc()
