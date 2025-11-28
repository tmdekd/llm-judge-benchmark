import requests
import argparse
import asyncio
import re
from typing import List, Dict
import os
import json
import dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(BASE_DIR / "settings.env")

QWEN_API = os.getenv("QWEN_API")


class ModelManager:

    def __init__(
        self,
        cpl_model_name="Qwen/Qwen2.5-72B-Instruct",
        base_url=QWEN_API,
    ):

        self.cpl_model_name = cpl_model_name
        self.cpl_model = self.cpl_model_name
        self.base_url = base_url
        print(f"---base url: {self.base_url}")

        self.headers = {"Content-Type": "application/json"}

    async def __completion__(self, system_prompt, user_prompt, is_json=False):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        data = {
            "model": self.cpl_model,
            "messages": messages,
        }

        if is_json:
            data["response_format"] = {"type": "json_object"}

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._make_request, data)

            response.raise_for_status()
            response_json = response.json()

            return response_json["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing error: {e}")
        except KeyError as e:
            raise Exception(f"Response format error: {e}")

    def _make_request(self, data):
        return requests.post(self.base_url, headers=self.headers, json=data)


def get_embedding(text):
    url = os.environ["EMB_API"]

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    data = {"text": text}

    try:
        response = requests.post(url, headers=headers, json=data)

        response.raise_for_status()

        print("Embedding Status Code:", response.status_code)
        return response.json()["embeddings"]

    except requests.exceptions.RequestException as e:
        print(f"RequestException: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JJSONDecodeError: {e}")
        return []


def get_reranker(query, documents):
    url = os.environ["RERANK_URL"]

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    data = {"query": query, "documents": documents}

    try:
        response = requests.post(url, headers=headers, json=data)

        response.raise_for_status()

        print("Reranker Status Code:", response.status_code)
        return response.json()["ranked_documents"]

    except requests.exceptions.RequestException as e:
        print(f"RequestException: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return []


if __name__ == "__main__":
    manager = ModelManager()

    user_query = "시중은행, 지방은행, 인터넷은행의 인가 요건 및 절차에 차이가 있는데 그 차이점은 무엇인가요?"
    system_prompt = "친절하게 답변해줘"

    result = asyncio.run(manager.__completion__(system_prompt, user_query))
    print(result)

    """
    {
    "id": "chatcmpl-8cd6f5d3e0634f59bdfef81b26535070",
    "object": "chat.completion",
    "created": 1750202414,
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "choices": [
        {
        "index": 0,
        "message": {
            "role": "assistant",
            "reasoning_content": null,
            "content": "산소포화도가 낮게 측정된 경우, 다음달에는 몇 가지 조치를 취할 수 있습니다:\n\n1. **의사와 상담**: 산소포화도가 낮게 측정된 이유를 확인하고, 의사와 상담하여 적절한 치료 계획을 수립하세요.\n2. **산소치료**: 의사의 지시에 따라 산소치료를 받으세요. 이는 산소포화도를 높이고, 호흡을 개선하는 데 도움이 될 수 있습니다.\n3. **생활습관 개선**: 건강한 생활습관을 유지하세요. 이는 다음과 같은 것들을 포함합니다:\n\t* 규칙적인 운동\n\t* 균형 잡힌 식사\n\t* 충분한 수면\n\t* 스트레스 관리\n4. **환경 개선**: 집안의 환경을 개선하세요. 이는 다음과 같은 것들을 포함합니다:\n\t* 공기청정기 사용\n\t* 담배연기 및 기타 유해 물질로부터 피하기\n\t* 적절한 환기\n5. **산소포화도 모니터링**: 산소포화도 모니터링을 계속 하세요. 이는 산소포화도가 낮게 측정된 경우, 빠르게 대응할 수 있도록 도와줍니다.\n6. **호흡 운동**: 호흡 운동을 해보세요. 이는 폐활량을 증가시키고, 호흡을 개선하는 데 도움이 될 수 있습니다.\n\n이러한 조치를 취하면 다음달에 산소포화도가 낮게 측정되는 것을 예방할 수 있을 것입니다. 그러나, 산소포화도가 낮게 측정된 경우, 의사와 상담하여 적절한 치료 계획을 수립하는 것이 중요합니다.",
            "tool_calls": []
        },
        "logprobs": null,
        "finish_reason": "stop",
        "stop_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 93,
        "total_tokens": 473,
        "completion_tokens": 380,
        "prompt_tokens_details": null
    },
    "prompt_logprobs": null,
    "kv_transfer_params": null
    }
    """
