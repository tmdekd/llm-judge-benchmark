import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import dotenv
from openai import AsyncOpenAI
from traceback import print_exc
from calculate_function.calculate_functions import calculate_faithfulness_score, calculate_relevance_score
import asyncio

# OpenAI API 동시 호출 최대 개수
MAX_LIMIT = 10

# LLM 평가 반복 횟수
REPETITION_NUM = 3

# =========================
# Logger 설정: log 폴더 생성
# =========================
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, "llm_evaluation_async.log"),
            encoding="utf-8-sig",
        ),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =========================
# OpenAI / 환경 설정
# =========================
dotenv.load_dotenv("./settings.env")
api_key = os.environ.get("OPENAI_KEY")

client = AsyncOpenAI(api_key=api_key)


def load_text(path: str) -> str:
    """주어진 경로에서 텍스트 파일을 읽어 문자열로 반환"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


async def score_row(
    llm_answer: str,
    retrieved_result: str,
    system_prompt: str,
    user_template: str,
) -> tuple[int, str]:
    """
    하나의 LLM 응답(llm_answer)과 그에 해당하는 retrieved_result(평가 근거)를 받아서
    GPT-5.1 모델을 비동기로 호출해 1~5점 사이의 score와 scoring_reason을 반환한다.
    반환값: (score: int, scoring_reason: str)
    """
    user_prompt = user_template.format(
        retrieved_result=str(retrieved_result),
        llm_answer=str(llm_answer),
    )

    # OpenAI Chat Completions API 비동기 호출
    response = await client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        # JSON object만 반환하도록 강제
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    score = int(data.get("score", 0))
    scoring_reason = data.get("scoring_reason", "")

    return score, scoring_reason


async def async_add_gpt_score_columns(
    df: pd.DataFrame,
    system_prompt: str,
    user_template: str,
    metric_prefix: str,
    MAX_LIMIT: int = 5,
) -> pd.DataFrame:
    """
    비동기 방식으로 주어진 DataFrame(df)의 각 행에 대해
    - llm_answer
    - retrieved_result
    값을 GPT-5.1로 평가하여
    {metric_prefix}_score, {metric_prefix}_scoring_reason 컬럼을 추가한 뒤,
    수정된 DataFrame을 반환한다.
    """

    scores: list[int] = [0] * len(df)
    reasons: list[str] = [""] * len(df)

    semaphore = asyncio.Semaphore(MAX_LIMIT)

    async def worker(pos: int, row: pd.Series):
        """
        pos: df 내에서의 위치(0, 1, 2, ...)
        row: 해당 위치의 행 데이터
        """
        llm_answer = row.get("llm_answer", "")
        retrieved_result = row.get("retrieved_result", "")

        async with semaphore:
            logger.info(f"Row {pos} 평가 중... ({metric_prefix})")
            try:
                score, reason = await score_row(
                    llm_answer=llm_answer,
                    retrieved_result=retrieved_result,
                    system_prompt=system_prompt,
                    user_template=user_template,
                )
            except Exception as e:
                logger.error(f"Row {pos} scoring failed ({metric_prefix}): {e}")
                score = 0
                reason = f"error: {e}"

        return pos, score, reason

    tasks = [worker(pos, row) for pos, (_, row) in enumerate(df.iterrows())]

    for coro in asyncio.as_completed(tasks):
        pos, score, reason = await coro
        scores[pos] = score
        reasons[pos] = reason

    df[f"{metric_prefix}_score"] = scores
    df[f"{metric_prefix}_scoring_reason"] = reasons

    return df


def build_results_from_df(df: pd.DataFrame, score_column: str):
    """
    calculate_*_score 함수에 넣기 위한 results 리스트 생성 유틸.
    df[score_column]에서 점수를 읽어서 [{"id": row_index, "score": score}, ...] 형태로 변환.
    """
    results = []
    for idx, score in df[score_column].items():
        if score is None:
            s = 0
        else:
            s = int(score)
        results.append({"id": idx, "score": s})

    return results


async def main_async(num_runs: int = 3):
    # ===== 경로 설정 =====
    PROMPT_FOLDER_PATH = "prompt/"
    SYSTEM_PROMPT_FAITH_PATH = PROMPT_FOLDER_PATH + "system_prompt_faithfulness.txt"
    SYSTEM_PROMPT_REL_PATH = PROMPT_FOLDER_PATH + "system_prompt_relevance.txt"
    USER_TEMPLATE_PATH = PROMPT_FOLDER_PATH + "user_template.txt"

    FOLDER_PATH = "data/test_dataset/"
    OUTPUT_PATH = FOLDER_PATH + "output/"
    SCORE_PATH = OUTPUT_PATH + "scores/"
    ENV_CSV_PATH = FOLDER_PATH + "environment_50.csv"
    HEALTH_CSV_PATH = FOLDER_PATH + "health_50.csv"

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(SCORE_PATH, exist_ok=True)

    # ===== 데이터 로드 =====
    df_env_base = pd.read_csv(ENV_CSV_PATH)
    df_health_base = pd.read_csv(HEALTH_CSV_PATH)

    logger.info(f"주거 환경 CSV rows: {len(df_env_base)}")
    logger.info(f"건강 CSV rows: {len(df_health_base)}\n")

    # ===== 프롬프트 로드 =====
    system_prompt_faith = load_text(SYSTEM_PROMPT_FAITH_PATH)
    system_prompt_rel = load_text(SYSTEM_PROMPT_REL_PATH)
    user_template = load_text(USER_TEMPLATE_PATH)

    logger.info("System Prompt (Faithfulness) loaded.")
    logger.info("System Prompt (Relevance) loaded.")
    logger.info("User Template loaded.\n")

    for i in range(1, num_runs + 1):
        logger.info(f"[RUN {i}] ===== LLM 평가 파이프라인 시작 =====\n")

        df_env = df_env_base.copy()
        df_health = df_health_base.copy()

        # ========= 주거 환경 데이터 + 건강 데이터 Faithfulness 평가 (두 CSV 동시 실행) =========
        logger.info("주거 환경/건강 데이터 기반 LLM 응답 평가 시작 - Faithfulness (두 CSV 동시 실행)")

        faith_start = time.time()

        env_faith_task = asyncio.create_task(
            async_add_gpt_score_columns(
                df=df_env,
                system_prompt=system_prompt_faith,
                user_template=user_template,
                metric_prefix="gpt_faithfulness",
                MAX_LIMIT=MAX_LIMIT,
            )
        )

        health_faith_task = asyncio.create_task(
            async_add_gpt_score_columns(
                df=df_health,
                system_prompt=system_prompt_faith,
                user_template=user_template,
                metric_prefix="gpt_faithfulness",
                MAX_LIMIT=MAX_LIMIT,
            )
        )

        df_env_scored, df_health_scored = await asyncio.gather(env_faith_task, health_faith_task)

        faith_end = time.time()
        logger.info(f"<TIME> Faithfulness (환경+건강 두 CSV 동시 평가) 소요 시간: {faith_end - faith_start:.2f}초\n")

        # ========= 주거 환경 데이터 + 건강 데이터 Relevance 평가 (두 CSV 동시 실행) =========
        logger.info("주거 환경/건강 데이터 기반 LLM 응답 평가 시작 - Relevance (두 CSV 동시 실행)")

        rel_start = time.time()

        env_rel_task = asyncio.create_task(
            async_add_gpt_score_columns(
                df=df_env_scored,
                system_prompt=system_prompt_rel,
                user_template=user_template,
                metric_prefix="gpt_relevance",
                MAX_LIMIT=MAX_LIMIT,
            )
        )

        health_rel_task = asyncio.create_task(
            async_add_gpt_score_columns(
                df=df_health_scored,
                system_prompt=system_prompt_rel,
                user_template=user_template,
                metric_prefix="gpt_relevance",
                MAX_LIMIT=MAX_LIMIT,
            )
        )

        df_env_scored, df_health_scored = await asyncio.gather(env_rel_task, health_rel_task)

        rel_end = time.time()
        logger.info(f"<TIME> Relevance (환경+건강 두 CSV 동시 평가) 소요 시간: {rel_end - rel_start:.2f}초\n")

        # ========= CSV 저장 =========
        env_output_path = OUTPUT_PATH + "environment_50_scored.csv"
        df_env_scored.to_csv(env_output_path, encoding="utf-8-sig", index=False)
        logger.info(f"[INFO] 주거 환경 데이터 기반 LLM 응답 스코어 결과 저장 완료: {env_output_path}")

        health_output_path = OUTPUT_PATH + "health_50_scored.csv"
        df_health_scored.to_csv(health_output_path, encoding="utf-8-sig", index=False)
        logger.info(f"[INFO] 건강 데이터 기반 LLM 응답 스코어 결과 저장 완료: {health_output_path}\n")

        # ========= 최종 점수 집계 (calculate_functions 적용) =========
        # Faithfulness
        env_faith_results = build_results_from_df(df_env_scored, "gpt_faithfulness_score")
        health_faith_results = build_results_from_df(df_health_scored, "gpt_faithfulness_score")

        env_faith_norm = calculate_faithfulness_score(env_faith_results, original=False)
        health_faith_norm = calculate_faithfulness_score(health_faith_results, original=False)

        logger.info(f"[ENV] Faithfulness : {env_faith_norm}")
        logger.info(f"[HEALTH] Faithfulness : {health_faith_norm}")

        logger.info(f"[ENV] Faithfulness (소수점 4자리) : {env_faith_norm:.4f}")
        logger.info(f"[HEALTH] Faithfulness (소수점 4자리) : {health_faith_norm:.4f}\n\n")

        # Relevance
        env_rel_results = build_results_from_df(df_env_scored, "gpt_relevance_score")
        health_rel_results = build_results_from_df(df_health_scored, "gpt_relevance_score")

        health_rel_norm = calculate_relevance_score(health_rel_results, original=False)
        env_rel_norm = calculate_relevance_score(env_rel_results, original=False)

        logger.info(f"[ENV] Relevance : {env_rel_norm}")
        logger.info(f"[HEALTH] Relevance : {health_rel_norm}")

        logger.info(f"[ENV] Relevance (소수점 4자리) : {env_rel_norm:.4f}")
        logger.info(f"[HEALTH] Relevance (소수점 4자리) : {health_rel_norm:.4f}\n")

        # ========= Faithfulness, Relevance 점수 기록 =========
        score_file_path = os.path.join(
            SCORE_PATH,
            f"faithfulness_relevance_scores{i}.txt",
        )

        kst = ZoneInfo("Asia/Seoul")
        now_kst = datetime.now(tz=kst)
        with open(score_file_path, "w", encoding="utf-8-sig") as f:
            f.write(f"Run: {i}\n")
            f.write("현재 시간 : " + now_kst.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            f.write("=== Faithfulness Scores ===\n")
            f.write(f"ENV_Faithfulness : {env_faith_norm}\n")
            f.write(f"HEALTH_Faithfulness : {health_faith_norm}\n\n")
            f.write(f"ENV_Faithfulness (소수점 4자리) : {env_faith_norm:.4f}\n")
            f.write(f"HEALTH_Faithfulness (소수점 4자리) : {health_faith_norm:.4f}\n\n")
            f.write(f"Faithfulness 평가 평균 : {(env_faith_norm + health_faith_norm) / 2:.4f}\n\n")
            f.write("=== Relevance Scores ===\n")
            f.write(f"ENV_Relevance : {env_rel_norm}\n")
            f.write(f"HEALTH_Relevance : {health_rel_norm}\n")
            f.write(f"ENV_Relevance (소수점 4자리) : {env_rel_norm:.4f}\n")
            f.write(f"HEALTH_Relevance (소수점 4자리) : {health_rel_norm:.4f}\n\n")
            f.write(f"Relevance 평가 평균 : {(env_rel_norm + health_rel_norm) / 2:.4f}\n")

        logger.info(f"[RUN {i}] Score file saved: {score_file_path}")
        logger.info(f"[RUN {i}] ===== LLM 평가 파이프라인 종료 =====")
        logger.info(f"[RUN {i}] ====================================\n\n")


if __name__ == "__main__":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main_async(num_runs=REPETITION_NUM))
    except Exception:
        logger.exception("Unhandled exception occurred:")
        print_exc()
