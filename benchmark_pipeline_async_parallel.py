import os
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import asyncio

from calculate_function.calculate_functions import (
    calculate_faithfulness_score,
    calculate_relevance_score,
)

from gpt.llm_client import load_text
from gpt.async_scorer import async_add_gpt_score_columns, build_results_from_df
from config import (
    PROMPT_DIR,
    OUTPUT_DIR,
    SCORE_DIR,
    ENV_CSV_PATH,
    HEALTH_CSV_PATH,
    LOG_DIR,
    MAX_LIMIT,
    REPETITION_NUM,
)
from tqdm import tqdm
import logging


# tqdm과 logger 충돌 방지용 핸들러
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # <-- tqdm-friendly 출력
            self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger("llm_judge")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# 파일 로그
file_handler = logging.FileHandler(LOG_DIR / "llm_evaluation_async.log", encoding="utf-8-sig")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# tqdm-safe console 로그
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(formatter)
logger.addHandler(tqdm_handler)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main_async(num_runs: int):

    logger.info("===== LLM 평가 파이프라인 시작 =====")

    df_env_base = pd.read_csv(ENV_CSV_PATH)
    df_health_base = pd.read_csv(HEALTH_CSV_PATH)
    logger.info(f"주거 환경 CSV rows: {len(df_env_base)}")
    logger.info(f"건강 CSV rows: {len(df_health_base)}")

    system_prompt_faith = load_text(PROMPT_DIR / "system_prompt_faithfulness.txt")
    system_prompt_rel = load_text(PROMPT_DIR / "system_prompt_relevance.txt")
    user_template = load_text(PROMPT_DIR / "user_template.txt")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_runs + 1):
        total_start = time.time()
        logger.info(f"[RUN {i}] =============== START ====================")

        df_env = df_env_base.copy()
        df_health = df_health_base.copy()

        # -----------------------------
        # Faithfulness
        # -----------------------------
        env_faith_task = async_add_gpt_score_columns(
            df_env,
            system_prompt_faith,
            user_template,
            "gpt_faithfulness",
            MAX_LIMIT,
            progress_desc=f"[RUN {i}] ENV Faithfulness",
            progress_position=0,
        )
        health_faith_task = async_add_gpt_score_columns(
            df_health,
            system_prompt_faith,
            user_template,
            "gpt_faithfulness",
            MAX_LIMIT,
            progress_desc=f"[RUN {i}] HEALTH Faithfulness",
            progress_position=1,
        )

        df_env_scored, df_health_scored = await asyncio.gather(env_faith_task, health_faith_task)

        # -----------------------------
        # Relevance
        # -----------------------------
        env_rel_task = async_add_gpt_score_columns(
            df_env_scored,
            system_prompt_rel,
            user_template,
            "gpt_relevance",
            MAX_LIMIT,
            progress_desc=f"[RUN {i}] ENV Relevance",
            progress_position=2,
        )
        health_rel_task = async_add_gpt_score_columns(
            df_health_scored,
            system_prompt_rel,
            user_template,
            "gpt_relevance",
            MAX_LIMIT,
            progress_desc=f"[RUN {i}] HEALTH Relevance",
            progress_position=3,
        )

        df_env_scored, df_health_scored = await asyncio.gather(env_rel_task, health_rel_task)

        # CSV 저장
        env_csv_path = OUTPUT_DIR / "environment_50_scored.csv"
        health_csv_path = OUTPUT_DIR / "health_50_scored.csv"
        df_env_scored.to_csv(env_csv_path, encoding="utf-8-sig", index=False)
        df_health_scored.to_csv(health_csv_path, encoding="utf-8-sig", index=False)

        # 점수 집계
        env_faith = calculate_faithfulness_score(
            build_results_from_df(df_env_scored, "gpt_faithfulness_score"),
            original=False,
        )
        health_faith = calculate_faithfulness_score(
            build_results_from_df(df_health_scored, "gpt_faithfulness_score"),
            original=False,
        )
        env_rel = calculate_relevance_score(
            build_results_from_df(df_env_scored, "gpt_relevance_score"),
            original=False,
        )
        health_rel = calculate_relevance_score(
            build_results_from_df(df_health_scored, "gpt_relevance_score"),
            original=False,
        )

        logger.info(f"[RUN {i}] ENV Faithfulness: {env_faith:.4f}")
        logger.info(f"[RUN {i}] HEALTH Faithfulness: {health_faith:.4f}")
        logger.info(f"[RUN {i}] ENV Relevance: {env_rel:.4f}")
        logger.info(f"[RUN {i}] HEALTH Relevance: {health_rel:.4f}")

        # 점수 파일 저장
        score_path = SCORE_DIR / f"faithfulness_relevance_scores{i}.txt"
        now_kst = datetime.now(tz=ZoneInfo("Asia/Seoul"))
        with score_path.open("w", encoding="utf-8-sig") as f:
            f.write(f"Run: {i}\n")
            f.write(f"현재 시간: {now_kst.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== Faithfulness ===\n")
            f.write(f"ENV_Faithfulness: {env_faith}\n")
            f.write(f"HEALTH_Faithfulness: {health_faith}\n\n")
            f.write("=== Relevance ===\n")
            f.write(f"ENV_Relevance: {env_rel}\n")
            f.write(f"HEALTH_Relevance: {health_rel}\n")

        total_elapsed = time.time() - total_start
        logger.info(f"[RUN {i}] TOTAL TIME: {total_elapsed:.2f}s")
        logger.info(f"[RUN {i}] =============== END ====================")

    logger.info("===== 모든 RUN 종료 =====")


if __name__ == "__main__":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main_async(REPETITION_NUM))
    except Exception:
        logger.exception("Unhandled exception occurred:")
