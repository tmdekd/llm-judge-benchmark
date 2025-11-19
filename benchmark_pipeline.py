import os
import time
import json
import logging
from pathlib import Path

import pandas as pd
import dotenv
from openai import OpenAI
from traceback import print_exc
from calculate_function.calculate_functions import calculate_faithfulness_score, calculate_relevance_score

# =========================
# Logger 설정: log 폴더 생성
# =========================
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler(
            os.path.join(LOG_DIR, "llm_evaluation.log"),
            encoding="utf-8",
        ),  # 파일 저장
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
client = OpenAI(api_key=api_key)


def load_text(path: str) -> str:
    """주어진 경로에서 텍스트 파일을 읽어 문자열로 반환"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def score_row(
    llm_answer: str,
    retrieved_result: str,
    system_prompt: str,
    user_template: str,
) -> tuple[int, str]:
    """
    하나의 LLM 응답(llm_answer)과 그에 해당하는 retrieved_result(근거)를 받아서
    GPT-5.1 모델을 호출해 1~5점 사이의 score와 scoring_reason을 반환한다.
    반환값: (score: int, scoring_reason: str)
    """

    # user_template.txt의 {retrieved_result}, {llm_answer} 자리에
    # 실제 값(retrieved_result, llm_answer)을 포맷팅하여 user 메시지 내용을 만든다.
    user_content = user_template.format(
        retrieved_result=str(retrieved_result),
        llm_answer=str(llm_answer),
    )

    # OpenAI Chat Completions API 호출
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        # system_prompt에서 JSON 포맷을 요구하고 있으므로
        # 이 옵션으로 JSON object만 반환하도록 강제
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    score = int(data.get("score", 0))
    scoring_reason = data.get("scoring_reason", "")

    return score, scoring_reason


def add_gpt_score_columns(
    df: pd.DataFrame,
    system_prompt: str,
    user_template: str,
    metric_prefix: str,
) -> pd.DataFrame:
    """
    주어진 DataFrame(df)의 각 행에 대해
    - llm_answer
    - retrieved_result
    값을 GPT-5.1로 평가하여
    {metric_prefix}_score, {metric_prefix}_scoring_reason 컬럼을 추가한 뒤,
    수정된 DataFrame을 반환한다.

    예:
    metric_prefix="gpt_faithfulness" ->
        gpt_faithfulness_score, gpt_faithfulness_scoring_reason
    metric_prefix="gpt_relevance" ->
        gpt_relevance_score, gpt_relevance_scoring_reason
    """

    scores: list[int] = []
    reasons: list[str] = []

    for idx, row in df.iterrows():
        llm_answer = row.get("llm_answer", "")
        retrieved_result = row.get("retrieved_result", "")

        logger.info(f"Row {idx} 평가 중... ({metric_prefix})")

        try:
            score, reason = score_row(
                llm_answer=llm_answer,
                retrieved_result=retrieved_result,
                system_prompt=system_prompt,
                user_template=user_template,
            )
        except Exception as e:
            logger.error(f"Row {idx} scoring failed ({metric_prefix}): {e}")
            score = 0
            reason = f"error: {e}"

        scores.append(score)
        reasons.append(reason)

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


def main():
    # ===== 경로 설정 =====
    PROMPT_FOLDER_PATH = "prompt/"
    SYSTEM_PROMPT_FAITH_PATH = PROMPT_FOLDER_PATH + "system_prompt_faithfulness.txt"
    SYSTEM_PROMPT_REL_PATH = PROMPT_FOLDER_PATH + "system_prompt_relevance.txt"
    USER_TEMPLATE_PATH = PROMPT_FOLDER_PATH + "user_template.txt"

    FOLDER_PATH = "data/test_dataset/"
    OUTPUT_PATH = FOLDER_PATH + "output/"
    ENV_CSV_PATH = FOLDER_PATH + "environment_50.csv"
    HEALTH_CSV_PATH = FOLDER_PATH + "health_50.csv"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ===== 데이터 로드 =====
    df_env = pd.read_csv(ENV_CSV_PATH)
    df_health = pd.read_csv(HEALTH_CSV_PATH)

    logger.info(f"환경 CSV rows: {len(df_env)}")
    logger.info(f"건강 CSV rows: {len(df_health)}")

    # ===== 프롬프트 로드 =====
    system_prompt_faith = load_text(SYSTEM_PROMPT_FAITH_PATH)
    system_prompt_rel = load_text(SYSTEM_PROMPT_REL_PATH)
    user_template = load_text(USER_TEMPLATE_PATH)

    logger.info("System Prompt (Faithfulness) loaded.")
    logger.info("System Prompt (Relevance) loaded.")
    logger.info("User Template loaded.")

    # ========= 환경형 데이터 =========
    logger.info("=[INFO] 주거 환경 데이터 기반 LLM 응답 평가 시작...")

    # 1) 사실충실도(Faithfulness) 평가 - 환경 CSV
    env_faith_start = time.time()
    df_env_scored = add_gpt_score_columns(
        df=df_env,
        system_prompt=system_prompt_faith,
        user_template=user_template,
        metric_prefix="gpt_faithfulness",
    )
    env_faith_end = time.time()
    logger.info(f"[TIME] 환경 CSV - Faithfulness 평가 소요 시간: {env_faith_end - env_faith_start:.2f}초")

    # 2) 관련성(Relevance) 평가 - 환경 CSV
    env_rel_start = time.time()
    df_env_scored = add_gpt_score_columns(
        df=df_env_scored,
        system_prompt=system_prompt_rel,
        user_template=user_template,
        metric_prefix="gpt_relevance",
    )
    env_rel_end = time.time()
    logger.info(f"[TIME] 환경 CSV - Relevance 평가 소요 시간: {env_rel_end - env_rel_start:.2f}초")

    # ========= 건강형 데이터 =========
    logger.info("=[INFO] 건강 데이터 기반 LLM 응답 평가 시작...")

    # 1) 사실충실도(Faithfulness) 평가 - 건강 CSV
    health_faith_start = time.time()
    df_health_scored = add_gpt_score_columns(
        df=df_health,
        system_prompt=system_prompt_faith,
        user_template=user_template,
        metric_prefix="gpt_faithfulness",
    )
    health_faith_end = time.time()
    logger.info(f"[TIME] 건강 CSV - Faithfulness 평가 소요 시간: {health_faith_end - health_faith_start:.2f}초")

    # 2) 관련성(Relevance) 평가 - 건강 CSV
    health_rel_start = time.time()
    df_health_scored = add_gpt_score_columns(
        df=df_health_scored,
        system_prompt=system_prompt_rel,
        user_template=user_template,
        metric_prefix="gpt_relevance",
    )
    health_rel_end = time.time()
    logger.info(f"[TIME] 건강 CSV - Relevance 평가 소요 시간: {health_rel_end - health_rel_start:.2f}초")

    # ========= CSV 저장 =========
    env_output_path = OUTPUT_PATH + "environment_50_scored.csv"
    df_env_scored.to_csv(env_output_path, encoding="utf-8-sig", index=False)
    logger.info(f"[INFO] 주거 환경 데이터 기반 LLM 응답 스코어 결과 저장 완료: {env_output_path}")

    health_output_path = OUTPUT_PATH + "health_50_scored.csv"
    df_health_scored.to_csv(health_output_path, encoding="utf-8-sig", index=False)
    logger.info(f"[INFO] 건강 데이터 기반 LLM 응답 스코어 결과 저장 완료: {health_output_path}")

    # ========= 최종 점수 집계 (calculate_functions 적용) =========
    # 환경 CSV
    env_faith_results = build_results_from_df(df_env_scored, "gpt_faithfulness_score")
    env_rel_results = build_results_from_df(df_env_scored, "gpt_relevance_score")

    env_faith_orig = calculate_faithfulness_score(env_faith_results, original=True)
    env_faith_norm = calculate_faithfulness_score(env_faith_results, original=False)
    env_rel_orig = calculate_relevance_score(env_rel_results, original=True)
    env_rel_norm = calculate_relevance_score(env_rel_results, original=False)

    logger.info(f"[ENV] Faithfulness(original): {env_faith_orig}, " f"Faithfulness(normalized): {env_faith_norm}")
    logger.info(f"[ENV] Relevance(original): {env_rel_orig}, " f"Relevance(normalized): {env_rel_norm}")

    # 건강 CSV
    health_faith_results = build_results_from_df(df_health_scored, "gpt_faithfulness_score")
    health_rel_results = build_results_from_df(df_health_scored, "gpt_relevance_score")

    health_faith_orig = calculate_faithfulness_score(health_faith_results, original=True)
    health_faith_norm = calculate_faithfulness_score(health_faith_results, original=False)
    health_rel_orig = calculate_relevance_score(health_rel_results, original=True)
    health_rel_norm = calculate_relevance_score(health_rel_results, original=False)

    logger.info(
        f"[HEALTH] Faithfulness(original): {health_faith_orig}, " f"Faithfulness(normalized): {health_faith_norm}"
    )
    logger.info(f"[HEALTH] Relevance(original): {health_rel_orig}, " f"Relevance(normalized): {health_rel_norm}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception occurred:")
        print_exc()
