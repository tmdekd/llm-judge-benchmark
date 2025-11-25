# gpt/async_scorer.py
import asyncio
import json
import logging
from typing import Tuple, List
import pandas as pd
from tqdm import tqdm

from .llm_client import get_client

logger = logging.getLogger(__name__)


async def score_row(
    llm_answer: str,
    retrieved_result: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gpt-5.1",
) -> Tuple[int, str]:
    client = get_client()
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        timeout=120,
    )

    data = json.loads(response.choices[0].message.content)
    return int(data.get("score", 0)), data.get("scoring_reason", "")


async def async_add_gpt_score_columns_two_dfs(
    df_env: pd.DataFrame,
    df_health: pd.DataFrame,
    system_prompt: str,
    user_template: str,
    metric_prefix: str,
    max_limit: int,
    progress_desc: str,
    model_name: str = "gpt-5.1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ENV, HEALTH 두 개의 DataFrame을 동시에 평가하고
    전체 row 개수(ENV+HEALTH)를 기준으로 tqdm 진행바 1개만 표시.
    """

    env_scores: List[int] = [0] * len(df_env)
    env_reasons: List[str] = [""] * len(df_env)
    health_scores: List[int] = [0] * len(df_health)
    health_reasons: List[str] = [""] * len(df_health)

    semaphore = asyncio.Semaphore(max_limit)

    async def worker(is_env: bool, pos: int, row: pd.Series):
        llm_answer = row.get("llm_answer", "")
        retrieved_result = row.get("retrieved_result", "")

        user_prompt = user_template.format(
            retrieved_result=str(retrieved_result),
            llm_answer=str(llm_answer),
        )

        max_retries = 3
        retry_delay = 2

        async with semaphore:
            for attempt in range(1, max_retries + 1):
                try:
                    score, reason = await score_row(
                        llm_answer=llm_answer,
                        retrieved_result=retrieved_result,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model_name=model_name,
                    )
                    return is_env, pos, score, reason

                except Exception as e:
                    logger.error(
                        f"[{'ENV' if is_env else 'HEALTH'}][row {pos}] "
                        f"{metric_prefix} 평가 실패 (재시도 {attempt}/{max_retries}): {e}"
                    )

                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                    else:
                        return is_env, pos, 0, f"{max_retries}회 재시도 후 실패: {e}"

        return is_env, pos, score, reason

    tasks = []

    # ENV 쪽 row 작업 추가
    for pos, (_, row) in enumerate(df_env.iterrows()):
        tasks.append(worker(True, pos, row))

    # HEALTH 쪽 row 작업 추가
    for pos, (_, row) in enumerate(df_health.iterrows()):
        tasks.append(worker(False, pos, row))

    total_rows = len(tasks)

    # 진행바 1개: 전체 row(ENV + HEALTH) 기준
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=total_rows,
        desc=progress_desc,
        leave=False,
        ncols=100,
        dynamic_ncols=False,
    ):
        is_env, pos, score, reason = await coro
        if is_env:
            env_scores[pos] = score
            env_reasons[pos] = reason
        else:
            health_scores[pos] = score
            health_reasons[pos] = reason

    df_env[f"{metric_prefix}_score"] = env_scores
    df_env[f"{metric_prefix}_scoring_reason"] = env_reasons
    df_health[f"{metric_prefix}_score"] = health_scores
    df_health[f"{metric_prefix}_scoring_reason"] = health_reasons

    return df_env, df_health


def build_results_from_df(df: pd.DataFrame, score_column: str):
    return [{"id": idx, "score": int(score) if score is not None else 0} for idx, score in df[score_column].items()]
