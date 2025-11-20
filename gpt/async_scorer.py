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
) -> Tuple[int, str]:
    client = get_client()
    response = await client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    data = json.loads(response.choices[0].message.content)
    return int(data.get("score", 0)), data.get("scoring_reason", "")


async def async_add_gpt_score_columns(
    df: pd.DataFrame,
    system_prompt: str,
    user_template: str,
    metric_prefix: str,
    max_limit: int,
    progress_desc: str,
    progress_position: int = 0,
) -> pd.DataFrame:

    scores: List[int] = [0] * len(df)
    reasons: List[str] = [""] * len(df)
    semaphore = asyncio.Semaphore(max_limit)

    async def worker(pos: int, row: pd.Series):
        llm_answer = row.get("llm_answer", "")
        retrieved_result = row.get("retrieved_result", "")

        user_prompt = user_template.format(
            retrieved_result=str(retrieved_result),
            llm_answer=str(llm_answer),
        )

        async with semaphore:
            try:
                score, reason = await score_row(
                    llm_answer=llm_answer,
                    retrieved_result=retrieved_result,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as e:
                logger.error(f"Row {pos} scoring failed ({metric_prefix}): {e}")
                score, reason = 0, f"error: {e}"

        return pos, score, reason

    tasks = [worker(pos, row) for pos, (_, row) in enumerate(df.iterrows())]

    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=progress_desc,
        position=progress_position,
        leave=True,
    ):
        pos, score, reason = await coro
        scores[pos] = score
        reasons[pos] = reason

    df[f"{metric_prefix}_score"] = scores
    df[f"{metric_prefix}_scoring_reason"] = reasons
    return df


def build_results_from_df(df: pd.DataFrame, score_column: str):
    return [{"id": idx, "score": int(score) if score is not None else 0} for idx, score in df[score_column].items()]
