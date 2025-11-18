# Faithfulness (사실충실도)
def calculate_faithfulness_score(results, original=True):
    D = sum(item["score"] for item in results)
    N = len(results)
    if original:
        result = (D / N) * 100
        return result
    result = (D / (N * 5)) * 100
    return result


# Relevance (관련성)
def calculate_relevance_score(results, original=True):
    N = len(results)
    D = sum(r["score"] for r in results)
    if original:
        result = D / N * 100
        return result
    result = D / (N * 5) * 100
    return result


sample_dummy = [
    {
        "id": "Q1",
        "score": 5,
    },
    {
        "id": "Q2",
        "score": 4,
    },
    {
        "id": "Q3",
        "score": 3,
    },
    {
        "id": "Q4",
        "score": 5,
    },
    {
        "id": "Q5",
        "score": 4,
    },
    {
        "id": "Q6",
        "score": 5,
    },
    {
        "id": "Q7",
        "score": 3,
    },
    {
        "id": "Q8",
        "score": 4,
    },
]

if __name__ == "__main__":
    score_faithfulness_original = calculate_faithfulness_score(results=sample_dummy, original=True)
    score_relevance_original = calculate_relevance_score(results=sample_dummy, original=True)
    score_faithfulness_normalization = calculate_faithfulness_score(results=sample_dummy, original=False)
    score_relevance_normalization = calculate_relevance_score(results=sample_dummy, original=False)

    print("Faithfulness:", score_faithfulness_original)
    print("Relevance:", score_relevance_original)
    print("Faithfulness_normalized:", score_faithfulness_normalization)
    print("Relevance_normalized:", score_relevance_normalization)
