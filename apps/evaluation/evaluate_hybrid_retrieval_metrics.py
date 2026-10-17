
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH_PATH = (
    ROOT / "data/evaluation/question_relevant_chunk_ids.csv"
)
RESULTS_PATH = ROOT / "data/evaluation/hybrid_results.json"

K_VALUES = (5, 10, 20)


def main():
    # Load chunk-level ground truth.
    ground_truth = {}

    with GROUND_TRUTH_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file):
            ground_truth[row["question_id"]] = {
                "answerable": row["answerable"].strip().lower() == "true",
                "relevant_ids": {
                    chunk_id.strip()
                    for chunk_id in row["relevant_chunk_ids"].split(";")
                    if chunk_id.strip()
                },
            }

    # Load ranked hybrid retrieval results.
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    per_case = []

    for result in results:
        question_id = result["question_id"]

        if question_id not in ground_truth:
            print(f"WARNING: No ground truth for {question_id}")
            continue

        judgment = ground_truth[question_id]
        relevant_ids = judgment["relevant_ids"]

        # Evaluate only answerable questions with relevance labels.
        if not judgment["answerable"] or not relevant_ids:
            continue

        retrieved_ids = [
            item["chunk_id"]
            for item in result["retrieved_chunks"]
        ]

        # Prevent duplicate retrieved IDs from inflating metrics.
        retrieved_ids = list(dict.fromkeys(retrieved_ids))

        row = {"id": question_id}

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )

        row["mrr"] = (
            1 / first_relevant_rank if first_relevant_rank else 0.0
        )

        for k in K_VALUES:
            top_k = retrieved_ids[:k]
            hits = sum(chunk_id in relevant_ids for chunk_id in top_k)

            # Precision@K uses K as the denominator.
            row[f"precision@{k}"] = hits / k

            # Recall@K measures coverage of all known relevant chunks.
            row[f"recall@{k}"] = hits / len(relevant_ids)

        per_case.append(row)

    print("\nPER-QUESTION METRICS")

    for row in per_case:
        print(
            f"{row['id']}: "
            + ", ".join(
                f"{metric}={value:.3f}"
                for metric, value in row.items()
                if metric != "id"
            )
        )

    print("\nMACRO-AVERAGE METRICS")

    if not per_case:
        print("No answerable questions with relevance labels found.")
        return

    metrics = (
        [f"precision@{k}" for k in K_VALUES]
        + [f"recall@{k}" for k in K_VALUES]
        + ["mrr"]
    )

    for metric in metrics:
        print(
            f"{metric}: "
            f"{mean(row[metric] for row in per_case):.3f}"
        )

    print(f"\nEvaluated questions: {len(per_case)}")


if __name__ == "__main__":
    main()
