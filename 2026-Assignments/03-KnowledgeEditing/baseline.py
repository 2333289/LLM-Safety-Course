import time

from src.real_easyedit import (
    DATA_DIR,
    RESULT_DIR,
    contains,
    load_hf_model,
    load_json,
    generate,
    runtime,
    save_json,
    write_log,
)


def main() -> None:
    start = time.perf_counter()
    facts = load_json(DATA_DIR / "custom_facts.json")
    model, tok = load_hf_model()
    records = []
    for fact in facts:
        answer = generate(model, tok, fact["prompt"])
        rephrase_answer = generate(model, tok, fact["rephrase_prompt"])
        locality_answer = generate(model, tok, fact["locality_prompt"])
        records.append(
            {
                **fact,
                "model_answer": answer,
                "rephrase_answer": rephrase_answer,
                "locality_answer": locality_answer,
                "target_hit": contains(answer, fact["target_new"]),
                "ground_truth_hit": contains(answer, fact["ground_truth"]),
                "locality_hit": contains(locality_answer, fact["locality_ground_truth"]),
            }
        )
    n = len(records)
    metrics = {
        "cases": n,
        "target_hit_rate": round(sum(r["target_hit"] for r in records) / n, 4),
        "old_knowledge_hit_rate": round(sum(r["ground_truth_hit"] for r in records) / n, 4),
        "locality_generation_rate": round(sum(r["locality_hit"] for r in records) / n, 4),
    }
    stats = runtime(start)
    save_json(RESULT_DIR / "baseline_results.json", {"records": records, "metrics": metrics, "runtime": stats})
    write_log(
        "baseline",
        [
            "Task 1 Baseline Evaluation (Qwen2.5-0.5B-Instruct)",
            f"cases: {metrics['cases']}",
            f"target answer hit before editing: {metrics['target_hit_rate'] * 100:.1f}%",
            f"old answer hit before editing: {metrics['old_knowledge_hit_rate'] * 100:.1f}%",
            f"locality generation sanity check: {metrics['locality_generation_rate'] * 100:.1f}%",
            f"elapsed seconds: {stats['elapsed_seconds']}",
            f"peak memory MB: {stats['peak_memory_mb']}",
            "saved: artifacts/results/baseline_results.json",
        ],
    )


if __name__ == "__main__":
    main()
