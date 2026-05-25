import argparse
from statistics import mean

from src.io_utils import contains_answer, load_json, load_jsonl, require_fields, save_json


REQUIRED_FIELDS = [
    "id",
    "target_new",
    "locality_ground_truth",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ES, PS and NS metrics.")
    parser.add_argument("--data", default="data/custom_facts.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="outputs/metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    facts = load_json(args.data)
    require_fields(facts, REQUIRED_FIELDS)
    pred_rows = load_jsonl(args.predictions)
    pred_by_id = {row["id"]: row for row in pred_rows}

    details = []
    for fact in facts:
        pred = pred_by_id.get(fact["id"])
        if pred is None:
            raise ValueError(f"Missing prediction for id={fact['id']}")
        es = contains_answer(pred.get("direct_prediction", ""), fact["target_new"])
        ps = contains_answer(pred.get("rephrase_prediction", ""), fact["target_new"])
        ns = contains_answer(pred.get("locality_prediction", ""), fact["locality_ground_truth"])
        details.append(
            {
                "id": fact["id"],
                "ES": int(es),
                "PS": int(ps),
                "NS": int(ns),
                "direct_prediction": pred.get("direct_prediction", ""),
                "rephrase_prediction": pred.get("rephrase_prediction", ""),
                "locality_prediction": pred.get("locality_prediction", ""),
            }
        )

    summary = {
        "num_examples": len(details),
        "ES_percent": round(mean(row["ES"] for row in details) * 100, 2),
        "PS_percent": round(mean(row["PS"] for row in details) * 100, 2),
        "NS_percent": round(mean(row["NS"] for row in details) * 100, 2),
        "details": details,
    }
    save_json(args.output, summary)
    print(f"ES={summary['ES_percent']} PS={summary['PS_percent']} NS={summary['NS_percent']}")
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
