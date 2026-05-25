import argparse
from pathlib import Path

from src.io_utils import load_json, require_fields, save_jsonl
from src.model_utils import generate_texts, load_hf_model, mock_predictions


REQUIRED_FIELDS = [
    "id",
    "prompt",
    "target_new",
    "ground_truth",
    "rephrase_prompt",
    "locality_prompt",
    "locality_ground_truth",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run pre-edit baseline generation.")
    parser.add_argument("--data", default="data/custom_facts.json")
    parser.add_argument("--output", default="outputs/baseline_predictions.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--backend", choices=["hf", "mock"], default="hf")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_json(args.data)
    require_fields(rows, REQUIRED_FIELDS)

    if args.backend == "mock":
        predictions = mock_predictions(rows, mode="baseline")
    else:
        model, tokenizer = load_hf_model(args.model, device_map=args.device_map, dtype=args.dtype)
        direct = generate_texts(model, tokenizer, [r["prompt"] for r in rows], args.max_new_tokens)
        rephrase = generate_texts(model, tokenizer, [r["rephrase_prompt"] for r in rows], args.max_new_tokens)
        locality = generate_texts(model, tokenizer, [r["locality_prompt"] for r in rows], args.max_new_tokens)
        predictions = []
        for row, d_pred, r_pred, l_pred in zip(rows, direct, rephrase, locality):
            predictions.append(
                {
                    "id": row["id"],
                    "direct_prediction": d_pred,
                    "rephrase_prediction": r_pred,
                    "locality_prediction": l_pred,
                }
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output, predictions)
    print(f"Saved {len(predictions)} baseline predictions to {args.output}")


if __name__ == "__main__":
    main()
