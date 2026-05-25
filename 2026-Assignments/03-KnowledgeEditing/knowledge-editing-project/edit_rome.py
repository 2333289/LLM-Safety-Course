import argparse
import json
import time
from pathlib import Path

from src.io_utils import load_json, require_fields, save_json, save_jsonl
from src.model_utils import generate_texts, mock_predictions


REQUIRED_FIELDS = [
    "id",
    "subject",
    "prompt",
    "target_new",
    "ground_truth",
    "rephrase_prompt",
    "locality_prompt",
    "locality_ground_truth",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-fact ROME editing with EasyEdit.")
    parser.add_argument("--data", default="data/custom_facts.json")
    parser.add_argument("--output", default="outputs/rome_predictions.jsonl")
    parser.add_argument("--metrics-output", default="outputs/rome_easyedit_metrics.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hparams", default="configs/ROME/qwen2.5-0.5b.yaml")
    parser.add_argument("--backend", choices=["easyedit", "mock"], default="easyedit")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def easyedit_supported_model_name(model_name: str) -> bool:
    lower = model_name.lower()
    return any(key in lower for key in ["t5", "gpt-3.5", "gpt", "llama", "baichuan", "chatglm", "internlm"])


def prepare_hparams_model(hparams, model_name: str):
    if easyedit_supported_model_name(model_name):
        hparams.model_name = model_name
        return hparams

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config._name_or_path = model_name
    hparams.model_name = (model, tokenizer)
    return hparams


def run_easyedit(rows, args):
    try:
        from easyeditor import BaseEditor, ROMEHyperParams
    except ImportError as exc:
        raise RuntimeError(
            "EasyEdit is not installed. Install it with `pip install -e EasyEdit` "
            "or run this script with `--backend mock` for pipeline verification."
        ) from exc

    predictions = []
    raw_metrics = []
    start = time.time()

    for idx, row in enumerate(rows, start=1):
        # Reload the base model for every request so each ROME edit starts from
        # the same unedited weights, as required by the assignment.
        hparams = ROMEHyperParams.from_hparams(args.hparams)
        hparams = prepare_hparams_model(hparams, args.model)
        editor = BaseEditor.from_hparams(hparams)
        editor.model_name = args.model
        locality_inputs = {
            "neighborhood": {
                "prompt": [row["locality_prompt"]],
                "ground_truth": [row["locality_ground_truth"]],
            }
        }
        metrics, edited_model, _ = editor.edit(
            prompts=[row["prompt"]],
            target_new=[row["target_new"]],
            ground_truth=[row["ground_truth"]],
            subject=[row["subject"]],
            rephrase_prompts=[row["rephrase_prompt"]],
            locality_inputs=locality_inputs,
            keep_original_weight=False,
        )
        tokenizer = getattr(editor, "tok", None) or getattr(editor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Could not find tokenizer on EasyEdit editor object.")

        direct, rephrase, locality = generate_texts(
            edited_model,
            tokenizer,
            [row["prompt"], row["rephrase_prompt"], row["locality_prompt"]],
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append(
            {
                "id": row["id"],
                "direct_prediction": direct,
                "rephrase_prediction": rephrase,
                "locality_prediction": locality,
            }
        )
        raw_metrics.append({"id": row["id"], "easyedit_metrics": metrics})
        print(f"[ROME] edited {idx}/{len(rows)}: {row['id']}")

    elapsed = time.time() - start
    save_json(args.metrics_output, {"elapsed_seconds": elapsed, "items": raw_metrics})
    return predictions


def main():
    args = parse_args()
    rows = load_json(args.data)
    require_fields(rows, REQUIRED_FIELDS)

    if args.backend == "mock":
        predictions = mock_predictions(rows, mode="edited")
        save_json(args.metrics_output, {"backend": "mock", "note": "Mock run for pipeline verification."})
    else:
        predictions = run_easyedit(rows, args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output, predictions)
    print(f"Saved {len(predictions)} ROME predictions to {args.output}")


if __name__ == "__main__":
    main()
