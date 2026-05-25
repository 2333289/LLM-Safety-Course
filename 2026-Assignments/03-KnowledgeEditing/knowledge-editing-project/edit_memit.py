import argparse
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
    parser = argparse.ArgumentParser(description="Run batch MEMIT editing with EasyEdit.")
    parser.add_argument("--data", default="data/memit_500_synthetic.json")
    parser.add_argument("--output", default="outputs/memit_predictions.jsonl")
    parser.add_argument("--metrics-output", default="outputs/memit_easyedit_metrics.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--hparams", default="configs/MEMIT/qwen2.5-0.5b.yaml")
    parser.add_argument("--backend", choices=["easyedit", "mock"], default="easyedit")
    parser.add_argument("--limit", type=int, default=None)
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
        import torch
        from easyeditor import BaseEditor, MEMITHyperParams
    except ImportError as exc:
        raise RuntimeError(
            "EasyEdit or torch is not installed. Install EasyEdit first, "
            "or run this script with `--backend mock` for pipeline verification."
        ) from exc

    hparams = MEMITHyperParams.from_hparams(args.hparams)
    hparams = prepare_hparams_model(hparams, args.model)

    editor = BaseEditor.from_hparams(hparams)
    editor.model_name = args.model
    locality_inputs = {
        "neighborhood": {
            "prompt": [row["locality_prompt"] for row in rows],
            "ground_truth": [row["locality_ground_truth"] for row in rows],
        }
    }

    start = time.time()
    peak_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    metrics, edited_model, _ = editor.edit(
        prompts=[row["prompt"] for row in rows],
        target_new=[row["target_new"] for row in rows],
        ground_truth=[row["ground_truth"] for row in rows],
        subject=[row["subject"] for row in rows],
        rephrase_prompts=[row["rephrase_prompt"] for row in rows],
        locality_inputs=locality_inputs,
        keep_original_weight=False,
    )
    elapsed = time.time() - start
    peak_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    tokenizer = getattr(editor, "tok", None) or getattr(editor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Could not find tokenizer on EasyEdit editor object.")

    direct = generate_texts(edited_model, tokenizer, [r["prompt"] for r in rows], args.max_new_tokens)
    rephrase = generate_texts(edited_model, tokenizer, [r["rephrase_prompt"] for r in rows], args.max_new_tokens)
    locality = generate_texts(edited_model, tokenizer, [r["locality_prompt"] for r in rows], args.max_new_tokens)
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

    save_json(
        args.metrics_output,
        {
            "elapsed_seconds": elapsed,
            "cuda_peak_memory_allocated_mb": round((peak_after - peak_before) / 1024 / 1024, 2),
            "easyedit_metrics": metrics,
        },
    )
    return predictions


def main():
    args = parse_args()
    rows = load_json(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    require_fields(rows, REQUIRED_FIELDS)

    if args.backend == "mock":
        predictions = mock_predictions(rows, mode="edited")
        save_json(args.metrics_output, {"backend": "mock", "note": "Mock run for pipeline verification."})
    else:
        predictions = run_easyedit(rows, args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output, predictions)
    print(f"Saved {len(predictions)} MEMIT predictions to {args.output}")


if __name__ == "__main__":
    main()
