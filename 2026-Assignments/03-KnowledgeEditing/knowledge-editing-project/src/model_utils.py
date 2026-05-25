from __future__ import annotations

from typing import Iterable, List, Optional


def load_hf_model(model_name: str, device_map: str = "auto", dtype: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = "auto"
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_texts(
    model,
    tokenizer,
    prompts: Iterable[str],
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> List[str]:
    import torch

    outputs: List[str] = []
    do_sample = temperature > 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def mock_predictions(rows, mode: str):
    results = []
    for row in rows:
        if mode == "baseline":
            direct = row["ground_truth"]
            rephrase = row["ground_truth"]
        else:
            direct = row["target_new"]
            rephrase = row["target_new"]
        results.append(
            {
                "id": row.get("id"),
                "direct_prediction": direct,
                "rephrase_prediction": rephrase,
                "locality_prediction": row["locality_ground_truth"],
            }
        )
    return results
