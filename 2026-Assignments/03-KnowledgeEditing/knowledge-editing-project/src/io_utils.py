import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def load_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Union[str, Path], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", text)
    return text


def contains_answer(prediction: str, answer: str) -> bool:
    return normalize_text(answer) in normalize_text(prediction)


def require_fields(rows: List[Dict[str, Any]], fields: Iterable[str]) -> None:
    missing = []
    for idx, row in enumerate(rows):
        for field in fields:
            if field not in row or row[field] in ("", None):
                missing.append(f"row {idx}: {field}")
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
