from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "vector_index.json"


class JsonVectorStore:
    def __init__(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        self.path = path
        self._lock = Lock()

    def add_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        with self._lock:
            existing = self._read_records()
            existing.extend(records)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(existing, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.path)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []

        with self._lock:
            records = self._read_records()

        scored_records = []
        for record in records:
            score = _cosine_similarity(
                query_embedding,
                record["embedding"],
            )
            scored_records.append({
                **record,
                "score": score,
            })

        return sorted(
            scored_records,
            key=lambda record: record["score"],
            reverse=True,
        )[:limit]

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        return json.loads(
            self.path.read_text(encoding="utf-8")
        )


def _cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match.")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    return dot_product / (left_norm * right_norm)
