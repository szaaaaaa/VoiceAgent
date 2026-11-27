"""Local Checkpoint Registry (PRD §3.2) decoupling offline training from live inference."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    model_type: str  # "asr" | "turn_adapter" | "tts"
    path: str
    experiment_id: str
    git_commit: str
    metrics: dict[str, float]
    promoted: bool = False
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    notes: str = ""


class CheckpointRegistry:
    def __init__(self, registry_path: str | Path = "models/registry.json") -> None:
        self.registry_path = Path(registry_path)

    def _read(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"active": {}, "records": []}
        return dict(json.loads(self.registry_path.read_text(encoding="utf-8")))

    def _write(self, data: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, indent=2, ensure_ascii=False)
        self.registry_path.write_text(serialized, encoding="utf-8")

    def register(self, record: CheckpointRecord, *, promote: bool = False) -> CheckpointRecord:
        if not Path(record.path).exists():
            raise FileNotFoundError(f"Cannot register missing checkpoint: {record.path}")
        data = self._read()
        record.promoted = promote
        data["records"].append(asdict(record))
        if promote:
            data["active"][record.model_type] = record.path
        self._write(data)
        return record

    def get_active(self, model_type: str) -> str | None:
        data = self._read()
        val = data.get("active", {}).get(model_type)
        return str(val) if val else None

    def list_records(self, model_type: str | None = None) -> list[dict[str, Any]]:
        records = list(self._read().get("records", []))
        if model_type is None:
            return records
        return [r for r in records if r.get("model_type") == model_type]
