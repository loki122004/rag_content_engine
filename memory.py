import json
import os
from collections import Counter
from typing import List

class PersistentFailureMemory:
    def __init__(self, storage_path: str = "pipeline_memory.json"):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"failure_counts": {}, "lessons_shipped": 0, "anti_patterns": []}, f)

    def record_failures(self, failed_checkpoints: List[str], reasons: List[str]):
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        counts = Counter(data.get("failure_counts", {}))
        for ck in failed_checkpoints:
            counts[ck] += 1
        data["failure_counts"] = dict(counts)

        for reason in reasons:
            if reason not in data["anti_patterns"]:
                data["anti_patterns"].append(reason)
        data["anti_patterns"] = data["anti_patterns"][-15:]

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_success(self):
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["lessons_shipped"] = data.get("lessons_shipped", 0) + 1
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_learned_anti_patterns(self) -> str:
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anti_patterns = data.get("anti_patterns", [])
        if not anti_patterns:
            return "No previous failure patterns recorded."
        return "\n".join(f"- WATCH OUT: {item}" for item in anti_patterns[-5:])
