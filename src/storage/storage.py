from __future__ import annotations

import dataclasses
import json
import os
import uuid
from datetime import datetime, timezone

from src.config import Config
from src.pipeline.result import PromptResult


class ResultStorage:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

    # Fields never written to disk (results are pushed to a public HF repo).
    _SECRET_FIELDS = ("together_api_key",)

    def _config_dict(self) -> dict:
        cfg = dataclasses.asdict(self.config)
        for key in self._SECRET_FIELDS:
            cfg.pop(key, None)
        return cfg

    def save(
        self,
        results: list[PromptResult],
        run_metadata: dict | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        short_id = run_id.split("-")[0]
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp_str}_{short_id}.json"
        path = os.path.join(self.config.output_dir, filename)

        payload = {
            "run_id": run_id,
            "run_timestamp": now.isoformat(),
            "config": self._config_dict(),
            "results": [r.to_dict() for r in results],
        }
        if run_metadata:
            payload["run_metadata"] = run_metadata

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        return path
