"""Pipeline.run(workers>1) keeps input order and runs every prompt once."""

import random
import time

from src.pipeline.pipeline import Pipeline
from src.pipeline.step import PipelineStep


class SlowEcho(PipelineStep):
    name = "slow_echo"

    def run(self, result):
        time.sleep(random.uniform(0, 0.02))
        result.final_answer = result.prompt.upper()
        return result


def test_workers_preserve_order():
    prompts = [{"id": f"p{i}", "text": f"q{i}"} for i in range(40)]
    out = Pipeline([SlowEcho()]).run(prompts, workers=8)
    assert [r.prompt_id for r in out] == [p["id"] for p in prompts]
    assert [r.final_answer for r in out] == [f"Q{i}" for i in range(40)]
