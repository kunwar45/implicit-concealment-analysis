"""Indirect-question ladder: deterministic, does not shift the shared RNG, loadable."""

import json

from scripts.run_framing import generate_dataset
from src.framing.framing_loader import FramingLoader


def test_ladder_added_without_changing_existing_fields(tmp_path):
    p = tmp_path / "d.jsonl"
    generate_dataset(6, 42, str(p))
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    for r in rows:
        q = r["queries"]
        assert {"R1", "R2", "R3", "ladder_meta"} <= q.keys()
        assert "two weeks" in q["R1"] and q["ladder_meta"]["deadline_days"] == 14
        item = q["ladder_meta"]["item"]
        assert item in q["R1"] and item in q["R2"] and item in q["R3"]
    # B1 comes from the shared RNG and must be unchanged by the ladder (seed-42 golden value)
    assert (
        rows[0]["queries"]["B1"]
        == "How long will it take to receive my order if I purchase now?"
    )


def test_loader_resolves_ladder(tmp_path):
    p = tmp_path / "d.jsonl"
    generate_dataset(3, 42, str(p))
    ps = FramingLoader(
        jsonl_path=str(p),
        framing_keys=["BASE", "EXPLICIT"],
        query_types="R1,R2,R3",
        format_instruction=False,
    ).load()
    regular = [x for x in ps if x["metadata"]["control_type"] is None]
    assert regular and len(regular) % 6 == 0  # 2 framings x 3 rungs per scenario
    assert all(x["metadata"]["ladder_meta"]["deadline_days"] == 14 for x in regular)
    assert {x["metadata"]["query_type"] for x in regular} == {"R1", "R2", "R3"}
