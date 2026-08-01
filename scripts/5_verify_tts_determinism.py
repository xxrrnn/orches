#!/usr/bin/env python3
"""Fail fast unless warmed TTS policy and PRM responses are byte-stable."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repeat(label: str, url: str, payload: dict[str, Any], normalize, attempts: int) -> str:
    post_json(url, payload)  # Deterministic warm-up; it is not part of the comparison.
    hashes = [stable_hash(normalize(post_json(url, payload))) for _ in range(attempts)]
    if len(set(hashes)) != 1:
        raise RuntimeError(f"{label} is not deterministic after warm-up: {hashes}")
    print(f"{label}: {hashes[0]}")
    return hashes[0]


def policy_seed(base_seed: int, model_name: str, prompt: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{model_name}|{prompt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--policy-address", required=True)
    parser.add_argument("--prm-address", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    if args.attempts < 2:
        raise ValueError("--attempts must be at least 2")
    trace = json.loads(args.fixture.read_text(encoding="utf-8"))
    manifest = json.loads((args.fixture.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    policy_event = trace["generation_events"][0]
    policy_payload = {
        "prompt": policy_event["prompt_text"],
        "n": manifest["tree_max_width"],
        "temperature": manifest["temperature"],
        "top_p": manifest["top_p"],
        "top_k": manifest["top_k"],
        "max_new_tokens": manifest["max_new_tokens"],
        "seed": policy_seed(manifest["seed"], policy_event["policy_model"], policy_event["prompt_text"]),
        "echo": False,
    }
    repeat(
        "policy",
        args.policy_address.rstrip("/") + "/worker_generate",
        policy_payload,
        lambda response: {
            key: response[key]
            for key in ("text", "output_token_ids", "output_token_len", "finish_reason")
        },
        args.attempts,
    )

    reward_candidate = trace["reward_events"][-1]["candidates"][0]
    question = trace["question"]["text"]
    prefix = question + "\n"
    worker_input = reward_candidate["prm_input_text"]
    if not worker_input.startswith(prefix):
        raise RuntimeError("fixture PRM input does not start with its question")
    prm_payload = {"input_str": [[question, worker_input[len(prefix):]]]}
    repeat(
        "prm",
        args.prm_address.rstrip("/") + "/worker_reward_inference",
        prm_payload,
        lambda response: response["reward"],
        args.attempts,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"determinism verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
