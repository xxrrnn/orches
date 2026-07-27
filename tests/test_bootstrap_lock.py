from __future__ import annotations

from orches.bootstrap import load_repository_locks


def test_bootstrap_reads_all_frozen_repositories() -> None:
    locks = load_repository_locks()

    assert set(locks) == {
        "AttAcc simulator",
        "AttAcc Ramulator2 gitlink",
        "AttAcc Ramulator2 build base",
        "Duplex LLMSimulator",
        "Duplex Ramulator2",
        "Compute-optimal TTS",
        "LLaVA-o1",
    }
    assert locks["AttAcc simulator"].commit == (
        "c60005143a6b492d7ef83231723386478b59a506"
    )
    assert locks["AttAcc Ramulator2 build base"].commit == (
        "b7c70275f04126c647edb989270cc429776955d1"
    )
