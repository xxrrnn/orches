from __future__ import annotations

import pytest

from orches.errors import ConfigurationError
from orches.predictor import (
    CandidateScorePath,
    PrmLayerPartition,
    ScoreAggregation,
    SpeculationSession,
    SpeculationState,
    predict_candidate,
)


def test_history_alignment_can_correct_candidate_ranking() -> None:
    candidates = (
        CandidateScorePath(
            "a",
            current_small_prm_score=0.4,
            historical_small_prm_scores=(0.0,),
            historical_large_prm_scores=(1.0,),
        ),
        CandidateScorePath(
            "b",
            current_small_prm_score=0.5,
            historical_small_prm_scores=(1.0,),
            historical_large_prm_scores=(0.0,),
        ),
    )

    unaligned = predict_candidate(
        candidates,
        history_alignment=False,
        aggregation=ScoreAggregation.MEAN,
    )
    aligned = predict_candidate(
        candidates,
        history_alignment=True,
        aggregation=ScoreAggregation.MEAN,
    )

    assert unaligned.candidate_id == "b"
    assert aligned.candidate_id == "a"
    assert aligned.margin > 0


def test_candidate_ties_use_stable_id_order() -> None:
    candidates = (
        CandidateScorePath("b", 0.5, (), ()),
        CandidateScorePath("a", 0.5, (), ()),
    )

    decision = predict_candidate(
        candidates,
        history_alignment=True,
        aggregation=ScoreAggregation.LAST,
    )

    assert decision.candidate_id == "a"


def test_correct_speculation_commits_tokens_and_saves_latency() -> None:
    session = SpeculationSession("candidate-a", 5, 0.25, 1024)

    resolution = session.resolve(
        "candidate-a",
        rollback_fixed_latency_s=0.01,
        regeneration_latency_per_token_s=0.05,
    )

    assert session.state is SpeculationState.COMMITTED
    assert resolution.final_candidate_id == "candidate-a"
    assert resolution.committed_speculative_tokens == 5
    assert resolution.saved_latency_s == 0.25
    assert resolution.discarded_kv_bytes == 0


def test_wrong_speculation_rolls_back_to_large_prm_branch() -> None:
    session = SpeculationSession("predicted", 5, 0.25, 1024)

    resolution = session.resolve(
        "large-prm-choice",
        rollback_fixed_latency_s=0.01,
        regeneration_latency_per_token_s=0.05,
    )

    assert session.state is SpeculationState.ROLLED_BACK
    assert resolution.final_candidate_id == "large-prm-choice"
    assert resolution.discarded_speculative_tokens == 5
    assert resolution.discarded_kv_bytes == 5 * 1024
    assert resolution.rollback_latency_s == 0.01
    assert resolution.regeneration_latency_s == 0.25


def test_speculation_cannot_be_resolved_twice() -> None:
    session = SpeculationSession("a", 1, 0.1, 128)
    session.resolve(
        "a",
        rollback_fixed_latency_s=0.0,
        regeneration_latency_per_token_s=0.0,
    )

    with pytest.raises(ConfigurationError, match="only once"):
        session.resolve(
            "a",
            rollback_fixed_latency_s=0.0,
            regeneration_latency_per_token_s=0.0,
        )


def test_fig13_partition_reuses_first_ten_layers_without_extra_model_work() -> None:
    partition = PrmLayerPartition(total_layers=32, small_prefix_layers=10)

    assert partition.small_prefix_layers == 10
    assert partition.large_suffix_layers == 22
    assert partition.small_prefix_layers + partition.large_suffix_layers == 32
    assert partition.adds_model_layers is False
