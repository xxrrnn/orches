"""Technique 2A candidate scoring with explicit history alignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ..errors import ConfigurationError


class ScoreAggregation(str, Enum):
    """Explicit alternatives for the paper's undisclosed path aggregation."""

    MEAN = "mean"
    MINIMUM = "minimum"
    LAST = "last"


@dataclass(frozen=True)
class CandidateScorePath:
    """Current small-PRM score plus completed historical score pairs."""

    candidate_id: str
    current_small_prm_score: float
    historical_small_prm_scores: tuple[float, ...]
    historical_large_prm_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ConfigurationError("candidate_id must not be empty")
        if len(self.historical_small_prm_scores) != len(
            self.historical_large_prm_scores
        ):
            raise ConfigurationError(
                "historical small- and large-PRM score arrays must have equal length"
            )
        scores = (
            self.current_small_prm_score,
            *self.historical_small_prm_scores,
            *self.historical_large_prm_scores,
        )
        if any(not isfinite(score) for score in scores):
            raise ConfigurationError("candidate PRM scores must be finite")

    def score_series(self, *, history_alignment: bool) -> tuple[float, ...]:
        history = (
            self.historical_large_prm_scores
            if history_alignment
            else self.historical_small_prm_scores
        )
        return (*history, self.current_small_prm_score)


@dataclass(frozen=True)
class PredictionDecision:
    """Ranked small-PRM prediction and its score margin."""

    candidate_id: str
    aggregate_scores: dict[str, float]
    margin: float
    history_alignment: bool
    aggregation: ScoreAggregation


def _aggregate(scores: tuple[float, ...], method: ScoreAggregation) -> float:
    if not scores:
        raise ConfigurationError("cannot aggregate an empty score path")
    if method is ScoreAggregation.MEAN:
        return sum(scores) / len(scores)
    if method is ScoreAggregation.MINIMUM:
        return min(scores)
    return scores[-1]


def predict_candidate(
    candidates: tuple[CandidateScorePath, ...],
    *,
    history_alignment: bool,
    aggregation: ScoreAggregation,
) -> PredictionDecision:
    """Predict one branch with deterministic candidate-ID tie breaking."""

    if not candidates:
        raise ConfigurationError("at least one predictor candidate is required")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ConfigurationError("predictor candidate IDs must be unique")

    aggregate_scores = {
        candidate.candidate_id: _aggregate(
            candidate.score_series(history_alignment=history_alignment),
            aggregation,
        )
        for candidate in candidates
    }
    ranked = sorted(
        aggregate_scores,
        key=lambda candidate_id: (-aggregate_scores[candidate_id], candidate_id),
    )
    margin = (
        aggregate_scores[ranked[0]] - aggregate_scores[ranked[1]]
        if len(ranked) > 1
        else float("inf")
    )
    return PredictionDecision(
        candidate_id=ranked[0],
        aggregate_scores=aggregate_scores,
        margin=margin,
        history_alignment=history_alignment,
        aggregation=aggregation,
    )
