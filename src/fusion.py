from __future__ import annotations

import math
from dataclasses import dataclass

from .models import FrameRecord


@dataclass(frozen=True)
class FusionConfig:
    cv_weight: float = 0.4
    vl_weight: float = 0.6

    score_min: float = 0.0
    score_max: float = 10.0

    reject_if_vlm_keep_false: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.cv_weight):
            raise ValueError(
                "cv_weight must be finite"
            )

        if not math.isfinite(self.vl_weight):
            raise ValueError(
                "vl_weight must be finite"
            )

        if self.cv_weight < 0:
            raise ValueError(
                "cv_weight must be >= 0"
            )

        if self.vl_weight < 0:
            raise ValueError(
                "vl_weight must be >= 0"
            )

        if self.cv_weight + self.vl_weight <= 0:
            raise ValueError(
                "At least one fusion weight must be > 0"
            )

        if self.score_min >= self.score_max:
            raise ValueError(
                "score_min must be less than score_max"
            )

def _validate_score(
    name: str,
    value: float | None,
    config: FusionConfig,
) -> float:
    if value is None:
        raise ValueError(f"{name} is missing")

    # bool 是 int 的子类，需要单独排除。
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a number, not bool"
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if not config.score_min <= value <= config.score_max:
        raise ValueError(
            f"{name} must be between "
            f"{config.score_min} and {config.score_max}, "
            f"got {value}"
        )

    return value


def fuse_frame_scores(
    frame: FrameRecord,
    config: FusionConfig,
) -> FrameRecord:
    # CV filter 已拒绝的图片不参与融合。
    if not frame.passed_filter:
        frame.final_score = None
        return frame

    if frame.vl_scores is None:
        raise ValueError(
            f"VLM scores missing: {frame.frame_id}"
        )

    # VLM 明确判定不保留。
    if (
        config.reject_if_vlm_keep_false
        and not frame.vl_scores.keep
    ):
        frame.final_score = None
        return frame

    cv_score = _validate_score(
        "cv_score",
        frame.cv_score,
        config,
    )

    vl_score = _validate_score(
        "vl_score",
        frame.vl_score,
        config,
    )

    total_weight = (
        config.cv_weight
        + config.vl_weight
    )

    normalized_cv_weight = (
        config.cv_weight / total_weight
    )

    normalized_vl_weight = (
        config.vl_weight / total_weight
    )

    frame.final_score = (
        cv_score * normalized_cv_weight
        + vl_score * normalized_vl_weight
    )

    return frame

def fuse_scores(
    frames: list[FrameRecord],
    config: FusionConfig | None = None,
) -> list[FrameRecord]:
    config = config or FusionConfig()

    for frame in frames:
        fuse_frame_scores(frame, config)

    return frames