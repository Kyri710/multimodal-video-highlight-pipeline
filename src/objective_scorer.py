from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import CVMetrics, FrameRecord


@dataclass(frozen=True)
class ObjectiveScorerConfig:
    sharpness_weight: float = 0.35
    brightness_weight: float = 0.15
    contrast_weight: float = 0.15
    saturation_weight: float = 0.05
    exposure_weight: float = 0.20
    clipping_weight: float = 0.10

    # Laplacian variance
    sharpness_bad: float = 20.0
    sharpness_good: float = 250.0

    # brightness: 0～255
    brightness_hard_low: float = 25.0
    brightness_ideal_low: float = 70.0
    brightness_ideal_high: float = 190.0
    brightness_hard_high: float = 230.0

    # grayscale std
    contrast_hard_low: float = 8.0
    contrast_ideal_low: float = 35.0
    contrast_ideal_high: float = 90.0
    contrast_hard_high: float = 128.0

    # HSV saturation: 0～255
    saturation_hard_low: float = 0.0
    saturation_ideal_low: float = 20.0
    saturation_ideal_high: float = 180.0
    saturation_hard_high: float = 255.0

    # overexposure / underexposure
    exposure_good_max: float = 0.05
    exposure_bad_min: float = 0.80

    # black / white clipping
    clipping_good_max: float = 0.01
    clipping_bad_min: float = 0.50

    # 清晰度中，相对排名占多少
    relative_sharpness_weight: float = 0.40

    def __post_init__(self) -> None:
        weights = [
            self.sharpness_weight,
            self.brightness_weight,
            self.contrast_weight,
            self.saturation_weight,
            self.exposure_weight,
            self.clipping_weight,
        ]

        if any(
            not math.isfinite(weight) or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "All objective score weights must be "
                "finite and >= 0"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one objective score weight "
                "must be > 0"
            )

        if not (
            self.sharpness_bad
            < self.sharpness_good
        ):
            raise ValueError(
                "sharpness_bad must be less than "
                "sharpness_good"
            )

        if not (
            self.brightness_hard_low
            < self.brightness_ideal_low
            <= self.brightness_ideal_high
            < self.brightness_hard_high
        ):
            raise ValueError(
                "Invalid brightness thresholds"
            )

        if not (
            self.contrast_hard_low
            < self.contrast_ideal_low
            <= self.contrast_ideal_high
            < self.contrast_hard_high
        ):
            raise ValueError(
                "Invalid contrast thresholds"
            )

        if not (
            self.saturation_hard_low
            < self.saturation_ideal_low
            <= self.saturation_ideal_high
            < self.saturation_hard_high
        ):
            raise ValueError(
                "Invalid saturation thresholds"
            )

        if not (
            0.0 <= self.exposure_good_max
            < self.exposure_bad_min
            <= 1.0
        ):
            raise ValueError(
                "Invalid exposure thresholds"
            )

        if not (
            0.0 <= self.clipping_good_max
            < self.clipping_bad_min
            <= 1.0
        ):
            raise ValueError(
                "Invalid clipping thresholds"
            )

        if not (
            0.0
            <= self.relative_sharpness_weight
            <= 1.0
        ):
            raise ValueError(
                "relative_sharpness_weight must be "
                "between 0 and 1"
            )

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _higher_is_better(
    value: float,
    bad: float,
    good: float,
) -> float:
    return _clamp01(
        (value - bad) / (good - bad)
    )


def _lower_is_better(
    value: float,
    good_max: float,
    bad_min: float,
) -> float:
    if value <= good_max:
        return 1.0

    if value >= bad_min:
        return 0.0

    return 1.0 - (
        (value - good_max)
        / (bad_min - good_max)
    )

def _range_score(
    value: float,
    hard_low: float,
    ideal_low: float,
    ideal_high: float,
    hard_high: float,
) -> float:
    if value <= hard_low:
        return 0.0

    if value < ideal_low:
        return (
            (value - hard_low)
            / (ideal_low - hard_low)
        )

    if value <= ideal_high:
        return 1.0

    if value < hard_high:
        return (
            (hard_high - value)
            / (hard_high - ideal_high)
        )

    return 0.0

def _validate_metrics(
    frame: FrameRecord,
) -> CVMetrics:
    metrics = frame.cv_metrics

    if metrics is None:
        raise ValueError(
            f"CV metrics missing: {frame.frame_id}"
        )

    numeric_values = [
        metrics.sharpness,
        metrics.brightness,
        metrics.contrast,
        metrics.saturation,
        metrics.overexposure_ratio,
        metrics.underexposure_ratio,
        metrics.black_clip_ratio,
        metrics.white_clip_ratio,
    ]

    if not all(
        math.isfinite(value)
        for value in numeric_values
    ):
        raise ValueError(
            f"CV metrics contain non-finite values: "
            f"{frame.frame_id}"
        )

    if metrics.sharpness < 0:
        raise ValueError(
            f"sharpness must be >= 0: "
            f"{frame.frame_id}"
        )

    if metrics.contrast < 0:
        raise ValueError(
            f"contrast must be >= 0: "
            f"{frame.frame_id}"
        )

    if not 0.0 <= metrics.brightness <= 255.0:
        raise ValueError(
            f"brightness must be between 0 and 255: "
            f"{frame.frame_id}"
        )

    if not 0.0 <= metrics.saturation <= 255.0:
        raise ValueError(
            f"saturation must be between 0 and 255: "
            f"{frame.frame_id}"
        )

    ratios = [
        metrics.overexposure_ratio,
        metrics.underexposure_ratio,
        metrics.black_clip_ratio,
        metrics.white_clip_ratio,
    ]

    if not all(0.0 <= ratio <= 1.0 for ratio in ratios):
        raise ValueError(
            f"CV ratios must be between 0 and 1: "
            f"{frame.frame_id}"
        )

    return metrics

def _relative_sharpness_score(
    value: float,
    sharpness_values: list[float],
) -> float | None:
    if len(sharpness_values) < 2:
        return None

    low = float(
        np.percentile(sharpness_values, 10)
    )
    high = float(
        np.percentile(sharpness_values, 90)
    )

    if high - low < 1e-9:
        return None

    return _clamp01(
        (value - low) / (high - low)
    )

def _compute_components(
    metrics: CVMetrics,
    sharpness_values: list[float],
    config: ObjectiveScorerConfig,
) -> dict[str, float]:
    absolute_sharpness = _higher_is_better(
        metrics.sharpness,
        config.sharpness_bad,
        config.sharpness_good,
    )

    relative_sharpness = (
        _relative_sharpness_score(
            metrics.sharpness,
            sharpness_values,
        )
    )

    if relative_sharpness is None:
        sharpness_score = absolute_sharpness
    else:
        relative_weight = (
            config.relative_sharpness_weight
        )

        sharpness_score = (
            absolute_sharpness
            * (1.0 - relative_weight)
            + relative_sharpness
            * relative_weight
        )

    brightness_score = _range_score(
        metrics.brightness,
        config.brightness_hard_low,
        config.brightness_ideal_low,
        config.brightness_ideal_high,
        config.brightness_hard_high,
    )

    contrast_score = _range_score(
        metrics.contrast,
        config.contrast_hard_low,
        config.contrast_ideal_low,
        config.contrast_ideal_high,
        config.contrast_hard_high,
    )

    saturation_score = _range_score(
        metrics.saturation,
        config.saturation_hard_low,
        config.saturation_ideal_low,
        config.saturation_ideal_high,
        config.saturation_hard_high,
    )

    overexposure_score = _lower_is_better(
        metrics.overexposure_ratio,
        config.exposure_good_max,
        config.exposure_bad_min,
    )

    underexposure_score = _lower_is_better(
        metrics.underexposure_ratio,
        config.exposure_good_max,
        config.exposure_bad_min,
    )

    # 取曝光两项中较差的一项，防止严重过曝
    # 被正常欠曝分数平均掉。
    exposure_score = min(
        overexposure_score,
        underexposure_score,
    )

    black_clip_score = _lower_is_better(
        metrics.black_clip_ratio,
        config.clipping_good_max,
        config.clipping_bad_min,
    )

    white_clip_score = _lower_is_better(
        metrics.white_clip_ratio,
        config.clipping_good_max,
        config.clipping_bad_min,
    )

    clipping_score = min(
        black_clip_score,
        white_clip_score,
    )

    return {
        "sharpness": sharpness_score,
        "brightness": brightness_score,
        "contrast": contrast_score,
        "saturation": saturation_score,
        "exposure": exposure_score,
        "clipping": clipping_score,
    }

def score_cv_frames(
    frames: list[FrameRecord],
    config: ObjectiveScorerConfig | None = None,
) -> list[FrameRecord]:
    config = config or ObjectiveScorerConfig()

    passed_frames = [
        frame
        for frame in frames
        if frame.passed_filter
    ]

    if not passed_frames:
        return frames

    metrics_by_frame: list[
        tuple[FrameRecord, CVMetrics]
    ] = []

    for frame in passed_frames:
        metrics = _validate_metrics(frame)
        metrics_by_frame.append((frame, metrics))

    sharpness_values = [
        metrics.sharpness
        for _, metrics in metrics_by_frame
    ]

    weights = {
        "sharpness": config.sharpness_weight,
        "brightness": config.brightness_weight,
        "contrast": config.contrast_weight,
        "saturation": config.saturation_weight,
        "exposure": config.exposure_weight,
        "clipping": config.clipping_weight,
    }

    total_weight = sum(weights.values())

    for frame, metrics in metrics_by_frame:
        components = _compute_components(
            metrics,
            sharpness_values,
            config,
        )

        normalized_score = sum(
            components[name] * weight
            for name, weight in weights.items()
        ) / total_weight

        frame.cv_score = 10.0 * normalized_score

        frame.cv_score_components = {
            name: value * 10.0
            for name, value in components.items()
        }

    # 被 CV filter 拒绝的帧明确没有 cv_score。
    for frame in frames:
        if not frame.passed_filter:
            frame.cv_score = None

    return frames