from dataclasses import dataclass

from .models import FrameRecord


@dataclass
class CVFilterConfig:
    black_brightness_max: float = 10.0
    black_underexposure_min: float = 0.95

    white_brightness_min: float = 245.0
    white_overexposure_min: float = 0.95

    underexposed_brightness_max: float = 25.0
    severe_underexposure_min: float = 0.80

    overexposed_brightness_min: float = 230.0
    severe_overexposure_min: float = 0.80

    min_contrast: float = 8.0
    severe_blur_threshold: float = 12.0
    low_information_sharpness: float = 20.0

    black_clip_min: float = 0.98
    white_clip_min: float = 0.98

    def __post_init__(self) -> None:
        ratio_fields = {
            "black_underexposure_min":
                self.black_underexposure_min,
            "white_overexposure_min":
                self.white_overexposure_min,
            "severe_underexposure_min":
                self.severe_underexposure_min,
            "severe_overexposure_min":
                self.severe_overexposure_min,
            "black_clip_min":
                self.black_clip_min,
            "white_clip_min":
                self.white_clip_min,
        }

        for name, value in ratio_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if self.severe_blur_threshold < 0:
            raise ValueError(
                "severe_blur_threshold must be >= 0"
            )

def apply_cv_filter(
    frames: list[FrameRecord],
    config: CVFilterConfig,
) -> list[FrameRecord]:

    for frame in frames:
        metrics = frame.cv_metrics

        if metrics is None:
            raise ValueError(
                f"CV metrics missing: {frame.frame_id}"
            )

        reasons: list[str] = []

        if (
            metrics.black_clip_ratio >= config.black_clip_min
            or (
            metrics.brightness <= config.black_brightness_max
            and metrics.underexposure_ratio
            >= config.black_underexposure_min
            )
        ):
            reasons.append("almost_black")

        elif (
            metrics.white_clip_ratio >= config.white_clip_min
            or (
                metrics.brightness >= config.white_brightness_min
                and metrics.overexposure_ratio
                >= config.white_overexposure_min
            )
        ):
            reasons.append("almost_white")

        elif (
            metrics.brightness
            <= config.underexposed_brightness_max
            and metrics.underexposure_ratio
            >= config.severe_underexposure_min
        ):
            reasons.append("severely_underexposed")

        elif (
            metrics.brightness
            >= config.overexposed_brightness_min
            and metrics.overexposure_ratio
            >= config.severe_overexposure_min
        ):
            reasons.append("severely_overexposed")

        if metrics.sharpness <= config.severe_blur_threshold:
            reasons.append("severely_blurred")

        if (
            metrics.contrast <= config.min_contrast
            and metrics.sharpness
            <= config.low_information_sharpness
        ):
            reasons.append("low_information")

        frame.reject_reasons = reasons
        frame.passed_filter = not reasons

    return frames