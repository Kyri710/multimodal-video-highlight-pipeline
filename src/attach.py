from .models import FrameRecord
from .image_quality import compute_cv_metrics


def attach_cv_metrics(
    frames: list[FrameRecord],
) -> list[FrameRecord]:

    for frame in frames:
        frame.cv_metrics = compute_cv_metrics(
            frame.image_path
        )

    return frames