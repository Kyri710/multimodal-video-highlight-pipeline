from __future__ import annotations

import math
from dataclasses import dataclass

from .models import FrameRecord


@dataclass(frozen=True)
class RankingConfig:
    top_k: int = 5
    min_time_gap_sec: float = 3.0
    require_vlm_keep: bool = True
    unique_group: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if not math.isfinite(self.min_time_gap_sec):
            raise ValueError(
                "min_time_gap_sec must be finite"
            )

        if self.min_time_gap_sec < 0:
            raise ValueError(
                "min_time_gap_sec must be >= 0"
            )

def _is_rankable(
    frame: FrameRecord,
    config: RankingConfig,
) -> bool:
    if not frame.passed_filter:
        return False

    if frame.final_score is None:
        return False

    if config.require_vlm_keep:
        if frame.vl_scores is None:
            return False

        if not frame.vl_scores.keep:
            return False

    return True

def _validated_final_score(
    frame: FrameRecord,
) -> float:
    value = frame.final_score

    if value is None:
        raise ValueError(
            f"Final score missing: {frame.frame_id}"
        )

    if isinstance(value, bool):
        raise ValueError(
            f"Invalid final score for {frame.frame_id}"
        )

    score = float(value)

    if not math.isfinite(score):
        raise ValueError(
            f"Final score must be finite: "
            f"{frame.frame_id}"
        )

    return score

def rank_frames(
    frames: list[FrameRecord],
    config: RankingConfig | None = None,
) -> list[FrameRecord]:
    config = config or RankingConfig()

    eligible_frames = [
        frame
        for frame in frames
        if _is_rankable(frame, config)
    ]

    # final_score 高的排前面。
    # 分数相同时，视频中更早出现的帧排前面。
    ranked_frames = sorted(
        eligible_frames,
        key=lambda frame: (
            -_validated_final_score(frame),
            frame.timestamp_sec,
            frame.frame_index,
        ),
    )

    selected_frames: list[FrameRecord] = []
    selected_group_ids: set[int] = set()

    for frame in ranked_frames:
        # 同一个相似帧组最多保留一张。
        if (
            config.unique_group
            and frame.group_id is not None
            and frame.group_id in selected_group_ids
        ):
            continue

        # 避免入选图片在时间上过于接近。
        too_close = any(
            abs(
                frame.timestamp_sec
                - selected.timestamp_sec
            ) < config.min_time_gap_sec
            for selected in selected_frames
        )

        if too_close:
            continue

        selected_frames.append(frame)

        if frame.group_id is not None:
            selected_group_ids.add(frame.group_id)

        if len(selected_frames) >= config.top_k:
            break

    return selected_frames