from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class VideoMetadata:
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int

@dataclass
class ExtractionResult:
    video_path: str
    output_dir: str
    metadata: VideoMetadata
    frames: list[FrameRecord]

    @property
    def saved_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CVMetrics:
    sharpness: float
    brightness: float
    contrast: float
    saturation: float
    overexposure_ratio: float
    underexposure_ratio: float
    black_clip_ratio: float
    white_clip_ratio: float


@dataclass
class VLScores:
    composition: float
    subject_prominence: float
    pose_expression: float
    background_cleanliness: float
    lighting_aesthetic: float
    moment_quality: float
    aesthetic: float
    keep: bool
    reason: str


@dataclass
class FrameRecord:
    frame_id: str
    image_path: Path
    frame_index: int
    timestamp_sec: float

    group_id: int | None = None

    cv_metrics: CVMetrics | None = None

    passed_filter: bool = True
    reject_reasons: list[str] = field(default_factory=list)

    cv_score: float | None = None

    vl_scores: VLScores | None = None
    vl_score: float | None = None

    final_score: float | None = None
    cv_score_components: dict[str, float] = field(default_factory=dict)

@dataclass
class HighlightResult:
    video_path: Path
    metadata: VideoMetadata
    frames: list[FrameRecord]
    selected_frames: list[FrameRecord]