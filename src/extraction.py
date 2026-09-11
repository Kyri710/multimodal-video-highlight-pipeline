from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Literal
from .models import VideoMetadata,FrameRecord,ExtractionResult

import cv2

ExtractMode = Literal["seconds", "frames"]


def extract_frames(
    video_path: str | Path,
    output_root: str | Path,
    *,
    mode: ExtractMode = "seconds",
    interval: float = 1.0,
    jpg_quality: int = 95,
) -> ExtractionResult:
    

    video_path = Path(video_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    #从视频中按时间间隔或帧间隔抽取 JPG 图片。

    if not video_path.exists():
        raise FileNotFoundError(
            f"Input video does not exist: {video_path}"
        )

    if not video_path.is_file():
        raise ValueError(
            f"Input path is not a file: {video_path}"
        )

    if mode not in ("seconds", "frames"):
        raise ValueError(
            f"Unsupported mode: {mode!r}. "
            "Expected 'seconds' or 'frames'."
        )

    if interval <= 0:
        raise ValueError(
            f"interval must be > 0, got {interval}"
        )

    if not 0 <= jpg_quality <= 100:
        raise ValueError(
            f"jpg_quality must be between 0 and 100, "
            f"got {jpg_quality}"
        )

    # ---------- 创建本次运行的独立目录 ----------

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        output_root
        / f"{video_path.stem}_{run_id}"
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    # ---------- 打开视频 ----------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV failed to open video: {video_path}"
        )

    try:
        # ---------- 获取视频元信息 ----------

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise ValueError(
                f"Invalid FPS reported by video: {fps}"
            )

        if total_frames < 0:
            raise ValueError(
                f"Invalid frame count reported by video: "
                f"{total_frames}"
            )

        duration_sec = (
            total_frames / fps
            if total_frames > 0
            else 0.0
        )

        metadata = VideoMetadata(
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )

        # ---------- 计算抽帧间隔 ----------

        if mode == "seconds":
            frame_interval = max(
                1,
                round(interval * fps)
            )

        else:
            if not float(interval).is_integer():
                raise ValueError(
                    "interval must be an integer "
                    "when mode='frames'"
                )

            frame_interval = int(interval)

        # ---------- 抽帧 ----------

        extracted_frames: list[FrameRecord] = []

        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / fps

                filename = (
                    f"frame_{frame_index:08d}_"
                    f"{timestamp_sec:010.3f}s.jpg"
                )

                output_path = output_dir / filename

                success = cv2.imwrite(
                    str(output_path),
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        jpg_quality,
                    ],
                )

                if not success:
                    raise RuntimeError(
                        f"Failed to save frame to: "
                        f"{output_path}"
                    )

                extracted_frames.append(
                    FrameRecord(
                        frame_id=output_path.stem,
                        image_path=output_path,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                    )
                )

            frame_index += 1

    finally:
        cap.release()

    return ExtractionResult(
        video_path=str(video_path),
        output_dir=str(output_dir),
        metadata=metadata,
        frames=extracted_frames,
    )