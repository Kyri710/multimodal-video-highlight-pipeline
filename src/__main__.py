from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fusion import FusionConfig
from .pipeline import PipelineConfig, run_pipeline
from .ranking import RankingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and rank highlight frames "
            "from a video."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Input video path",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs"),
        help="Output root directory (default: runs)",
    )

    parser.add_argument(
        "--mode",
        choices=("seconds", "frames"),
        default="seconds",
        help=(
            "Interpret --interval as seconds or "
            "frame count"
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Frame extraction interval (default: 1.0)",
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100",
    )

    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=10,
        help="Perceptual hash grouping threshold",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of highlights to select",
    )

    parser.add_argument(
        "--min-gap",
        type=float,
        default=3.0,
        help=(
            "Minimum time gap between highlights "
            "in seconds"
        ),
    )

    parser.add_argument(
        "--cv-weight",
        type=float,
        default=0.4,
        help="CV score fusion weight",
    )

    parser.add_argument(
        "--vl-weight",
        type=float,
        default=0.6,
        help="VLM score fusion weight",
    )

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        extraction_mode=args.mode,
        extraction_interval=args.interval,
        jpg_quality=args.jpg_quality,
        hash_threshold=args.hash_threshold,
        fusion=FusionConfig(
            cv_weight=args.cv_weight,
            vl_weight=args.vl_weight,
        ),
        ranking=RankingConfig(
            top_k=args.top_k,
            min_time_gap_sec=args.min_gap,
        ),
    )

    try:
        result = run_pipeline(
            video_path=args.video,
            output_root=args.output,
            config=config,
        )
    except KeyboardInterrupt:
        print(
            "\nPipeline cancelled by user.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Pipeline failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print()
    print("Pipeline completed.")
    print(f"Candidates: {len(result.frames)}")
    print(
        f"Selected highlights: "
        f"{len(result.selected_frames)}"
    )

    for rank, frame in enumerate(
        result.selected_frames,
        start=1,
    ):
        print(
            f"{rank}. "
            f"score={frame.final_score:.3f}, "
            f"time={frame.timestamp_sec:.3f}s, "
            f"image={frame.image_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())