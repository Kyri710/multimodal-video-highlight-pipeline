from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .attach import attach_cv_metrics
from .extraction import ExtractMode, extract_frames
from .frame_filter import (
    CVFilterConfig,
    apply_cv_filter,
)
from .frame_selector import (
    group_similar_frames,
    select_sharpest,
)
from .fusion import FusionConfig, fuse_scores
from .models import HighlightResult
from .objective_scorer import (
    ObjectiveScorerConfig,
    score_cv_frames,
)
from .ranking import RankingConfig, rank_frames
from .vl_scorer import VLScorer


@dataclass(frozen=True)
class PipelineConfig:
    extraction_mode: ExtractMode = "seconds"
    extraction_interval: float = 1.0
    jpg_quality: int = 95

    hash_threshold: int = 10

    cv_filter: CVFilterConfig = field(
        default_factory=CVFilterConfig
    )

    fusion: FusionConfig = field(
        default_factory=FusionConfig
    )

    ranking: RankingConfig = field(
        default_factory=RankingConfig
    )

    objective: ObjectiveScorerConfig = field(
        default_factory=ObjectiveScorerConfig
    )

def run_pipeline(
    video_path: str | Path,
    output_root: str | Path,
    config: PipelineConfig | None = None,
    vl_scorer: VLScorer | None = None,
) -> HighlightResult:
    config = config or PipelineConfig()

    # 1. 抽帧
    extraction = extract_frames(
        video_path=video_path,
        output_root=output_root,
        mode=config.extraction_mode,
        interval=config.extraction_interval,
        jpg_quality=config.jpg_quality,
    )

    # 2. 相似帧分组
    groups = group_similar_frames(
        extraction.frames,
        hash_threshold=config.hash_threshold,
    )

    # 3. 每组选择最清晰帧
    candidates = select_sharpest(groups)

    # 4. 只对候选帧计算完整 CV 指标
    attach_cv_metrics(candidates)

    # 5. CV 硬过滤
    apply_cv_filter(
        candidates,
        config.cv_filter,
    )

    passed_candidates = [
        frame
        for frame in candidates
        if frame.passed_filter
    ]

    # 没有候选帧时，不创建 VLM 客户端。
    if passed_candidates:
        # 6. 计算 CV 客观总分
        score_cv_frames(
            passed_candidates,
            config.objective,
        )

        # 7. VLM 主观评分
        scorer = vl_scorer or VLScorer()
        scorer.score_frames(passed_candidates)

        # 8. 双源融合
        fuse_scores(
            passed_candidates,
            config.fusion,
        )

        # 9. 排序和时间去重
        selected_frames = rank_frames(
            passed_candidates,
            config.ranking,
        )
    else:
        selected_frames = []

    # 10. 汇总结果
    result = HighlightResult(
        video_path=Path(extraction.video_path),
        metadata=extraction.metadata,
        frames=candidates,
        selected_frames=selected_frames,
    )

    # 11. 保存 JSON
    result_path = (
        Path(extraction.output_dir)
        / "result.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return result

