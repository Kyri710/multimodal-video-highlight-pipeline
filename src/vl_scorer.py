from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .models import FrameRecord, VLScores


@dataclass(frozen=True)
class VLScorerConfig:
    model: str = "qwen3-vl-plus"
    base_url: str = (
        "https://llm-tb4j5b0f8x85tqni.cn-beijing."
        "maas.aliyuncs.com/compatible-mode/v1"
    )
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout: float = 60.0
    max_retries: int = 2

class VLScorer:
    def __init__(
        self,
        config: VLScorerConfig | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config or VLScorerConfig()

        # 测试时允许注入假客户端，不需要真实 API Key。
        if client is not None:
            self.client = client
            return

        load_dotenv()

        api_key = os.getenv(self.config.api_key_env)

        if not api_key:
            raise RuntimeError(
                f"Missing environment variable: "
                f"{self.config.api_key_env}"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
    @staticmethod
    def encode_image(image_path: Path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        with image_path.open("rb") as file:
            return base64.b64encode(file.read()).decode("ascii")

    @staticmethod
    def get_mime_type(image_path: Path) -> str:
        suffix = image_path.suffix.lower()

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }

        try:
            return mime_types[suffix]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported image type: {suffix}"
            ) from exc

    def score_frame(self, frame: FrameRecord) -> FrameRecord:
        if not frame.passed_filter:
            return frame

        if frame.cv_metrics is None:
            raise ValueError(
                f"CV metrics missing: {frame.frame_id}"
            )

        image_base64 = self.encode_image(frame.image_path)
        mime_type = self.get_mime_type(frame.image_path)

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{mime_type};base64,"
                                    f"{image_base64}"
                                )
                            },
                        },
                        {
                            "type": "text",
                            "text": self._build_prompt(frame),
                        },
                    ],
                }
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                f"Empty VLM response: {frame.frame_id}"
            )

        data = self._parse_response(content)
        scores = self._make_scores(data)

        frame.vl_scores = scores
        frame.vl_score = self._calculate_vl_score(scores)

        return frame

    def score_frames(
        self,
        frames: list[FrameRecord],
    ) -> list[FrameRecord]:

        for frame in frames:
            if frame.passed_filter:
                self.score_frame(frame)

        return frames

    def _build_prompt(self, frame: FrameRecord) -> str:
        metrics = frame.cv_metrics

        return f"""
    你是视频高光画面评审器。
    请结合图片和下面的客观指标进行评价：
    {json.dumps(
        metrics.__dict__ if metrics else {},
        ensure_ascii=False
    )}

    每项给出 0 到 10 的数字，分数越高越好。

    只返回一个合法 JSON 对象：
    {{
    "composition": 0,
    "subject_prominence": 0,
    "pose_expression": 0,
    "background_cleanliness": 0,
    "lighting_aesthetic": 0,
    "moment_quality": 0,
    "aesthetic": 0,
    "keep": true,
    "reason": ""
    }}
    """

    def _parse_response(self, content: str) -> dict:
        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1])

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "VLM response is not valid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "VLM response must be a JSON object"
            )

        return data

    def _make_scores(self, data: dict) -> VLScores:
        score_fields = [
            "composition",
            "subject_prominence",
            "pose_expression",
            "background_cleanliness",
            "lighting_aesthetic",
            "moment_quality",
            "aesthetic",
        ]

        values: dict[str, float] = {}

        for field_name in score_fields:
            if field_name not in data:
                raise ValueError(
                    f"Missing VLM score: {field_name}"
                )

            value = data[field_name]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"{field_name} must be a number"
                )

            value = float(value)

            if not 0.0 <= value <= 10.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 10"
                )

            values[field_name] = value

        keep = data.get("keep")
        reason = data.get("reason")

        if not isinstance(keep, bool):
            raise ValueError("keep must be a boolean")

        if not isinstance(reason, str):
            raise ValueError("reason must be a string")

        return VLScores(
            **values,
            keep=keep,
            reason=reason.strip(),
        )

    @staticmethod
    def _calculate_vl_score(scores: VLScores) -> float:
        values = [
            scores.composition,
            scores.subject_prominence,
            scores.pose_expression,
            scores.background_cleanliness,
            scores.lighting_aesthetic,
            scores.moment_quality,
            scores.aesthetic,
        ]

        return sum(values) / len(values)