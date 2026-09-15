from __future__ import annotations

import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image, ImageChops

from src.pipeline import PipelineConfig, run_pipeline
from src.ranking import RankingConfig
from src.inpaint import InpaintService

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

INPAINT_SERVICE = InpaintService()

def to_pil_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()

    if isinstance(value, np.ndarray):
        return Image.fromarray(value)

    if isinstance(value, (str, Path)):
        with Image.open(value) as opened:
            return opened.copy()

    raise ValueError(
        f"Unsupported image value: {type(value)}"
    )

def extract_mask(
    editor_value: dict | None,
) -> Image.Image:
    if not editor_value:
        raise gr.Error("编辑器中没有图片")

    background_value = editor_value.get("background")
    layers = editor_value.get("layers") or []

    if background_value is None:
        raise gr.Error("编辑器背景图片缺失")

    if not layers:
        raise gr.Error("请先在图片上涂出重绘区域")

    background = to_pil_image(background_value)
    mask = Image.new(
        "L",
        background.size,
        color=0,
    )

    for layer_value in layers:
        if layer_value is None:
            continue

        layer = to_pil_image(layer_value).convert("RGBA")

        if layer.size != background.size:
            layer = layer.resize(
                background.size,
                Image.Resampling.NEAREST,
            )

        # 白色固定画笔：亮度与透明度共同决定蒙版。
        luminance = layer.convert("L")
        alpha = layer.getchannel("A")

        layer_mask = ImageChops.multiply(
            luminance,
            alpha,
        )

        mask = ImageChops.lighter(
            mask,
            layer_mask,
        )

    if mask.getbbox() is None:
        raise gr.Error("蒙版为空，请涂出重绘区域")

    return mask

def run_inpaint(
    editor_value: dict | None,
    selected_image_path: str | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    strength: float,
    steps: int,
    guidance_scale: float,
):
    if not selected_image_path:
        raise gr.Error("请先选择一张高光图片")

    if not prompt or not prompt.strip():
        raise gr.Error("请输入重绘提示词")

    mask = extract_mask(editor_value)

    source_path = Path(selected_image_path)
    output_dir = source_path.parent / "edits"

    try:
        result = INPAINT_SERVICE.inpaint(
            image_path=source_path,
            mask_image=mask,
            output_dir=output_dir,
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip(),
            seed=int(seed),
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
        )
    except Exception as exc:
        raise gr.Error(
            f"重绘失败：{exc}"
        ) from exc

    return (
        str(result.image_path),
        str(result.metadata_path),
        f"重绘完成：`{result.image_path.name}`",
    )


def analyze_video(
    video_path: str | None,
    interval: float,
    top_k: int,
    min_gap: float,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("请先上传视频")

    progress(
        0,
        desc="正在分析视频，请勿关闭页面",
    )

    config = PipelineConfig(
        extraction_mode="seconds",
        extraction_interval=float(interval),
        ranking=RankingConfig(
            top_k=int(top_k),
            min_time_gap_sec=float(min_gap),
        ),
    )

    try:
        result = run_pipeline(
            video_path=video_path,
            output_root=RUNS_DIR,
            config=config,
        )
    except Exception as exc:
        raise gr.Error(
            f"视频分析失败：{exc}"
        ) from exc

    gallery_items = []
    frame_records = []

    for rank, frame in enumerate(
        result.selected_frames,
        start=1,
    ):
        caption = (
            f"#{rank} | "
            f"final={frame.final_score:.3f} | "
            f"time={frame.timestamp_sec:.3f}s"
        )

        gallery_items.append(
            (str(frame.image_path), caption)
        )

        frame_records.append(
            {
                "rank": rank,
                "frame_id": frame.frame_id,
                "image_path": str(frame.image_path),
                "timestamp_sec": frame.timestamp_sec,
                "cv_score": frame.cv_score,
                "vl_score": frame.vl_score,
                "final_score": frame.final_score,
            }
        )

    progress(1, desc="分析完成")

    status = (
        f"分析完成：共得到 {len(result.frames)} 张候选图，"
        f"最终选中 {len(result.selected_frames)} 张高光图。"
    )

    return gallery_items, frame_records, status

def select_highlight(
    frame_records: list[dict],
    evt: gr.SelectData,
):
    if not frame_records:
        raise gr.Error("当前没有高光结果")

    index = evt.index

    if isinstance(index, tuple):
        index = index[0]

    if not isinstance(index, int):
        raise gr.Error("无法识别所选图片")

    if not 0 <= index < len(frame_records):
        raise gr.Error("所选图片索引无效")

    frame = frame_records[index]
    image_path = frame["image_path"]

    details = (
        f"### 当前选择：第 {frame['rank']} 名\n\n"
        f"- 时间：`{frame['timestamp_sec']:.3f}s`\n"
        f"- CV 分数：`{frame['cv_score']:.3f}`\n"
        f"- VLM 分数：`{frame['vl_score']:.3f}`\n"
        f"- 最终分数：`{frame['final_score']:.3f}`"
    )

    return image_path, image_path, details


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Multimodal Video Highlight Pipeline"
    ) as app:
        gr.Markdown(
            """
# 视频高光筛选与局部重绘

第一步先分析视频。分析会调用 Qwen3-VL，可能产生 API 用量。
"""
        )

        frame_state = gr.State([])
        selected_path_state = gr.State(None)

        with gr.Row():
            video_input = gr.Video(
                label="输入视频",
                sources=["upload"],
            )

            with gr.Column():
                interval = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    value=1.0,
                    step=0.5,
                    label="抽帧间隔（秒）",
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="高光图片数量",
                )

                min_gap = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=3,
                    step=0.5,
                    label="高光之间最小时间间隔（秒）",
                )

                analyze_button = gr.Button(
                    "开始高光筛选",
                    variant="primary",
                )

        analysis_status = gr.Markdown()

        gallery = gr.Gallery(
            label="高光候选",
            columns=5,
            rows=1,
            allow_preview=True,
            interactive=False,
        )

        selected_details = gr.Markdown(
            "请先从上方高光候选中选择一张图片。"
        )

        image_editor = gr.ImageEditor(
            label="白色画笔区域将被重绘",
            type="pil",
            image_mode="RGBA",
            sources=(),
            transforms=(),
            fixed_canvas=False,
            brush=gr.Brush(
                colors=["#FFFFFF"],
                default_color="#FFFFFF",
                color_mode="fixed",
                default_size=30,
            ),
            layers=gr.LayerOptions(
                allow_additional_layers=False,
            ),
        )

        with gr.Row():
            prompt = gr.Textbox(
                label="重绘提示词",
                placeholder=(
                    "例如：a black leather jacket, "
                    "matching the original style"
                ),
                lines=3,
            )

            negative_prompt = gr.Textbox(
                label="负面提示词",
                value="blurry, low quality, distorted",
                lines=3,
            )

        with gr.Row():
            seed = gr.Number(
                label="Seed",
                value=42,
                precision=0,
            )

            strength = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.85,
                step=0.05,
                label="Strength",
            )

            steps = gr.Slider(
                minimum=10,
                maximum=60,
                value=30,
                step=1,
                label="推理步数",
            )

            guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                value=7.5,
                step=0.5,
                label="Guidance Scale",
            )

        inpaint_button = gr.Button(
            "开始局部重绘",
            variant="primary",
        )

        with gr.Row():
            inpaint_output = gr.Image(
                label="重绘结果",
                type="filepath",
            )

            metadata_output = gr.File(
                label="重绘参数 JSON",
            )

        inpaint_status = gr.Markdown()

        analyze_button.click(
            fn=analyze_video,
            inputs=[
                video_input,
                interval,
                top_k,
                min_gap,
            ],
            outputs=[
                gallery,
                frame_state,
                analysis_status,
            ],
            concurrency_limit=1,
            concurrency_id="video-analysis",
        )

        gallery.select(
            fn=select_highlight,
            inputs=[frame_state],
            outputs=[
                image_editor,
                selected_path_state,
                selected_details,
            ],
            queue=False,
        )

        inpaint_button.click(
            fn=run_inpaint,
            inputs=[
                image_editor,
                selected_path_state,
                prompt,
                negative_prompt,
                seed,
                strength,
                steps,
                guidance_scale,
            ],
            outputs=[
                inpaint_output,
                metadata_output,
                inpaint_status,
            ],
            concurrency_limit=1,
            concurrency_id="gpu-inpaint",
        )

    return app

if __name__ == "__main__":
    demo = build_app()

    demo.queue(
        default_concurrency_limit=1,
        max_size=10,
    )

    demo.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        show_error=True,
    )