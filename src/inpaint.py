from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image


@dataclass(frozen=True)
class InpaintConfig:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    max_side: int = 512


@dataclass(frozen=True)
class InpaintResult:
    image_path: Path
    mask_path: Path
    metadata_path: Path


class InpaintService:
    def __init__(self, config: InpaintConfig | None = None) -> None:
        self.config = config or InpaintConfig()
        self._pipe = None  # 第一次重绘时才加载模型

        if self.config.max_side < 64 or self.config.max_side % 8 != 0:
            raise ValueError("max_side must be a multiple of 8 and >= 64")

    def _get_pipeline(self):
        if self._pipe is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in this Python environment")

            self._pipe = AutoPipelineForInpainting.from_pretrained(
                self.config.model_id,
                dtype=torch.float16,
                use_safetensors=False,
                local_files_only=True,
            ).to("cuda")

        return self._pipe

    @staticmethod
    def _load_mask(mask_image: Image.Image | str | Path) -> Image.Image:
        # 这里要求传入已转换好的黑白蒙版：
        # 白色 = 重绘，黑色 = 保留。
        if isinstance(mask_image, (str, Path)):
            with Image.open(mask_image) as opened:
                return opened.convert("L")

        return mask_image.convert("L")

    def inpaint(
        self,
        image_path: str | Path,
        mask_image: Image.Image | str | Path,
        output_dir: str | Path,
        prompt: str,
        *,
        negative_prompt: str = "",
        seed: int = 42,
        strength: float = 0.85,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> InpaintResult:
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if seed < 0:
            raise ValueError("seed must be >= 0")
        if not 0 < strength <= 1:
            raise ValueError("strength must be in (0, 1]")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")
        if guidance_scale < 0:
            raise ValueError("guidance_scale must be >= 0")

        with Image.open(image_path) as opened:
            original = opened.convert("RGB")

        mask = self._load_mask(mask_image)

        # 允许分辨率不同，但长宽比例必须对应。
        original_ratio = original.width / original.height
        mask_ratio = mask.width / mask.height
        if abs(original_ratio - mask_ratio) > 0.01:
            raise ValueError("Mask and image aspect ratios do not match")

        if mask.size != original.size:
            mask = mask.resize(original.size, Image.Resampling.NEAREST)

        if mask.getbbox() is None:
            raise ValueError("Mask is empty; there is no area to inpaint")

        # 保持长宽比例，不把竖屏画面强行拉成正方形。
        scale = self.config.max_side / max(original.size)
        target_size = (
            max(64, round(original.width * scale / 8) * 8),
            max(64, round(original.height * scale / 8) * 8),
        )

        working_image = original.resize(
            target_size, Image.Resampling.LANCZOS
        )
        working_mask = mask.resize(
            target_size, Image.Resampling.NEAREST
        )

        pipe = self._get_pipeline()
        generator = torch.Generator(device="cuda").manual_seed(seed)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=working_image,
            mask_image=working_mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # 被安全检查拦截时，不把黑图当作成功结果保存。
        flags = output.nsfw_content_detected
        if flags is not None and any(flags):
            raise RuntimeError("Generated image was blocked by the safety checker")

        generated = output.images[0]

        # 在原始尺寸上合成：蒙版外保持原图像素。
        generated_full = generated.resize(
            original.size, Image.Resampling.LANCZOS
        )
        final_image = Image.composite(generated_full, original, mask)

        output_dir.mkdir(parents=True, exist_ok=True)
        edit_id = f"{image_path.stem}_{uuid4().hex[:8]}"
        result_path = output_dir / f"{edit_id}.png"
        mask_path = output_dir / f"{edit_id}_mask.png"
        metadata_path = output_dir / f"{edit_id}.json"

        final_image.save(result_path)
        mask.save(mask_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "source_image": str(image_path),
                    "result_image": str(result_path),
                    "mask_image": str(mask_path),
                    "model_id": self.config.model_id,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "strength": strength,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "model_size": target_size,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return InpaintResult(
            image_path=result_path,
            mask_path=mask_path,
            metadata_path=metadata_path,
        )