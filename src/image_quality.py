from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import CVMetrics


def compute_cv_metrics(image_path: Path) -> CVMetrics:
    #遍历文件夹中的 JPG 图片并计算 CV 质量指标。
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(
            f"OpenCV failed to read image: {image_path}"
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return CVMetrics(
        sharpness=float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        saturation=float(hsv[:, :, 1].mean()),
        overexposure_ratio=float(
            np.mean(gray > 245)
        ),
        underexposure_ratio=float(
            np.mean(gray < 15)
        ),
        black_clip_ratio=float(
            np.mean(gray <= 2)
        ),
        white_clip_ratio=float(
            np.mean(gray >= 253)
        ),
    )