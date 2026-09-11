from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
from PIL import Image

from .models import FrameRecord


def compute_phash(frame: FrameRecord) -> imagehash.ImageHash:
    #计算一张图片的 perceptual hash

    image_path = frame.image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Frame image does not exist: {image_path}"
        )

    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img.convert("RGB"))

    except Exception as exc:
        raise ValueError(
            f"Failed to read image for hashing: {image_path}"
        ) from exc


def compute_sharpness(frame: FrameRecord) -> float:
    #使用 Laplacian variance 计算清晰度。

    image_path = frame.image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Frame image does not exist: {image_path}"
        )

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(
            f"OpenCV failed to read image: {image_path}"
        )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    return float(sharpness)


def group_similar_frames(
    frames: list[FrameRecord],
    hash_threshold: int,
) -> list[list[FrameRecord]]:
    
    #按 perceptual hash 将连续帧划分为相似组。

    #为避免链式分组，每张新图片与当前组的“参考帧”比较，而不是与上一张图片比较。

    #当前实现使用每组第一张图片作为 reference。

    if hash_threshold < 0:
        raise ValueError(
            f"hash_threshold must be >= 0, got {hash_threshold}"
        )

    if not frames:
        return []

    # 确保按照视频时间顺序处理
    sorted_frames = sorted(
        frames,
        key=lambda frame: frame.frame_index,
    )

    groups: list[list[FrameRecord]] = []

    current_group: list[FrameRecord] = []

    reference_hash = None

    for frame in sorted_frames:
        current_hash = compute_phash(frame)

        # 第一张图片
        if not current_group:
            current_group = [frame]
            reference_hash = current_hash
            continue

        distance = current_hash - reference_hash

        if distance <= hash_threshold:
            current_group.append(frame)

        else:
            # 当前组结束
            groups.append(current_group)

            # 新建下一组
            current_group = [frame]
            reference_hash = current_hash

    # 最后一组
    if current_group:
        groups.append(current_group)

    # 写入 group_id
    for group_id, group in enumerate(groups):
        for frame in group:
            frame.group_id = group_id

    return groups


def select_sharpest(
    groups: list[list[FrameRecord]],
) -> list[FrameRecord]:

    selected_frames: list[FrameRecord] = []

    for group in groups:
        if not group:
            continue

        best_frame = max(
            group,
            key=compute_sharpness,
        )

        selected_frames.append(best_frame)

    return selected_frames