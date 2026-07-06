# -*- coding: utf-8 -*-
"""
Qwen VL module for VLM-R1 (Referring Expression / REC) with:
- format reward
- bbox IoU reward
- segmentation (mask IoU) reward using SAM3

Notes:
- SAM3 is lazily initialized on the first call of segmentation reward.
- All hard-coded SAM3 paths are kept the same as in your original GRPO script.
"""

from __future__ import annotations

import os
import re
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from trl.data_utils import maybe_apply_chat_template

from open_r1.vlm_modules.vlm_module import VLMBaseModule

# -----------------------------
# SAM3 imports (lazy-safe)
# -----------------------------
# Keep the path unchanged (same as your original script).
SAM3_REPO_PATH = os.environ.get("SAM3_REPO_PATH", os.path.join(os.environ.get("DGSEG_ROOT", "."), "sam3"))
if SAM3_REPO_PATH not in sys.path:
    sys.path.append(SAM3_REPO_PATH)

SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", os.path.join(os.environ.get("MODEL_ROOT", "./models"), "sam3.pt"))

_SAM3_PROCESSOR = None
_SAM3_READY = False
_SAM3_IMPORT_ERROR: Optional[Exception] = None


def _lazy_init_sam3() -> bool:
    """
    Lazy init SAM3 to avoid heavy import/init at module import time.

    Returns:
        True if SAM3 is ready; False otherwise (will not raise).
    """
    global _SAM3_PROCESSOR, _SAM3_READY, _SAM3_IMPORT_ERROR
    if _SAM3_READY:
        return True
    if _SAM3_IMPORT_ERROR is not None:
        return False

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        sam3_model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
        sam3_model.eval()
        if torch.cuda.is_available():
            sam3_model = sam3_model.cuda()
        _SAM3_PROCESSOR = Sam3Processor(sam3_model)
        _SAM3_READY = True
        return True
    except Exception as e:
        _SAM3_IMPORT_ERROR = e
        _SAM3_READY = False
        _SAM3_PROCESSOR = None
        return False

def _sanitize_xyxy_int(
    box: Sequence[float],
    width: int,
    height: int,
) -> Optional[List[int]]:
    if box is None or len(box) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
    except Exception:
        return None

    # order
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)

    # clamp
    x0 = max(0.0, min(x0, width - 1.0))
    x1 = max(0.0, min(x1, width - 1.0))
    y0 = max(0.0, min(y0, height - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))

    # minimal size
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None

    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def _xyxy_to_cxcywh_norm(box_xyxy: Sequence[float], height: int, width: int) -> torch.Tensor:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = x1 - x0
    h = y1 - y0
    return torch.tensor([cx / width, cy / height, w / width, h / height])


def _safe_get_image_path(kwargs: Dict[str, Any], i: int) -> Optional[str]:
    """
    VLM-R1 often passes image_path as list[list[str]] (multi-image friendly).
    We support a few common keys.
    """
    for k in ("image_path", "image", "img_path", "image_paths"):
        if k in kwargs and kwargs[k] is not None:
            v = kwargs[k]
            try:
                if isinstance(v, list) and len(v) > i:
                    vi = v[i]
                    if isinstance(vi, (list, tuple)) and len(vi) > 0:
                        return str(vi[0])
                    return str(vi)
            except Exception:
                continue
    return None


def _safe_get_text_prompt(kwargs: Dict[str, Any], i: int) -> str:
    """
    Use the referring expression / question as SAM3 text prompt.
    """
    for k in ("problem", "question", "prompt", "query", "text", "label"):
        if k in kwargs and kwargs[k] is not None:
            v = kwargs[k]
            try:
                if isinstance(v, list) and len(v) > i:
                    vi = v[i]
                    if isinstance(vi, (list, tuple)) and len(vi) > 0:
                        return str(vi[0])
                    return str(vi)
            except Exception:
                continue
    return ""


def _extract_bbox_from_completion(content: str) -> Optional[List[int]]:
    """
    Extract the first [x1, y1, x2, y2] bbox inside <answer>...</answer>.
    Returns integer bbox in the model-input coordinate space.
    """
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    bbox_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    try:
        m = re.search(answer_tag_pattern, content, re.DOTALL)
        if not m:
            return None
        answer_body = m.group(1).strip()

        # Try JSON first
        try:
            obj = json.loads(answer_body)
            if isinstance(obj, list) and isinstance(obj, dict):
                obj = obj[0]
            if isinstance(obj, list) and len(obj) >= 4:
                return [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
            if isinstance(obj, dict):
                for key in ("bbox", "box", "xyxy", "bbox_2d"):
                    if key in obj and isinstance(obj[key], list) and len(obj[key]) >= 4:
                        b = obj[key]
                        return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
        except Exception:
            pass

        # Regex fallback
        m2 = re.search(bbox_pattern, answer_body)
        if not m2:
            return None
        return [int(m2.group(1)), int(m2.group(2)), int(m2.group(3)), int(m2.group(4))]
    except Exception:
        return None

def _extract_label_from_completion(content: str) -> Optional[str]:
    """
    Extract the label inside <answer>...</answer>.
    Returns label as string.
    """
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    label_pattern = r"'label'\s*:\s*'([^']*)'"
    try:
        m = re.search(answer_tag_pattern, content, re.DOTALL)
        if not m:
            return None
        answer_body = m.group(1).strip()

        # Try JSON first
        try:
            obj = json.loads(answer_body)
            if isinstance(obj, list):
                obj = obj[0]
            if isinstance(obj, str):
                return str(obj[0])
            elif isinstance(obj, dict):
                for key in ("label", "label_text", "description"):
                    if key in obj and isinstance(obj[key], str):
                        return obj[key]
        except Exception:
            pass

        m2 = re.search(label_pattern, answer_body)
        if not m2:
            return None
        return m2.group(1).strip()
    except Exception:
        return None


def _resize_bbox_to_image_xyxy(
    bbox_xyxy: List[int],
    input_h: int,
    input_w: int,
    image_h: int,
    image_w: int,
) -> List[float]:
    """
    Convert bbox coordinates from model-input coordinate space to original image pixel space.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = x1 / max(1.0, float(input_w)) * float(image_w)
    y1 = y1 / max(1.0, float(input_h)) * float(image_h)
    x2 = x2 / max(1.0, float(input_w)) * float(image_w)
    y2 = y2 / max(1.0, float(input_h)) * float(image_h)
    return [x1, y1, x2, y2]


def _bbox_iou_xyxy(box1: Sequence[float], box2: Sequence[float]) -> float:
    """
    IoU in xyxy format; robust to float inputs.
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _safe_get_gt_segmentation(solution_item: Any, kwargs: Dict[str, Any], i: int) -> Any:
    """
    Retrieve GT segmentation from kwargs or from solution if possible.
    This is intentionally permissive to match different dataset packers.
    """
    for k in ("segmentation", "segmentations", "gt_segmentation", "gt_mask", "mask", "masks"):
        if k in kwargs and kwargs[k] is not None:
            v = kwargs[k]
            try:
                if isinstance(v, list) and len(v) > i:
                    return v[i]
            except Exception:
                pass

    # Fallback: try to parse from solution's <answer> json dict
    try:
        if isinstance(solution_item, str):
            answer_tag_pattern = r"<answer>(.*?)</answer>"
            m = re.search(answer_tag_pattern, solution_item, re.DOTALL)
            if m:
                obj = json.loads(m.group(1).strip())
                if isinstance(obj, dict):
                    for key in ("segmentation", "mask", "rle"):
                        if key in obj:
                            return obj[key]
                return obj
    except Exception:
        pass

    return None

def _load_coco_style_mask(seg: Any, image_w: int, image_h: int) -> Optional["torch.Tensor"]:
    """
    seg: COCO polygon style (list[list[float]]) or RLE dict.
    Returns boolean mask tensor on CPU with shape (H, W), dtype=bool.
    """
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        return None

    if seg is None:
        return None

    try:
        if isinstance(seg, dict) and "counts" in seg and "size" in seg:
            m = mask_utils.decode(seg)  # (H, W) or (H, W, 1)
        else:
            rle = mask_utils.frPyObjects(seg, image_h, image_w)
            m = mask_utils.decode(rle)
    except Exception:
        return None

    if m is None:
        return None

    if getattr(m, "ndim", 0) == 3:
        m = (m > 0).any(axis=2)

    m = (m > 0)
    return torch.as_tensor(m, dtype=torch.bool, device="cpu")


def _mask_iou(pred: "torch.Tensor", gt: "torch.Tensor") -> float:
    pred = pred.to(torch.bool)
    gt = gt.to(torch.bool)
    inter = torch.logical_and(pred, gt).sum().item()
    union = torch.logical_or(pred, gt).sum().item()
    if union == 0:
        return 0.0
    return float(inter / union)

def compute_mask_bbox_iou(
    mask: np.ndarray,
    bbox: List[int],
) -> float:
    """Compute the IoU between a mask and a bounding box."""
    mask = mask.astype(bool)
    x0, y0, x1, y1 = bbox

    H, W = mask.shape[:2]

    bbox_mask = np.zeros((H, W), dtype=bool)
    bbox_mask[y0:y1, x0:x1] = True

    intersection = np.logical_and(mask, bbox_mask)
    union = np.logical_or(mask, bbox_mask)

    return float(intersection.sum() / union.sum())

@torch.no_grad()
def _sam3_predict_masks(
    image: Image.Image,
    text_prompt: str,
    pos_bbox_xyxy: List[int],
) -> List["torch.Tensor"]:
    """
    Returns candidate masks as boolean torch tensors on CPU, each of shape (H, W).
    """
    if not _lazy_init_sam3():
        return []

    assert _SAM3_PROCESSOR is not None
    state = _SAM3_PROCESSOR.set_image(image)

    if text_prompt is not None:
        state = _SAM3_PROCESSOR.set_text_prompt(prompt=text_prompt, state=state)

    if text_prompt is None and pos_bbox_xyxy is None:
        return []

    masks: List[np.ndarray] = []
    if "masks" in state and hasattr(state["masks"], "numel") and state["masks"].numel() > 0:
        m = state["masks"]  # [K,H,W] or [K,1,H,W]
        if m.ndim == 4:
            m = m[:, 0]
        m = (m > 0).detach().cpu().numpy().astype(bool)  # (K,H,W)

        # 1) Score each candidate mask: pos_iou - neg_weight * mean(neg_ious).
        if pos_bbox_xyxy is not None:
            scores = []
            for k in range(m.shape[0]):
                sub_mask = m[k]
                s = compute_mask_bbox_iou(sub_mask, pos_bbox_xyxy)
                scores.append(float(s))

            # 2) Find the best score and merge masks with the best or near-best score.
            max_s = min(max(scores), 0.8)
            eps = 1e-6
            top_indices = [i for i, s in enumerate(scores) if s >= max_s - eps]

            merged = np.zeros_like(m[0], dtype=bool)
            for i in top_indices:
                merged |= m[i]

            masks.append(merged)
        else:
            masks = [m[0]]

    return masks


class Qwen2VLModule_seg(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            return Qwen2VLForConditionalGeneration
        if "Qwen2.5-VL" in model_id:
            return Qwen2_5_VLForConditionalGeneration
        raise ValueError(f"Unsupported model: {model_id}")

    def post_model_init(self, model, processing_class):
        return

    def get_processing_class(self):
        return AutoProcessor

    def get_vision_modules_keywords(self):
        return ["visual"]

    def get_custom_multimodal_keywords(self):
        return ["pixel_values", "image_grid_thw"]

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        return [("image_processor", "max_pixels"), ("image_processor", "min_pixels")]

    def prepare_prompt(self, processing_class, inputs: List[Dict[str, Any]]):
        return [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]

    def prepare_model_inputs(
        self,
        processing_class,
        prompts_text: List[str],
        images: List[Image.Image],
        return_tensors: str = "pt",
        padding: bool = True,
        padding_side: str = "left",
        add_special_tokens: bool = False,
    ):
        additional_output = None
        if images is not None and len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )
            additional_output = [
                {"image_grid_thw": image_grid_thw} for image_grid_thw in prompt_inputs.get("image_grid_thw", [])
            ]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens,
            )
        return prompt_inputs, additional_output

    @staticmethod
    def get_question_template(task_type: str):
        if task_type == "rec":
            return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        if task_type == "ic":
            return (
                "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. "
                "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
                "i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            )
        if task_type == "odLength":
            system_prompt = (
                "First thinks about the reasoning process in the mind and then provides the user with the answer. "
                "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
                "i.e., <think> reasoning process here </think><answer> answer here </answer>"
            )
            return system_prompt + "\n{Question}"
        return (
            "{Question} First output the thinking process in <think> </think> tags "
            "and then output the final answer in <answer> </answer> tags."
        )

    # -----------------------------
    # Reward functions
    # -----------------------------
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """
        <think>...</think><answer>{ ... [x1, y1, x2, y2] ... }</answer>
        """
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            if log_path:
                with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Format reward -------------\n")
                    for content, match in zip(completion_contents, matches):
                        f.write(f"Has format: {bool(match)}\n")
                        f.write(f"Content: {content}\n\n")

        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """
        BBox IoU reward (continuous IoU in [0,1]).
        - completion bbox: model-input coordinate space (pixel coords after Qwen image processor)
        - solution bbox: <answer>[x1,y1,x2,y2]</answer> in original image coords
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards: List[float] = []
        answer_tag_pattern = r"<answer>(.*?)</answer>"

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_path = _safe_get_image_path(kwargs, i)
            if image_path is None:
                rewards.append(0.0)
                continue

            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size

            input_h, input_w = image_h, image_w
            try:
                grid_thw = kwargs.get("image_grid_thw", None)
                if isinstance(grid_thw, list) and len(grid_thw) > i:
                    thw = grid_thw[i]
                    thw_list = thw.tolist() if hasattr(thw, "tolist") else list(thw)
                    input_h = int(thw_list[1] * 14)
                    input_w = int(thw_list[2] * 14)
            except Exception:
                pass

            # gt bbox
            try:
                m = re.findall(answer_tag_pattern, sol, re.DOTALL)
                if not m:
                    rewards.append(0.0)
                    continue
                gt_obj = json.loads(m[-1].strip())
                if not (isinstance(gt_obj, list) and len(gt_obj) >= 4):
                    rewards.append(0.0)
                    continue
                gt_bbox = [float(gt_obj[0]), float(gt_obj[1]), float(gt_obj[2])+ float(gt_obj[0]), float(gt_obj[3])+ float(gt_obj[1])]
            except Exception:
                rewards.append(0.0)
                continue

            pred_bbox_in = _extract_bbox_from_completion(content)
            if pred_bbox_in is None:
                rewards.append(0.0)
                continue

            pred_bbox_img = _resize_bbox_to_image_xyxy(pred_bbox_in, input_h, input_w, image_h, image_w)
            reward = _bbox_iou_xyxy(pred_bbox_img, gt_bbox)
            rewards.append(float(reward))

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH", "")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                if log_path:
                    problem = _safe_get_text_prompt(kwargs, i)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} BBox IoU reward: {reward:.4f} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"pred_bbox_in: {pred_bbox_in}\n")
                        f.write(f"pred_bbox_img: {pred_bbox_img}\n")
                        f.write(f"gt_bbox: {gt_bbox}\n")
                        f.write(f"content: {content}\n\n")

        return rewards

    @staticmethod
    def segmentation_reward(completions, solution, **kwargs):
        """
        Segmentation reward using SAM3 (best-of-masks IoU):
        1) Extract predicted bbox from completion
        2) Convert bbox to original image coords
        3) Run SAM3 with (image, text_prompt, predicted bbox) to obtain candidate masks
        4) Reward by best mask IoU with GT segmentation mask

        Return:
            best mask IoU in [0,1].
            If you want binary reward like your original script, set env:
              SEG_REWARD_THRESHOLD=0.5
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards: List[float] = []

        seg_threshold = None
        try:
            if os.getenv("SEG_REWARD_THRESHOLD", ""):
                seg_threshold = float(os.getenv("SEG_REWARD_THRESHOLD"))
        except Exception:
            seg_threshold = None

        for i, (content, sol_item) in enumerate(zip(contents, solution)):
            image_path = _safe_get_image_path(kwargs, i)
            if image_path is None:
                rewards.append(0.0)
                continue

            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size

            input_h, input_w = image_h, image_w
            try:
                grid_thw = kwargs.get("image_grid_thw", None)
                if isinstance(grid_thw, list) and len(grid_thw) > i:
                    thw = grid_thw[i]
                    thw_list = thw.tolist() if hasattr(thw, "tolist") else list(thw)
                    input_h = int(thw_list[1] * 14)
                    input_w = int(thw_list[2] * 14)
            except Exception:
                pass

            pred_bbox_in = _extract_bbox_from_completion(content)
            if pred_bbox_in is not None:
                pred_bbox_img_f = _resize_bbox_to_image_xyxy(pred_bbox_in, input_h, input_w, image_h, image_w)
                pred_bbox_img = _sanitize_xyxy_int(pred_bbox_img_f, width=image_w, height=image_h)
            else:
                pred_bbox_img = None

            gt_seg = _safe_get_gt_segmentation(sol_item, kwargs, i)
            gt_seg = json.loads(gt_seg.strip())
            gt_mask = _load_coco_style_mask(gt_seg, image_w=image_w, image_h=image_h)

            if gt_mask is None:
                rewards.append(0.0)
                continue

            label = _extract_label_from_completion(content)
            cand_masks = _sam3_predict_masks(image=image, text_prompt=label, pos_bbox_xyxy=pred_bbox_img)
            if not cand_masks or len(cand_masks) == 0:
                rewards.append(0.0)
                continue

            mask_iou = _mask_iou(torch.tensor(cand_masks[0]), gt_mask)
            reward = float(mask_iou)
            if seg_threshold is not None:
                reward = 1.0 if mask_iou >= seg_threshold else 0.0

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH", "")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                if log_path:
                    with open(log_path.replace(".txt", "_seg.txt"), "a", encoding="utf-8") as f:
                        f.write(f"------------- {current_time} Seg reward: {reward:.4f}  -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {label}\n")
                        f.write(f"pred_bbox_in: {pred_bbox_in}\n")
                        f.write(f"pred_bbox_img: {pred_bbox_img}\n")
                        f.write(f"content: {content}\n\n")

        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        """
        Map reward function names used by VLM-R1 configs to concrete callables.
        """
        if task_type != "rec":
            raise ValueError(f"Unsupported task type: {task_type}")

        if func == "accuracy":
            return Qwen2VLModule_seg.iou_reward
        if func == "format":
            return Qwen2VLModule_seg.format_reward_rec
        if func in ("segmentation", "seg", "mask_iou"):
            return Qwen2VLModule_seg.segmentation_reward

        raise ValueError(f"Unsupported reward function: {func}")
