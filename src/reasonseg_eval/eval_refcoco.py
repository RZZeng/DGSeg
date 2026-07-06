# eval_refcoco.py
# -*- coding: utf-8 -*-
"""
RefCOCO(/+/g) evaluation with:
- DDP MLLM inference mode identical to eval_reasonseg.py (cache jsonl, gather rank_items -> main rank eval)
- RefCOCO dataset building / GT mask decoding logic referenced from the RefCOCO evaluation utilities
- DGSeg fusion inference = PixelGate dual-branch SAM3 (text branch + bbox branch) from eval_reasonseg.py

Usage (example):
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 eval_refcoco.py
"""

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import os
import re
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ============================================================
# DDP
# ============================================================
def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return local_rank, world_size, rank
    return local_rank, 1, 0


local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
main_rank = 0
print(f"Process {rank}/{world_size} using {device}")


# ============================================================
# Configs (edit these)
# ============================================================
# ---- dataset ----
DATASET = os.environ.get("DATASET", "refcoco+")  # refcoco / refcoco+ / refcocog
SPLIT = os.environ.get("SPLIT", "testA")          # val / testA / testB / test
TEST_STEP = int(os.environ.get("TEST_STEP", "2250"))
RESIZE = int(os.environ.get("RESIZE", "1024"))    # used as max_pixels=RESIZE*RESIZE for Qwen image input

RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"Qwen2.5-VL-7B-Instruct-rec-lora-{DATASET}-{SPLIT}-dgseg_fusion",
)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

# RefCOCO json is the "dataset.json" you used in the logits fusion script
REFCOCO_JSON_DIR = os.environ.get("REFCOCO_JSON_DIR", os.path.join(os.environ.get("DATA_ROOT", "./data"), f"{DATASET}_{SPLIT}_dataset.json"))
# COCO image folder
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", os.path.join(os.environ.get("DATA_ROOT", "./data"), "refer_seg/images/mscoco/images/train2014"))

OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    f"../refcoco_output/overall_json/{DATASET}_{SPLIT}_results_{RUN_NAME}.json",
)
VIS_DIR = os.environ.get("VIS_DIR", f"../refcoco_output/vis/{DATASET}_{SPLIT}_vis_{RUN_NAME}")

# ---- inference ----
BSZ = 8
MAX_SAMPLES = None          # set int to subsample (on sample-level before flatten)
MAX_IMAGE_SIDE = None       # optional: if set, will resize image and ALSO resize GT mask accordingly
MAX_NEW_TOKENS = 256

# Debug: if TARGET_JSON_FILE exists, skip MLLM inference and directly evaluate on main rank
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").lower() in {"1", "true", "yes"}
CACHE_ONLY = os.environ.get("CACHE_ONLY", "false").lower() in {"1", "true", "yes"}
TARGET_JSON_FILE = os.environ.get(
    "TARGET_JSON_FILE",
    f"../refcoco_output/overall_json/{DATASET}_{SPLIT}_9000samples_dataset.jsonl",
)


# ---- visualization ----
VIS_MAX_SAVE = 400
VIS_ALPHA = 0.45

# ============================================================
# SAM3 + PixelGate fusion
# ============================================================
SAM3_REPO_PATH = os.environ.get("SAM3_REPO_PATH", os.path.join(os.environ.get("DGSEG_ROOT", "."), "sam3"))
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", os.path.join(os.environ.get("MODEL_ROOT", "./models"), "sam3.pt"))

if SAM3_REPO_PATH not in os.sys.path:
    os.sys.path.append(SAM3_REPO_PATH)

_SAM3_MODEL = None
_SAM3_PROCESSOR = None
_SAM3_READY = False
_SAM3_IMPORT_ERROR = None

# PixelGate fusion net
USE_PIXEL_GATE_FUSION = True
FUSION_CKPT = os.environ.get("FUSION_CKPT", os.path.join(os.environ.get("DGSEG_ROOT", "."), "checkpoints", "sam3_fusion.pt"))
_FUSION_NET = None
_FUSION_READY = False
_FUSION_ERROR = None

FUSION_IN_CHANNELS = 256
FUSION_HIDDEN = 64
FUSION_GATE_T = 1.0
FUSION_GATE_CLAMP = (0.01, 0.99)
ORACLE_SOFTMAX_SCALE = 100.0 


def lazy_init_sam3() -> bool:
    global _SAM3_MODEL, _SAM3_PROCESSOR, _SAM3_READY, _SAM3_IMPORT_ERROR
    if _SAM3_READY:
        return True
    if _SAM3_IMPORT_ERROR is not None:
        return False
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        sam3_model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT, enable_inst_interactivity=True)
        sam3_model.eval()
        if torch.cuda.is_available():
            sam3_model = sam3_model.cuda()
        _SAM3_MODEL = sam3_model
        _SAM3_PROCESSOR = Sam3Processor(sam3_model)
        _SAM3_READY = True
        return True
    except Exception as e:
        _SAM3_IMPORT_ERROR = e
        _SAM3_READY = False
        _SAM3_PROCESSOR = None
        return False


class PixelGateMaskLogitsFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        gn_groups: int = 8,
        gate_temperature: float = 1.0,
        clamp_gate: Optional[Tuple[float, float]] = (0.01, 0.99),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.gate_temperature = gate_temperature
        self.clamp_gate = clamp_gate

        cin = 2 * in_channels
        self.conv1 = nn.Conv2d(cin, hidden_channels, 3, padding=1)
        g = min(gn_groups, hidden_channels)
        while hidden_channels % g != 0 and g > 1:
            g -= 1
        self.norm1 = nn.GroupNorm(g, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, 1, 3, padding=1)

    def forward(self, pixel_a, pixel_b, logits_a, logits_b, return_gate=False):
        if logits_a.dim() == 3:
            logits_a = logits_a.unsqueeze(1)
        if logits_b.dim() == 3:
            logits_b = logits_b.unsqueeze(1)

        B, C, H, W = pixel_a.shape
        logH, logW = logits_a.shape[-2:]

        x = torch.cat([pixel_a, pixel_b], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        gate_logits = self.conv2(x)  # (B,1,H,W)

        gate = torch.sigmoid(gate_logits / max(self.gate_temperature, 1e-6))
        if self.clamp_gate is not None:
            lo, hi = self.clamp_gate
            gate = gate.clamp(lo, hi)

        if (logH, logW) != (H, W):
            gate = F.interpolate(gate, size=(logH, logW), mode="bilinear", align_corners=False)

        fused = gate * logits_a + (1.0 - gate) * logits_b
        if return_gate:
            return fused, gate
        return fused


def lazy_init_fusion_net() -> bool:
    global _FUSION_NET, _FUSION_READY, _FUSION_ERROR
    if _FUSION_READY:
        return True
    if _FUSION_ERROR is not None:
        return False

    if not USE_PIXEL_GATE_FUSION:
        _FUSION_READY = True
        return True

    try:
        net = PixelGateMaskLogitsFusion(
            in_channels=FUSION_IN_CHANNELS,
            hidden_channels=FUSION_HIDDEN,
            gate_temperature=FUSION_GATE_T,
            clamp_gate=FUSION_GATE_CLAMP,
        ).to(device).eval()

        ckpt = torch.load(FUSION_CKPT, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "fusion_net" in ckpt:
            net.load_state_dict(ckpt["fusion_net"], strict=True)
        else:
            net.load_state_dict(ckpt, strict=False)

        _FUSION_NET = net
        _FUSION_READY = True
        return True
    except Exception as e:
        _FUSION_ERROR = e
        _FUSION_READY = False
        _FUSION_NET = None
        return False


# ============================================================
# Utilities: parsing / bbox resize / masks
# ============================================================
ANSWER_JSON_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)
BBOX_RE = re.compile(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]")

def extract_answer_json(output_text: str) -> Dict[str, Any]:
    if output_text is None:
        return {}
    m = ANSWER_JSON_RE.search(output_text)
    if not m:
        # fallback: try to find a json-ish substring
        s = output_text.strip()
        try:
            return json.loads(s)
        except Exception:
            return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}

# def extract_bbox_answer(output_text: str) -> Optional[List[float]]:
#     ans = extract_answer_json(output_text)
#     if isinstance(ans, List):
#         ans = ans[0]
#     # common keys
#     for k in ["bbox", "box", "bboxes", "pred_bbox", "pred_box", "bbox_2d"]:
#         if k in ans:
#             v = ans[k]
#             if isinstance(v, (list, tuple)) and len(v) >= 4:
#                 return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
#     # regex fallback
#     m = BBOX_RE.search(output_text or "")
#     if m:
#         return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]
#     return None

# def extract_label_answer(output_text: str) -> Optional[str]:
#     ans = extract_answer_json(output_text)
#     if isinstance(ans, List):
#         ans = ans[0]
#     for k in ["label", "category", "name", "pred_label", "object"]:
#         if k in ans:
#             v = ans[k]
#             if isinstance(v, str):
#                 return v
#     # soft fallback: none
#     return None

def extract_bbox_answer(content: str) -> List[int]:
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    bbox_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    m = re.search(answer_tag_pattern, content, re.DOTALL)
    if not m:
        return [0, 0, 0, 0]
    ans = m.group(1).strip()
    m2 = re.search(bbox_pattern, ans, re.DOTALL)
    if not m2:
        return [0, 0, 0, 0]
    return [int(m2.group(1)), int(m2.group(2)), int(m2.group(3)), int(m2.group(4))]

def extract_label_answer(content: str) -> Optional[str]:
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
            if isinstance(obj, list) and len(obj) > 0:
                obj = obj[0]
            if isinstance(obj, dict):
                for key in ("label", "label_text", "description"):
                    if key in obj and isinstance(obj[key], str):
                        return obj[key].strip()
        except Exception:
            pass

        m2 = re.search(label_pattern, answer_body)
        if not m2:
            return None
        return m2.group(1).strip()
    except Exception:
        return None

def resize_bbox_xyxy(bbox_xyxy: Optional[List[float]], in_h: int, in_w: int, out_h: int, out_w: int) -> Optional[List[float]]:
    if bbox_xyxy is None or len(bbox_xyxy) < 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy[:4]]
    # assume bbox is in [0,in_w/in_h]
    sx = float(out_w) / max(1.0, float(in_w))
    sy = float(out_h) / max(1.0, float(in_h))
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]

def sanitize_xyxy_int(box: List[float], width: int, height: int) -> Optional[List[int]]:
    if box is None or len(box) < 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
    x0 = max(0.0, min(x0, width - 1.0))
    x1 = max(0.0, min(x1, width - 1.0))
    y0 = max(0.0, min(y0, height - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]

def xyxy_to_xywh_norm(bbox_xyxy: List[int], W: int, H: int) -> List[float]:
    x0, y0, x1, y1 = bbox_xyxy
    x0, x1 = float(x0), float(x1)
    y0, y1 = float(y0), float(y1)
    cx = (x0 + x1) * 0.5 / max(1.0, float(W))
    cy = (y0 + y1) * 0.5 / max(1.0, float(H))
    bw = (x1 - x0) / max(1.0, float(W))
    bh = (y1 - y0) / max(1.0, float(H))
    return [cx, cy, bw, bh]

def ensure_same_hw(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask is None:
        return mask
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    m = mask.astype(np.uint8) * 255
    m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (m > 127)

def softmax2(a: float, b: float, scale: float = 100.0) -> Tuple[float, float]:
    """
    softmax([a,b] * scale) -> (wa, wb)
    Larger scale produces sharper weights, closer to argmax.
    """
    x0 = a * scale
    x1 = b * scale
    m = max(x0, x1)
    e0 = np.exp(x0 - m)
    e1 = np.exp(x1 - m)
    s = e0 + e1
    return float(e0 / s), float(e1 / s)

def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    if union <= 0:
        return 0.0, inter, union
    return float(inter / union), inter, union

def sigmoid_to_logits(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))

def is_empty_masks(masks_t: torch.Tensor) -> bool:
    if masks_t is None:
        return True
    if hasattr(masks_t, "numel") and masks_t.numel() == 0:
        return True
    return False

def mask_in_bbox_ratio(mask: np.ndarray, bbox_xyxy: List[float], eps: float = 1e-6) -> float:
    assert mask.ndim == 2
    H, W = mask.shape
    m = mask.astype(bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    x1, y1, x2, y2 = bbox_xyxy
    x1i = int(np.floor(x1)); y1i = int(np.floor(y1))
    x2i = int(np.ceil(x2));  y2i = int(np.ceil(y2))
    x1i = max(0, min(W, x1i)); x2i = max(0, min(W, x2i))
    y1i = max(0, min(H, y1i)); y2i = max(0, min(H, y2i))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    in_cnt = int(m[y1i:y2i, x1i:x2i].sum())
    return float(in_cnt) / float(total + eps)

def select_best_mask_by_bbox_iou_scaled(
    out: Dict[str, Any],
    bbox_xyxy_image: List[float],
    img_hw: Tuple[int, int],
) -> Dict[str, Any]:
    """
    From SAM3 *_alloutput dict:
      - pick all masks with containment ratio close to the max (capped by 0.8 like your code)
      - merge masks by OR
      - merge logits by: if any >0 at pixel -> max; else mean
    Return:
      {
        "best_mask": (1,1,H,W) torch.bool
        "best_logits": (1,H,W) torch.float32 logits
        "pixel_embed": (1,C,H,W) torch.float32 or None
      }
    """
    masks = out["masks"].to(torch.bool)        # (N,1,Hm,Wm)
    logits = out["masks_logits"]               # (N,1,Hm,Wm) (sigmoid-space in SAM3)
    N, _, Hm, Wm = masks.shape
    device = logits.device

    Himg, Wimg = img_hw
    x1, y1, x2, y2 = bbox_xyxy_image
    # scale to mask grid
    sx1 = x1 / max(1.0, Wimg) * float(Wm)
    sx2 = x2 / max(1.0, Wimg) * float(Wm)
    sy1 = y1 / max(1.0, Himg) * float(Hm)
    sy2 = y2 / max(1.0, Himg) * float(Hm)
    bbox_scaled = [sx1, sy1, sx2, sy2]

    bbox_m = bbox_mask_from_xyxy(bbox_scaled, H=Hm, W=Wm, device=device)  # (1,1,Hm,Wm)
    inter = (masks & bbox_m).flatten(2).sum(dim=-1).squeeze(1)  # (N,)
    union = (masks | bbox_m).flatten(2).sum(dim=-1).squeeze(1)  # (N,)
    ious = inter.float() / (union.float() + 1e-6)               # (N,)
    best_idx = torch.argmax(ious).item()
    return {
        "best_mask": masks[best_idx:best_idx + 1],      # (1,1,Hm,Wm)
        "best_logits": logits[best_idx:best_idx + 1],   # (1,1,Hm,Wm)
        "best_idx": best_idx,
        "best_bbox_iou": ious[best_idx],
    }

def select_masks_by_bbox_containment_scaled(
    out: Dict[str, torch.Tensor],
    bbox_xyxy_image: List[float],
    img_hw: Tuple[int, int],
    ratio_cap: float = 0.9,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    masks = out["masks"].to(torch.bool)          # (N,1,Hm,Wm)
    logits = out["masks_logits"]                 # (N,1,Hm,Wm)
    N, _, Hm, Wm = masks.shape
    device_t = masks.device

    Himg, Wimg = img_hw
    x1, y1, x2, y2 = bbox_xyxy_image
    sx1 = x1 / max(1.0, float(Wimg)) * float(Wm)
    sx2 = x2 / max(1.0, float(Wimg)) * float(Wm)
    sy1 = y1 / max(1.0, float(Himg)) * float(Hm)
    sy2 = y2 / max(1.0, float(Himg)) * float(Hm)
    bbox_scaled = [sx1, sy1, sx2, sy2]

    bbox_m = bbox_mask_from_xyxy(bbox_scaled, H=Hm, W=Wm, device=device_t)  # (1,1,Hm,Wm)

    inter = (masks & bbox_m).flatten(2).sum(dim=-1).squeeze(1).float()
    area = masks.flatten(2).sum(dim=-1).squeeze(1).float()
    ratios = inter / (area + 1e-6)

    best_ratio = ratios.max()
    cap_t = torch.tensor(ratio_cap, device=device_t, dtype=ratios.dtype)
    threshold = torch.minimum(best_ratio, cap_t)
    top_indices = torch.where(ratios >= (threshold - eps))[0]

    if top_indices.numel() == 0:
        best_idx = torch.argmax(ratios).item()
        top_indices = torch.tensor([best_idx], device=device_t, dtype=torch.long)

    logits_top = logits[top_indices].float()
    total_logits = torch.where(
        (logits_top > 0).any(dim=0),
        logits_top.max(dim=0).values,
        logits_top.mean(dim=0),
    )
    if total_logits.dim() == 3:
        total_logits = total_logits.unsqueeze(0)

    total_mask = masks[top_indices].any(dim=0)

    return {
        "best_mask": total_mask,          # (1,1,Hm,Wm) bool
        "best_logits": total_logits,      # (1,1,Hm,Wm) float (sigmoid-space-like)
        "best_idx": top_indices,          # (T,)
        "ratios": ratios,
        "threshold": threshold,
        "best_bbox_iou": best_ratio,
    }


@torch.no_grad()
def run_branch_text(img: Image.Image, processor, label: str) -> Dict[str, torch.Tensor]:
    state = processor.set_image(img)
    if hasattr(processor, "set_text_prompt_alloutput"):
        out = processor.set_text_prompt_alloutput(prompt=label, state=state)
        return out
    out = processor.set_text_prompt(prompt=label, state=state)
    return out if isinstance(out, dict) else state


@torch.no_grad()
def run_branch_bbox(img: Image.Image, processor, bbox_xywh_norm: List[float]) -> Dict[str, torch.Tensor]:
    state = processor.set_image(img)
    if hasattr(processor, "reset_all_prompts"):
        processor.reset_all_prompts(state)
    if hasattr(processor, "add_geometric_prompt_alloutput"):
        out = processor.add_geometric_prompt_alloutput(box=bbox_xywh_norm, label=True, state=state)
        return out
    if hasattr(processor, "add_geometric_prompt"):
        out = processor.add_geometric_prompt(box=bbox_xywh_norm, label=True, state=state)
        return out if isinstance(out, dict) else state
    return state

def bbox_mask_from_xyxy(bbox_xyxy: List[float], H: int, W: int, device=None) -> torch.Tensor:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))
    m = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)
    if x2 > x1 and y2 > y1:
        m[:, :, y1:y2, x1:x2] = True
    return m

# def select_best_mask_by_bbox_iou_scaled(
#     out: Dict[str, torch.Tensor],
#     bbox_xyxy_image: List[float],
#     img_hw: Tuple[int, int],
# ) -> Dict[str, torch.Tensor]:
#     """
#     bbox_xyxy_image is in original image coordinate. If SAM3 mask resolution differs,
#     scale bbox into (Hmask,Wmask) coordinate before IoU.
#     """
#     masks = out["masks"].to(torch.bool)        # (N,1,Hm,Wm)
#     logits = out["masks_logits"]               # (N,1,Hm,Wm) (sigmoid-space in SAM3)
#     N, _, Hm, Wm = masks.shape
#     device = logits.device

#     Himg, Wimg = img_hw
#     x1, y1, x2, y2 = bbox_xyxy_image
#     # scale to mask grid
#     sx1 = x1 / max(1.0, Wimg) * float(Wm)
#     sx2 = x2 / max(1.0, Wimg) * float(Wm)
#     sy1 = y1 / max(1.0, Himg) * float(Hm)
#     sy2 = y2 / max(1.0, Himg) * float(Hm)
#     bbox_scaled = [sx1, sy1, sx2, sy2]

#     bbox_m = bbox_mask_from_xyxy(bbox_scaled, H=Hm, W=Wm, device=device)  # (1,1,Hm,Wm)
#     inter = (masks & bbox_m).flatten(2).sum(dim=-1).squeeze(1)  # (N,)
#     union = (masks | bbox_m).flatten(2).sum(dim=-1).squeeze(1)  # (N,)
#     ious = inter.float() / (union.float() + 1e-6)               # (N,)
#     best_idx = torch.argmax(ious).item()
#     return {
#         "best_mask": masks[best_idx:best_idx + 1],      # (1,1,Hm,Wm)
#         "best_logits": logits[best_idx:best_idx + 1],   # (1,1,Hm,Wm)
#         "best_idx": best_idx,
#         "best_bbox_iou": ious[best_idx],
#     }

def make_empty_mask(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W), dtype=bool)

def draw_title(img: Image.Image, title: str) -> Image.Image:
    # reuse draw_bbox_and_text for simple title box
    return draw_bbox_and_text(img, bbox_xyxy=None, lines=[title])

def gate_mean_on_gt_foreground(gate_hw: np.ndarray, gt_mask_hw: np.ndarray) -> Optional[float]:
    """
    gate_hw: (Hg, Wg) float in [0,1]
    gt_mask_hw: (H, W) bool (original image size)
    Return mean gate over positions where gt is foreground (True).
    We downsample GT to gate resolution using nearest.
    """
    if gate_hw is None or gt_mask_hw is None:
        return None
    Hg, Wg = gate_hw.shape[:2]
    if Hg <= 0 or Wg <= 0:
        return None

    gt_u8 = gt_mask_hw.astype(np.uint8) * 255
    gt_small = cv2.resize(gt_u8, (Wg, Hg), interpolation=cv2.INTER_NEAREST) > 0
    if gt_small.sum() == 0:
        return None
    return float(gate_hw[gt_small].mean())


@torch.no_grad()
def sam3_predict_mask_pixel_gate_fusion_with_branches(
    img: Image.Image,
    label: str,
    bbox_xyxy_image: List[int],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Returns:
      pred_mask_bool_np (H,W) aligned to current img resolution
      info dict: branch_used, gate_mean
    """
    info: Dict[str, Any] = {"branch_used": None, "gate_mean": None}

    ok_sam = lazy_init_sam3()
    if not ok_sam:
        info["branch_used"] = "sam3_init_failed"
        return None, info
    assert _SAM3_PROCESSOR is not None

    Himg, Wimg = img.size[1], img.size[0]
    bbox_xywh_norm = xyxy_to_xywh_norm(bbox_xyxy_image, W=Wimg, H=Himg)
    bbox_xyxy_image_f = [float(v) for v in bbox_xyxy_image]

    out_text = run_branch_text(img, _SAM3_PROCESSOR, label)
    out_bbox = run_branch_bbox(img, _SAM3_PROCESSOR, bbox_xywh_norm)

    has_text = isinstance(out_text, dict) and ("masks" in out_text) and (not is_empty_masks(out_text["masks"]))
    has_bbox = isinstance(out_bbox, dict) and ("masks" in out_bbox) and (not is_empty_masks(out_bbox["masks"]))

    if not has_text and not has_bbox:
        info["branch_used"] = "none"
        return None, info

    text_mask_np = None
    bbox_mask_np = None
    sel_a = None
    sel_b = None
    if has_text:
        if DATASET in ["refcoco", "refcoco+"]:
            sel_a = select_best_mask_by_bbox_iou_scaled(out_text, bbox_xyxy_image_f, img_hw=(Himg, Wimg))
        else:
            sel_a = select_masks_by_bbox_containment_scaled(out_text, bbox_xyxy_image_f, img_hw=(Himg, Wimg), ratio_cap = 0.95)
        text_mask_np = sel_a["best_mask"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
        text_mask_np = ensure_same_hw(text_mask_np, target_h=Himg, target_w=Wimg)

    if has_bbox:
        if DATASET in ["refcoco", "refcoco+"]:
            sel_b = select_best_mask_by_bbox_iou_scaled(out_bbox, bbox_xyxy_image_f, img_hw=(Himg, Wimg))
        else:
            sel_b = select_masks_by_bbox_containment_scaled(out_bbox, bbox_xyxy_image_f, img_hw=(Himg, Wimg), ratio_cap = 0.95)
        bbox_mask_np = sel_b["best_mask"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
        bbox_mask_np = ensure_same_hw(bbox_mask_np, target_h=Himg, target_w=Wimg)


    # ========================================================
    # [NEW] Avg-Logits (0.5/0.5) fusion mask (when both branches exist)
    #   - Uses exactly the same logits conversion + scaling as PixelGate path
    #   - Independent of PixelGate availability
    # ========================================================
    avg05_mask_np = None
    if has_text and has_bbox and (sel_a is not None) and (sel_b is not None):
        logits_a = sel_a["best_logits"].to(device)  # sigmoid-space
        logits_b = sel_b["best_logits"].to(device)  # sigmoid-space
        logits_a = sigmoid_to_logits(logits_a)
        logits_b = sigmoid_to_logits(logits_b)

        # scale logits_b into logits_a range (keep sign)  (same as your fusion code)
        Amax, Amin = logits_a.max(), logits_a.min()
        if (logits_b > 0).any():
            Bp = torch.where(
                logits_b > 0,
                logits_b / logits_b[logits_b > 0].max() * Amax,
                torch.zeros_like(logits_b),
            )
        else:
            Bp = torch.zeros_like(logits_b)

        if (logits_b < 0).any():
            Bn = torch.where(
                logits_b < 0,
                logits_b / (-logits_b[logits_b < 0].min()) * (-Amin),
                torch.zeros_like(logits_b),
            )
        else:
            Bn = torch.zeros_like(logits_b)

        logits_b = Bp + Bn

        avg_logits = 0.5 * logits_a + 0.5 * logits_b
        avg_prob = torch.sigmoid(avg_logits)
        avg05_mask_np = (avg_prob > 0.5).squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
        avg05_mask_np = ensure_same_hw(avg05_mask_np, target_h=Himg, target_w=Wimg)
    else:
        # if not both branches, avg05 just degenerates to the available branch (for completeness)
        if has_bbox:
            avg05_mask_np = bbox_mask_np
        else:
            avg05_mask_np = text_mask_np

    # ========================================================
    # PixelGate fused result (UNCHANGED)
    # ========================================================
    # If no fusion, fallback same as before
    if (not USE_PIXEL_GATE_FUSION) or (not lazy_init_fusion_net()) or (_FUSION_NET is None):
        if has_bbox:
            info["branch_used"] = "bbox_only_no_fusion"
            return text_mask_np, bbox_mask_np, bbox_mask_np, avg05_mask_np, info
        info["branch_used"] = "text_only_no_fusion"
        return text_mask_np, bbox_mask_np, text_mask_np, avg05_mask_np, info

    # One branch only => same as before
    if not has_text and has_bbox:
        info["branch_used"] = "bbox_only"
        return None, bbox_mask_np, bbox_mask_np, avg05_mask_np, info
    if has_text and not has_bbox:
        info["branch_used"] = "text_only"
        return text_mask_np, None, text_mask_np, avg05_mask_np, info

    # Both present => pixel-gate fusion
    pixel_a = out_text.get("pixel_embed", None)
    pixel_b = out_bbox.get("pixel_embed", None)
    if pixel_a is None or pixel_b is None:
        info["branch_used"] = "bbox_only_no_pixel_embed"
        return text_mask_np, bbox_mask_np, bbox_mask_np, avg05_mask_np, info

    assert sel_a is not None and sel_b is not None

    logits_a = sel_a["best_logits"].to(device)  # sigmoid-space
    logits_b = sel_b["best_logits"].to(device)  # sigmoid-space
    logits_a = sigmoid_to_logits(logits_a)
    logits_b = sigmoid_to_logits(logits_b)

    # scale logits_b into logits_a range (keep sign) (unchanged)
    Amax, Amin = logits_a.max(), logits_a.min()
    if (logits_b > 0).any():
        Bp = torch.where(logits_b > 0, logits_b / logits_b[logits_b > 0].max() * Amax, torch.zeros_like(logits_b))
    else:
        Bp = torch.zeros_like(logits_b)
    if (logits_b < 0).any():
        Bn = torch.where(logits_b < 0, logits_b / (-logits_b[logits_b < 0].min()) * (-Amin), torch.zeros_like(logits_b))
    else:
        Bn = torch.zeros_like(logits_b)
    logits_b = Bp + Bn

    # [NEW] cache processed branch logits for oracle fusion (do NOT dump to json)
    # shape: (1,1,Hm,Wm) logits-space
    try:
        info["_logits_a"] = logits_a.detach().float().cpu()  # torch.Tensor CPU
        info["_logits_b"] = logits_b.detach().float().cpu()
    except Exception:
        info["_logits_a"] = None
        info["_logits_b"] = None

    pixel_a = pixel_a.to(device)
    pixel_b = pixel_b.to(device)

    fused_logits, gate = _FUSION_NET(pixel_a, pixel_b, logits_a, logits_b, return_gate=True)
    fused_prob = torch.sigmoid(fused_logits)
    fused_mask = (fused_prob > 0.5).squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
    fused_mask = ensure_same_hw(fused_mask, target_h=Himg, target_w=Wimg)

    info["branch_used"] = "pixel_gate_fused"
    info["gate_mean"] = float(gate.mean().item()) if gate is not None else None

    if gate is not None:
        gate_map = gate.squeeze(0).squeeze(0).detach().float().cpu().numpy()  # (Hg,Wg)
        info["gate_map"] = gate_map
        info["gate_hw"] = [int(gate_map.shape[0]), int(gate_map.shape[1])]

    return text_mask_np, bbox_mask_np, fused_mask, avg05_mask_np, info


# ============================================================
# RefCOCO dataset loading + GT mask decoding
# ============================================================
def load_refcoco_json(json_dir: str, image_folder: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    json: list of entries with keys like:
      - img_name / image_file
      - sentences: list[str]
      - segmentation: polygon or RLE
    """
    with open(json_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Dict[str, Any]] = []
    for item in data:
        img_name = item.get("image_file", item.get("img_name", item.get("file_name", "")))
        image_path = os.path.join(image_folder, img_name)
        if not os.path.exists(image_path):
            # allow alternative key
            alt = item.get("image_path", "")
            if alt and os.path.exists(alt):
                image_path = alt
            else:
                continue
        samples.append(
            {
                "image_path": image_path,
                "image_file": img_name,
                "sentences": item.get("sentences", item.get("text", [])),
                "segmentation": item.get("segmentation", None),
            }
        )
        if max_samples is not None and len(samples) >= int(max_samples):
            break
    return samples


def load_coco_polygon_mask(segmentation: Any, image_w: int, image_h: int) -> Optional[torch.Tensor]:
    """
    Accept COCO polygon list or RLE dict/list.
    Return: (H,W) uint8 tensor {0,1}
    """
    if segmentation is None:
        return None
    try:
        from pycocotools import mask as maskUtils
    except Exception as e:
        print(f"[WARN] pycocotools not available: {e}")
        return None

    # RLE dict
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        rle = segmentation
        m = maskUtils.decode(rle)  # (H,W) or (H,W,1)
        if m.ndim == 3:
            m = m[:, :, 0]
        return torch.from_numpy(m.astype(np.uint8))

    # list: could be polygons or list of RLEs
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return None

        # list of dict RLEs
        if isinstance(segmentation[0], dict) and "counts" in segmentation[0]:
            rles = segmentation
            m = np.zeros((image_h, image_w), dtype=np.uint8)
            for rle in rles:
                mi = maskUtils.decode(rle)
                if mi.ndim == 3:
                    mi = mi[:, :, 0]
                m = np.maximum(m, mi.astype(np.uint8))
            return torch.from_numpy(m)

        # polygons: list of list[float]
        rles = maskUtils.frPyObjects(segmentation, image_h, image_w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        return torch.from_numpy(m.astype(np.uint8))

    return None


# ============================================================
# Visualization helpers
# ============================================================
def _get_font(size: int = 18) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def overlay_mask_on_image(img: Image.Image, mask_bool: np.ndarray, alpha: float = 0.45, color=(255, 0, 0)) -> Image.Image:
    if mask_bool is None:
        return img
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool.astype(bool)

    base = img.convert("RGBA")
    ov_np = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    ov_np[mask_bool] = np.array([color[0], color[1], color[2], int(255 * alpha)], dtype=np.uint8)
    overlay = Image.fromarray(ov_np, mode="RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

def draw_bbox_and_text(img: Image.Image, bbox_xyxy: Optional[List[int]], lines: List[str]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _get_font(18)

    if bbox_xyxy is not None:
        x0, y0, x1, y1 = bbox_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)

    if lines:
        text = "\n".join(lines)
        try:
            bb = draw.multiline_textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = (420, 80)

        pad = 6
        x, y = 8, 8
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
        draw.multiline_text((x, y), text, fill=(255, 255, 255), font=font)
    return out

def concat_h(images: List[Image.Image]) -> Image.Image:
    images = [im.convert("RGB") for im in images]
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    H = max(heights)
    W = sum(widths)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas

# ===========================
# [ADD] Gate heatmap + 2x3 grid + origin mask
# ===========================
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def grid_2x3(imgs: List[Image.Image], bg=(0, 0, 0)) -> Image.Image:
    """
    imgs: length==6, already same size ideally
    layout:
      [0,1,2]
      [3,4,5]
    """
    assert len(imgs) == 6
    imgs = [im.convert("RGB") for im in imgs]
    W = max(im.size[0] for im in imgs)
    H = max(im.size[1] for im in imgs)
    # force same size for neat layout
    imgs = [im.resize((W, H), Image.BILINEAR) for im in imgs]

    canvas = Image.new("RGB", (3 * W, 2 * H), bg)
    canvas.paste(imgs[0], (0 * W, 0 * H))
    canvas.paste(imgs[1], (1 * W, 0 * H))
    canvas.paste(imgs[2], (2 * W, 0 * H))
    canvas.paste(imgs[3], (0 * W, 1 * H))
    canvas.paste(imgs[4], (1 * W, 1 * H))
    canvas.paste(imgs[5], (2 * W, 1 * H))
    return canvas


def draw_bbox_and_text_custom(
    img: Image.Image,
    bbox_xyxy: Optional[List[int]],
    lines: List[str],
    bbox_width: int = 3,
    bbox_color=(0, 255, 0),
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _get_font(18)

    if bbox_xyxy is not None:
        x0, y0, x1, y1 = bbox_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=bbox_color, width=int(bbox_width))

    if lines:
        text = "\n".join(lines)
        try:
            bb = draw.multiline_textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = (420, 80)

        pad = 6
        x, y = 8, 8
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
        draw.multiline_text((x, y), text, fill=(255, 255, 255), font=font)
    return out


@torch.no_grad()
def sam3_predict_origin_text_bbox_mask(
    img: Image.Image,
    label: str,
    bbox_xyxy_image: List[int],
) -> Optional[np.ndarray]:
    """
    Origin behavior: text + bbox prompt simultaneously in SAM3.
    Output: bool mask at image resolution (H,W)
    """
    ok = lazy_init_sam3()
    if not ok:
        return None
    assert _SAM3_PROCESSOR is not None

    W, H = img.size
    state = _SAM3_PROCESSOR.set_image(img)

    # text prompt
    state = _SAM3_PROCESSOR.set_text_prompt(prompt=label, state=state)

    # bbox prompt (normalized cxcywh)
    bbox_xywh_norm = xyxy_to_xywh_norm(bbox_xyxy_image, W=W, H=H)
    # NOTE: Sam3Processor expects torch tensor or list depending on implementation; follow your branch code style
    if hasattr(_SAM3_PROCESSOR, "add_geometric_prompt_alloutput"):
        out = _SAM3_PROCESSOR.add_geometric_prompt_alloutput(box=bbox_xywh_norm, label=True, state=state)
    else:
        out = _SAM3_PROCESSOR.add_geometric_prompt(box=bbox_xywh_norm, label=True, state=state)
        if not isinstance(out, dict):
            out = state

    if not isinstance(out, dict) or ("masks" not in out) or is_empty_masks(out["masks"]):
        return None

    # select best by bbox IoU (same as your branch selection)
    sel = select_best_mask_by_bbox_iou_scaled(out, [float(v) for v in bbox_xyxy_image], img_hw=(H, W))
    m = sel["best_mask"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)  # (Hm,Wm) or already (H,W)
    m = ensure_same_hw(m, target_h=H, target_w=W)
    return m


def gate_heatmap_with_colorbar(
    gate_map_hw: np.ndarray,
    union_mask_hw: np.ndarray,
    out_size_wh: Tuple[int, int],
    title: str = "Gate heatmap",
    vmin: float = 0.0,
    vmax: float = 1.0,
    center: float = 0.5,
) -> Image.Image:
    """
    gate_map_hw: (Hg,Wg) in [0,1]
    union_mask_hw: (H,W) bool (where to show gate)
    out_size_wh: (W,H) final size

    Visual: use a diverging colormap centered at 0.5 with a clear boundary.
    """
    W, H = out_size_wh
    if gate_map_hw is None:
        return Image.new("RGB", (W, H), (0, 0, 0))

    gate_big = cv2.resize(gate_map_hw.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    if union_mask_hw is not None:
        gate_big = gate_big.copy()
        gate_big[~union_mask_hw.astype(bool)] = 0.0

    # --- key: make 0.5 a strong visual boundary ---
    from matplotlib.colors import TwoSlopeNorm

    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    fig = plt.figure(figsize=(W / 120.0, H / 120.0), dpi=120)
    ax = fig.add_subplot(111)

    # diverging cmap: obvious “two sides”
    im = ax.imshow(gate_big, norm=norm, cmap="seismic")  # or "coolwarm"

    # draw explicit boundary at 0.5
    try:
        ax.contour(gate_big, levels=[center], colors="black", linewidths=1.2)
    except Exception:
        pass

    ax.set_title(title + " (center=0.5)", fontsize=10)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"])

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    hm = Image.open(buf).convert("RGB")
    hm = hm.resize((W, H), Image.BILINEAR)
    return hm


# ============================================================
# Prompt
# ============================================================
QUESTION_TEMPLATE = (
    "{Question} First output the thinking process in <think> </think> tags and then output the final answer in "
    "<answer> </answer> tags. Output the final answer in JSON format."
)
problem_template = "Please provide the bounding box coordinate and the label of the region this sentence describes: <sentence>"


# ============================================================
# Cache IO
# ============================================================
def read_jsonl(path: str) -> List[Any]:
    outs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outs.append(json.loads(line))
    return outs


# ============================================================
# Eval on main rank (SAM3 + PixelGate fusion)
# ============================================================
def evaluate_and_save(meta_all: List[Dict[str, Any]], outputs_all: List[Any], n_eval: int):
    assert rank == main_rank

    os.makedirs(VIS_DIR, exist_ok=True)
    _ = lazy_init_sam3()
    _ = lazy_init_fusion_net()

    final_output = []

    # PixelGate fused metrics (existing)
    iou_list, intersection_list, union_list = [], [], []
    success_05 = 0

    # text/bbox branch metrics (existing)
    iou_text_list, inter_text_list, union_text_list = [], [], []
    iou_bbox_list, inter_bbox_list, union_bbox_list = [], [], []

    # [NEW] Avg-Logits (0.5/0.5) metrics
    iou_avg_list, inter_avg_list, union_avg_list = [], [], []
    success_05_avg = 0

    # [NEW] gate statistics on GT-foreground
    gate_fg_list = []        # gate mean on gt foreground
    iou_diff_list = []       # (iou_text - iou_bbox) for corresponding samples
    gate_valid_cnt = 0

    iou_oracle_list, inter_oracle_list, union_oracle_list = [], [], []
    success_05_oracle = 0

    vis_saved = 0

    for idx in tqdm(range(n_eval), total=n_eval):
        s = meta_all[idx]
        output_text, input_h, input_w, _, _ = outputs_all[idx]

        image_path = s["image_path"]
        query_text = s.get("sentence", "")
        gt_seg = s.get("gt_segmentation", None)

        # open original image
        img0 = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        orig_W, orig_H = img0.size

        # optional resize image for evaluation/SAM3 prompting
        img = img0
        scale = 1.0
        if MAX_IMAGE_SIDE is not None:
            w, h = img.size
            scale = min(float(MAX_IMAGE_SIDE) / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        W, H = img.size

        pred_bbox_in = extract_bbox_answer(output_text)
        pred_bbox_img_f = resize_bbox_xyxy(pred_bbox_in, input_h, input_w, H, W)
        pred_bbox_xyxy = sanitize_xyxy_int(pred_bbox_img_f, width=W, height=H)
        pred_label = extract_label_answer(output_text)

        # GT mask decoded at original size, then resized if needed
        gt0 = load_coco_polygon_mask(gt_seg, image_w=orig_W, image_h=orig_H)
        gt_mask = gt0.cpu().numpy().astype(bool) if gt0 is not None else None
        if gt_mask is not None and (W != orig_W or H != orig_H):
            gt_mask = ensure_same_hw(gt_mask, target_h=H, target_w=W)

        text_pred = make_empty_mask(H, W)
        bbox_pred = make_empty_mask(H, W)
        fused_pred = make_empty_mask(H, W)
        avg05_pred = make_empty_mask(H, W)

        fusion_info = {}

        if pred_bbox_xyxy is not None and gt_mask is not None:
            use_label = pred_label if (pred_label is not None and len(pred_label.strip()) > 0) else query_text

            tmask, bmask, fmask, amask, fusion_info = sam3_predict_mask_pixel_gate_fusion_with_branches(
                img=img,
                label=use_label,
                bbox_xyxy_image=pred_bbox_xyxy,
            )

            if tmask is not None:
                text_pred = ensure_same_hw(tmask, target_h=H, target_w=W)
            if bmask is not None:
                bbox_pred = ensure_same_hw(bmask, target_h=H, target_w=W)
            if fmask is not None:
                fused_pred = ensure_same_hw(fmask, target_h=H, target_w=W)
            if amask is not None:
                avg05_pred = ensure_same_hw(amask, target_h=H, target_w=W)

        # ---- IoUs ----
        iou_t, inter_t, uni_t = mask_iou(text_pred, gt_mask)
        iou_b, inter_b, uni_b = mask_iou(bbox_pred, gt_mask)
        iou_f, inter_f, uni_f = mask_iou(fused_pred, gt_mask)
        iou_a, inter_a, uni_a = mask_iou(avg05_pred, gt_mask)

        # branch lists
        iou_text_list.append(iou_t); inter_text_list.append(inter_t); union_text_list.append(uni_t)
        iou_bbox_list.append(iou_b); inter_bbox_list.append(inter_b); union_bbox_list.append(uni_b)

        # pixel-gate fused lists
        iou_list.append(iou_f); intersection_list.append(inter_f); union_list.append(uni_f)
        if iou_f > 0.5:
            success_05 += 1

        # avg05 lists
        iou_avg_list.append(iou_a); inter_avg_list.append(inter_a); union_avg_list.append(uni_a)
        if iou_a > 0.5:
            success_05_avg += 1

        oracle_pred = make_empty_mask(H, W)

        # need both processed branch logits (in info), otherwise fallback to better branch mask
        la = fusion_info.get("_logits_a", None) if isinstance(fusion_info, dict) else None
        lb = fusion_info.get("_logits_b", None) if isinstance(fusion_info, dict) else None

        if (la is not None) and (lb is not None):
            # la/lb: torch CPU, shape (1,1,Hm,Wm), logits-space
            wa, wb = softmax2(iou_t, iou_b, scale=ORACLE_SOFTMAX_SCALE)

            # weighted sum logits
            oracle_logits = wa * la + wb * lb  # torch CPU
            oracle_prob = torch.sigmoid(oracle_logits)
            oracle_mask_small = (oracle_prob > 0.5).squeeze(0).squeeze(0).numpy().astype(bool)

            # resize to (H,W)
            oracle_pred = ensure_same_hw(oracle_mask_small, target_h=H, target_w=W)
        else:
            # fallback: choose the better branch mask
            oracle_pred = text_pred if (iou_t >= iou_b) else bbox_pred

        iou_o, inter_o, uni_o = mask_iou(oracle_pred, gt_mask)
        iou_oracle_list.append(iou_o)
        inter_oracle_list.append(inter_o)
        union_oracle_list.append(uni_o)
        if iou_o > 0.5:
            success_05_oracle += 1

        # ---- [NEW] gate mean on GT foreground + relation to branch IoU ----
        gate_fg_mean = None
        if isinstance(fusion_info, dict) and fusion_info.get("gate_map", None) is not None:
            gate_map = fusion_info["gate_map"]  # (Hg,Wg)
            gate_fg_mean = gate_mean_on_gt_foreground(gate_map, gt_mask)
            if gate_fg_mean is not None:
                gate_valid_cnt += 1
                gate_fg_list.append(gate_fg_mean)
                iou_diff_list.append(iou_t - iou_b)  # positive => text better => expect gate_fg_mean higher

        # ---- visualization (unchanged panels: GT/Text/BBox/Fused) ----
        do_save = True
        if VIS_MAX_SAVE is not None and vis_saved >= int(VIS_MAX_SAVE):
            do_save = False

        if do_save:
            # ----------------------------
            # [ADD] compute origin mask + IoU (only for visualization / record)
            # ----------------------------
            origin_pred = make_empty_mask(H, W)
            iou_origin = 0.0
            if pred_bbox_xyxy is not None and gt_mask is not None:
                use_label = pred_label if (pred_label is not None and len(pred_label.strip()) > 0) else query_text
                om = sam3_predict_origin_text_bbox_mask(img=img, label=use_label, bbox_xyxy_image=pred_bbox_xyxy)
                if om is not None:
                    origin_pred = ensure_same_hw(om, target_h=H, target_w=W)
                    iou_origin, _, _ = mask_iou(origin_pred, gt_mask)

            # union region for gate visualization: text ∪ bbox ∪ GT
            union_mask = None
            if gt_mask is not None:
                union_mask = (text_pred.astype(bool) | bbox_pred.astype(bool) | gt_mask.astype(bool))

            # ----------------------------
            # [TEXT] overlay + show label
            # ----------------------------
            img_text = overlay_mask_on_image(img, text_pred, alpha=VIS_ALPHA)
            img_text = draw_bbox_and_text_custom(
                img_text,
                bbox_xyxy=None,
                lines=[
                    "Text branch",
                    f"label: {pred_label}",
                    f"IoU: {iou_t:.3f}",
                ],
                bbox_width=3,
            )

            # ----------------------------
            # [BBOX] overlay + draw thick bbox
            # ----------------------------
            img_bbox = overlay_mask_on_image(img, bbox_pred, alpha=VIS_ALPHA)
            img_bbox = draw_bbox_and_text_custom(
                img_bbox,
                bbox_xyxy=pred_bbox_xyxy,
                lines=[
                    "BBox branch",
                    f"IoU: {iou_b:.3f}",
                ],
                bbox_width=6,   # thicker
                bbox_color=(0, 255, 0),
            )

            # ----------------------------
            # [ORIGIN] overlay + show IoU
            # ----------------------------
            img_origin = overlay_mask_on_image(img, origin_pred, alpha=VIS_ALPHA)
            img_origin = draw_bbox_and_text_custom(
                img_origin,
                bbox_xyxy=pred_bbox_xyxy,  # optional: show bbox too
                lines=[
                    "Origin (Text+BBox)",
                    f"IoU: {iou_origin:.3f}",
                ],
                bbox_width=4,
                bbox_color=(255, 255, 0),
            )

            # ----------------------------
            # [GT] overlay + show query text
            # ----------------------------
            img_gt = overlay_mask_on_image(img, gt_mask, alpha=VIS_ALPHA)
            img_gt = draw_bbox_and_text_custom(
                img_gt,
                bbox_xyxy=None,
                lines=[
                    "GT",
                    f"query: {query_text}",
                ],
                bbox_width=3,
            )

            # ----------------------------
            # [FUSED] overlay + show IoU + optional gate_fg
            # ----------------------------
            extra = ""
            if gate_fg_mean is not None:
                extra = f"gate_fg={gate_fg_mean:.3f}"
            img_fused = overlay_mask_on_image(img, fused_pred, alpha=VIS_ALPHA)
            img_fused = draw_bbox_and_text_custom(
                img_fused,
                bbox_xyxy=None,
                lines=[
                    "Fused (PixelGate)",
                    f"IoU: {iou_f:.3f}",
                    extra,
                ] if extra else [
                    "Fused (PixelGate)",
                    f"IoU: {iou_f:.3f}",
                ],
                bbox_width=3,
            )

            # ----------------------------
            # [GATE HEATMAP] show only union region; background=0; with legend
            # ----------------------------
            gate_map = fusion_info.get("gate_map", None) if isinstance(fusion_info, dict) else None
            img_gate = gate_heatmap_with_colorbar(
                gate_map_hw=gate_map,
                union_mask_hw=union_mask,
                out_size_wh=(W, H),
                title="Gate heatmap (union region only)",
                vmin=0.0,
                vmax=1.0,
            )

            # ----------------------------
            # 2x3 layout:
            #   [Text, BBox, Origin]
            #   [GT,   Fused, Gate]
            # ----------------------------
            panel = grid_2x3([img_text, img_bbox, img_origin, img_gt, img_fused, img_gate])

            base = os.path.splitext(os.path.basename(image_path))[0]
            out_png = os.path.join(
                VIS_DIR,
                f"{idx:06d}_{base}_f{iou_f:.3f}_o{iou_origin:.3f}_t{iou_t:.3f}_b{iou_b:.3f}.png"
            )
            panel.save(out_png)
            vis_saved += 1


        final_output.append(
            {
                "idx": idx,
                "global_idx": s.get("global_idx", idx),
                "image_file": s.get("image_file", ""),
                "image_path": image_path,
                "sentence": query_text,
                "pred_label": pred_label,
                "model_output": output_text,
                "pred_bbox_in": pred_bbox_in,
                "pred_bbox_img": pred_bbox_xyxy,

                "iou_text": float(iou_t),
                "iou_bbox": float(iou_b),

                # PixelGate fused
                "iou_fused": float(iou_f),
                "iou_oracle": float(iou_o),

                # [NEW] Avg05 fusion
                "iou_avg05": float(iou_a),
                "iou_origin": float(iou_origin),

                # [NEW] gate mean on GT foreground
                "gate_fg_mean": (float(gate_fg_mean) if gate_fg_mean is not None else None),
            }
        )

    # -------------------------
    # summary metrics: gIoU + cIoU
    # -------------------------
    def compute_giou_ciou(iou_list_, inter_list_, union_list_):
        g_iou = float(sum(iou_list_) / max(1, len(iou_list_)))
        acc_inter = float(sum(inter_list_) / max(1, len(inter_list_)))
        acc_union = float(sum(union_list_) / max(1, len(union_list_)))
        c_iou = float(acc_inter / max(1e-6, acc_union))
        return g_iou, c_iou

    giou_t, ciou_t = compute_giou_ciou(iou_text_list, inter_text_list, union_text_list)
    giou_b, ciou_b = compute_giou_ciou(iou_bbox_list, inter_bbox_list, union_bbox_list)

    giou_f, ciou_f = compute_giou_ciou(iou_list, intersection_list, union_list)
    acc_05 = float(success_05 / max(1, len(iou_list)))

    giou_a, ciou_a = compute_giou_ciou(iou_avg_list, inter_avg_list, union_avg_list)
    acc_05_a = float(success_05_avg / max(1, len(iou_avg_list)))

    giou_o, ciou_o = compute_giou_ciou(iou_oracle_list, inter_oracle_list, union_oracle_list)
    acc_05_o = float(success_05_oracle / max(1, len(iou_oracle_list)))

    # -------------------------
    # [NEW] gate-vs-branch relation analysis
    # -------------------------
    gate_analysis = {
        "num_gate_valid": int(gate_valid_cnt),
        "pearson_corr_gatefg_vs_iou_diff": None,
        "agreement_rate": None,
        "mean_gatefg_when_text_better": None,
        "mean_gatefg_when_bbox_better": None,
    }

    if gate_valid_cnt > 1:
        g = np.array(gate_fg_list, dtype=np.float32)
        d = np.array(iou_diff_list, dtype=np.float32)  # iou_text - iou_bbox
        # Pearson correlation
        if float(g.std()) > 1e-8 and float(d.std()) > 1e-8:
            corr = float(np.corrcoef(g, d)[0, 1])
            gate_analysis["pearson_corr_gatefg_vs_iou_diff"] = corr

        # Direction agreement: if text better (d>0), expect gate_fg > 0.5; if bbox better (d<0), expect gate_fg < 0.5
        pos = d > 1e-8
        neg = d < -1e-8
        agree = 0
        total = int(pos.sum() + neg.sum())
        if total > 0:
            agree += int(((g[pos] > 0.5).sum()) if pos.any() else 0)
            agree += int(((g[neg] < 0.5).sum()) if neg.any() else 0)
            gate_analysis["agreement_rate"] = float(agree / max(1, total))

        if pos.any():
            gate_analysis["mean_gatefg_when_text_better"] = float(g[pos].mean())
        if neg.any():
            gate_analysis["mean_gatefg_when_bbox_better"] = float(g[neg].mean())

    print(
        f"\n[RESULT] ReasonSeg {SPLIT} (N={len(iou_list)})\n"
        f"  Text  : gIoU={giou_t:.4f}, cIoU={ciou_t:.4f}\n"
        f"  BBox  : gIoU={giou_b:.4f}, cIoU={ciou_b:.4f}\n"
        f"  Avg05 : gIoU={giou_a:.4f}, cIoU={ciou_a:.4f}, IoU>0.5={acc_05_a*100:.2f}%\n"
        f"  Gate  : gIoU={giou_f:.4f}, cIoU={ciou_f:.4f}, IoU>0.5={acc_05*100:.2f}%\n"
        f"  Oracle: gIoU={giou_o:.4f}, cIoU={ciou_o:.4f}, IoU>0.5={acc_05_o*100:.2f}%  (scale={ORACLE_SOFTMAX_SCALE})\n"
        f"  GateFG analysis: valid={gate_analysis['num_gate_valid']}, "
        f"pearson={gate_analysis['pearson_corr_gatefg_vs_iou_diff']}, "
        f"agree={gate_analysis['agreement_rate']}, "
        f"mean_gatefg(text>bbox)={gate_analysis['mean_gatefg_when_text_better']}, "
        f"mean_gatefg(bbox>text)={gate_analysis['mean_gatefg_when_bbox_better']}\n"
    )

    if _SAM3_IMPORT_ERROR is not None:
        print(f"[WARN] SAM3 init error: {_SAM3_IMPORT_ERROR}")
    if _FUSION_ERROR is not None:
        print(f"[WARN] Fusion init/load error: {_FUSION_ERROR}")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": RUN_NAME,
                "model_path": MODEL_PATH,
                "num_samples": len(iou_list),

                "metrics": {
                    "text": {"gIoU": giou_t, "cIoU": ciou_t},
                    "bbox": {"gIoU": giou_b, "cIoU": ciou_b},

                    # [NEW] avg05 vs gate
                    "avg05": {"gIoU": giou_a, "cIoU": ciou_a, "acc_iou_gt_0.5": acc_05_a},
                    "pixel_gate": {"gIoU": giou_f, "cIoU": ciou_f, "acc_iou_gt_0.5": acc_05},
                },

                # [NEW] gate analysis summary
                "gate_analysis": gate_analysis,

                "vis_dir": VIS_DIR,
                "vis_saved": vis_saved,
                "fusion_ckpt": FUSION_CKPT if USE_PIXEL_GATE_FUSION else None,
                "results": final_output,
                "oracle": {"gIoU": giou_o, "cIoU": ciou_o, "acc_iou_gt_0.5": acc_05_o, "scale": ORACLE_SOFTMAX_SCALE},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Saved results to {OUTPUT_PATH}")
    print(f"[INFO] Visualization saved to {VIS_DIR}")
    print("-" * 100)


# ============================================================
# Main (ReasonSeg-dgseg_fusion run mode, RefCOCO dataset build)
# ============================================================
def main():
    if rank == main_rank:
        print(f"[INFO] Evaluating {DATASET} {SPLIT}: {REFCOCO_JSON_DIR}")

    # load samples (image + segmentation + sentences)
    samples = load_refcoco_json(REFCOCO_JSON_DIR, image_folder=IMAGE_FOLDER, max_samples=MAX_SAMPLES)

    # flatten to (image, sentence) items like your refcoco logits script
    flat_items: List[Dict[str, Any]] = []
    for s in samples:
        texts = s.get("sentences", [])
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            texts = []
        for t in texts:
            flat_items.append(
                {
                    "image_path": s["image_path"],
                    "image_file": s.get("image_file", ""),
                    "sentence": t,
                    "gt_segmentation": s.get("segmentation", None),
                }
            )

    total_items = len(flat_items)
    if rank == main_rank:
        print(f"[INFO] samples={len(samples)} -> flattened items={total_items}")

    # -------------------------
    # DEBUG: if have cached MLLM jsonl, only main rank evaluates
    # -------------------------
    if DEBUG_MODE and os.path.exists(TARGET_JSON_FILE):
        if rank != main_rank:
            dist.barrier()
            return

        print(f"[INFO] Loading cached MLLM output from: {TARGET_JSON_FILE}")
        cached = read_jsonl(TARGET_JSON_FILE)
        if len(cached) < total_items:
            print(f"[WARN] cached outputs ({len(cached)}) < items ({total_items}). Will evaluate min(len).")
        n_eval = min(len(cached), total_items)

        meta_all = []
        for gi in range(n_eval):
            it = flat_items[gi]
            meta_all.append(
                {
                    "global_idx": gi,
                    "image_path": it["image_path"],
                    "image_file": it.get("image_file", ""),
                    "sentence": it.get("sentence", ""),
                    "gt_segmentation": it.get("gt_segmentation", None),
                }
            )

        outputs_all = []
        for i in range(n_eval):
            mo = cached[i]
            if isinstance(mo, (list, tuple)) and len(mo) >= 5:
                outputs_all.append(mo[:5])
            elif isinstance(mo, dict):
                outputs_all.append(
                    [
                        mo.get("output_text", mo.get("model_output", "")),
                        mo.get("input_h", mo.get("input_height", 1)),
                        mo.get("input_w", mo.get("input_width", 1)),
                        mo.get("img_h", mo.get("img_height", 1)),
                        mo.get("img_w", mo.get("img_width", 1)),
                    ]
                )
            else:
                outputs_all.append(["", 1, 1, 1, 1])

        evaluate_and_save(meta_all, outputs_all, n_eval)
        dist.barrier()
        return

    # -------------------------
    # DDP MLLM inference (all ranks)
    # -------------------------
    per_rank = total_items // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else total_items
    my_indices = list(range(start_idx, end_idx))
    my_items = [flat_items[i] for i in my_indices]

    # build model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    messages = []
    meta_rank = []
    for gi, it in zip(my_indices, my_items):
        image_path = it["image_path"]
        sent = it.get("sentence", "")

        problem = problem_template.replace("<sentence>", sent)
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}", "max_pixels": RESIZE * RESIZE},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=problem)},
                ],
            }
        ]
        messages.append(msg)
        meta_rank.append(
            {
                "global_idx": gi,
                "image_path": image_path,
                "image_file": it.get("image_file", ""),
                "sentence": sent,
                "gt_segmentation": it.get("gt_segmentation", None),
            }
        )

    rank_items: List[Dict[str, Any]] = []

    for bi in tqdm(range(0, len(messages), BSZ), disable=(rank != main_rank)):
        batch_messages = messages[bi : bi + BSZ]
        batch_meta = meta_rank[bi : bi + BSZ]

        text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)

        fixed_image_inputs = []
        for im in image_inputs:
            if isinstance(im, Image.Image):
                im = ImageOps.exif_transpose(im).convert("RGB")
            fixed_image_inputs.append(im)

        inputs = processor(
            text=text,
            images=fixed_image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for j, output_text in enumerate(batch_output_text):
            input_h = int(inputs["image_grid_thw"][j][1] * 14)
            input_w = int(inputs["image_grid_thw"][j][2] * 14)

            img_path = batch_messages[j][0]["content"][0]["image"].split("file://")[1]
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size

            rank_items.append(
                {
                    "global_idx": int(batch_meta[j]["global_idx"]),
                    "meta": batch_meta[j],
                    "model_out": [output_text, input_h, input_w, img_h, img_w],
                }
            )

    # optional per-rank dump
    if DEBUG_MODE:
        os.makedirs(os.path.dirname(TARGET_JSON_FILE) or ".", exist_ok=True)
        with open(f"{TARGET_JSON_FILE}.rank{rank}.jsonl", "w", encoding="utf-8") as f:
            for it in rank_items:
                f.write(json.dumps(it["model_out"], ensure_ascii=False) + "\n")

    # gather to main rank
    gathered = [None] * world_size
    dist.all_gather_object(gathered, rank_items)

    if rank == main_rank:
        flat: List[Dict[str, Any]] = []
        for chunk in gathered:
            if chunk:
                flat.extend(chunk)
        flat.sort(key=lambda x: x["global_idx"])

        meta_all = []
        outputs_all = []
        for it in flat:
            meta_all.append(it["meta"])
            outputs_all.append(it["model_out"])

        # write merged jsonl (so later you can run debug eval)
        if DEBUG_MODE:
            with open(TARGET_JSON_FILE, "w", encoding="utf-8") as f:
                for mo in outputs_all:
                    f.write(json.dumps(mo, ensure_ascii=False) + "\n")
            print(f"[INFO] Wrote merged MLLM outputs to: {TARGET_JSON_FILE}")

        if not CACHE_ONLY:
            evaluate_and_save(meta_all, outputs_all, len(outputs_all))

    dist.barrier()


if __name__ == "__main__":
    main()
