# eval_reasonseg.py
# -*- coding: utf-8 -*-
"""
ReasonSeg evaluation with:
- Multi-instance parsing from <answer>...</answer>
- For each instance: SAM3 text branch + bbox branch + PixelGate fusion (+ avg05) (used by DGSeg)
- OR-merge instance masks as final per-sample prediction (used by DGSeg)
- Also compute ORIGIN baseline: SAM3 with BOTH prompts (text + bbox) simultaneously
- Optional visualization: 2x3 panels (Text / BBox / Origin ; GT / Fused / GateHeatmap)
  - Text panel: show label (used_label)
  - BBox panel: draw bboxes (thicker)
  - GT panel: show query
  - Gate heatmap: show only union(GT, Text, BBox) region, background=0, centered at 0.5 with clear boundary + legend
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
SPLIT = os.environ.get("SPLIT", "test")
RESIZE = int(os.environ.get("RESIZE", "1024"))
TEST_STEP = int(os.environ.get("TEST_STEP", "2250"))

RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"Qwen2.5-VL-7B-Instruct-reasonseg_{SPLIT}-{RESIZE}{RESIZE}-dgseg_fusion",
)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

REASONSEG_JSON_DIR = os.environ.get("REASONSEG_JSON_DIR", os.path.join(os.environ.get("DATA_ROOT", "./data"), "ReasonSeg", SPLIT))
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    f"../reasonseg_output/overall_json/reasonseg_seg_results_{RUN_NAME}.json",
)
VIS_DIR = os.environ.get("VIS_DIR", f"../reasonseg_output/vis/reasonseg_vis_{RUN_NAME}")


FUSION_CKPT = os.environ.get("FUSION_CKPT", os.path.join(os.environ.get("DGSEG_ROOT", "."), "checkpoints", "sam3_fusion.pt"))


# Inference
BSZ = 1
MAX_SAMPLES = None
MAX_IMAGE_SIDE = None  # e.g., 1024
MAX_NEW_TOKENS = 256

# Debug: if TARGET_JSON_FILE exists, skip MLLM inference and directly evaluate
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").lower() in {"1", "true", "yes"}
TARGET_JSON_FILE = os.environ.get(
    "TARGET_JSON_FILE",
    f"../reasonseg_output/overall_json/{RUN_NAME}.jsonl",
)

# Visualization
VIS_MAX_SAVE = 500
VIS_ALPHA = 0.45

# SAM3
# -----------------------------
SAM3_REPO_PATH = os.environ.get("SAM3_REPO_PATH", os.path.join(os.environ.get("DGSEG_ROOT", "."), "sam3"))
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", os.path.join(os.environ.get("MODEL_ROOT", "./models"), "sam3.pt"))

# PixelGate fusion
USE_PIXEL_GATE_FUSION = True
FUSION_IN_CHANNELS = 256
FUSION_HIDDEN = 64
FUSION_GATE_T = 1.0
FUSION_GATE_CLAMP = (0.01, 0.99)
ORACLE_SOFTMAX_SCALE = 100.0


# ============================================================
# SAM3 lazy init
# ============================================================
if SAM3_REPO_PATH not in os.sys.path:
    os.sys.path.append(SAM3_REPO_PATH)

_SAM3_MODEL = None
_SAM3_PROCESSOR = None
_SAM3_READY = False
_SAM3_IMPORT_ERROR = None

_FUSION_NET = None
_FUSION_READY = False
_FUSION_ERROR = None


def lazy_init_sam3() -> bool:
    global _SAM3_MODEL, _SAM3_PROCESSOR, _SAM3_READY, _SAM3_IMPORT_ERROR
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
        _SAM3_MODEL = sam3_model
        _SAM3_PROCESSOR = Sam3Processor(sam3_model)
        _SAM3_READY = True
        return True
    except Exception as e:
        _SAM3_IMPORT_ERROR = e
        _SAM3_READY = False
        _SAM3_PROCESSOR = None
        return False


# ============================================================
# PixelGate fusion net
# ============================================================
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


def sigmoid_to_logits(sigmoid: torch.Tensor) -> torch.Tensor:
    sigmoid = sigmoid.clamp(1e-6, 1 - 1e-6)
    return torch.log(sigmoid / (1 - sigmoid))


# ============================================================
# Utilities
# ============================================================
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


def resize_bbox_xyxy(
    bbox: List[int],
    input_height: int,
    input_width: int,
    image_height: int,
    image_width: int,
) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = x1 / max(1.0, float(input_width)) * float(image_width)
    y1 = y1 / max(1.0, float(input_height)) * float(image_height)
    x2 = x2 / max(1.0, float(input_width)) * float(image_width)
    y2 = y2 / max(1.0, float(input_height)) * float(image_height)
    return [x1, y1, x2, y2]


def xyxy_to_xywh_norm(xyxy: List[int], W: int, H: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1n = x1 / max(1.0, W)
    y1n = y1 / max(1.0, H)
    x2n = x2 / max(1.0, W)
    y2n = y2 / max(1.0, H)
    x1n, y1n, x2n, y2n = [max(0.0, min(1.0, v)) for v in [x1n, y1n, x2n, y2n]]
    w = max(0.0, x2n - x1n)
    h = max(0.0, y2n - y1n)
    return [x1n + 0.5 * w, y1n + 0.5 * h, w, h]


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


# ----------------------------
# multi-instance parser
# ----------------------------
def _extract_answer_body(content: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _normalize_instance(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    bbox = None
    for k in ("bbox_2d", "bbox", "box", "bbox_xyxy"):
        if k in obj:
            bbox = obj.get(k)
            break

    if bbox is None:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        bbox4 = [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))]
    except Exception:
        return None

    label = None
    for k in ("label", "label_text", "description", "name"):
        if k in obj and isinstance(obj[k], str):
            label = obj[k].strip()
            break

    return {"bbox_2d": bbox4, "label": label}


def extract_instances_from_output(content: str) -> List[Dict[str, Any]]:
    body = _extract_answer_body(content)
    if not body:
        return []

    body_s = body.strip()

    # 1) try direct JSON
    for candidate in (body_s, body_s.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                inst = _normalize_instance(obj)
                return [inst] if inst is not None else []
            if isinstance(obj, list):
                out = []
                for it in obj:
                    if isinstance(it, dict):
                        inst = _normalize_instance(it)
                        if inst is not None:
                            out.append(inst)
                return out
        except Exception:
            pass

    # 2) try find {...} blocks
    blocks = re.findall(r"\{[^{}]*\}", body_s, flags=re.DOTALL)
    out_blocks = []
    for b in blocks:
        bb = b.strip()
        for candidate in (bb, bb.replace("'", '"')):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    inst = _normalize_instance(obj)
                    if inst is not None:
                        out_blocks.append(inst)
                break
            except Exception:
                continue
    if out_blocks:
        return out_blocks

    # 3) regex fallback
    pat = re.compile(
        r"bbox_2d\s*[:=]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\].*?"
        r"label\s*[:=]\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE | re.DOTALL,
    )
    out = []
    for m in pat.finditer(body_s):
        bbox4 = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
        label = m.group(5).strip()
        out.append({"bbox_2d": bbox4, "label": label})
    return out


def mask_iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> Tuple[float, float, float]:
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 0.0, 0.0, 0.0
    return float(inter / union), float(inter), float(union)


def ensure_same_hw(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask_bool is None:
        return None
    if mask_bool.shape[0] == target_h and mask_bool.shape[1] == target_w:
        return mask_bool.astype(bool)
    m = mask_bool.astype(np.uint8) * 255
    m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (m > 0)


def softmax2(a: float, b: float, scale: float = 100.0) -> Tuple[float, float]:
    x0 = a * scale
    x1 = b * scale
    m = max(x0, x1)
    e0 = np.exp(x0 - m)
    e1 = np.exp(x1 - m)
    s = e0 + e1
    return float(e0 / s), float(e1 / s)


# ============================================================
# ReasonSeg loader
# ============================================================
def load_reasonseg_json(json_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_samples is not None and max_samples < 0:
        max_samples = None

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    json_files.sort()

    samples: List[Dict[str, Any]] = []
    num_skipped_no_img = 0

    for file in json_files:
        p = os.path.join(json_dir, file)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        sentences = data.get("text", [])
        image_path = p.replace(".json", ".jpg")
        image_file = file.replace(".json", ".jpg")
        if not os.path.exists(image_path):
            num_skipped_no_img += 1
            continue

        samples.append(
            {
                "image_path": image_path,
                "image_file": image_file,
                "sentences": sentences,
                "json_file": p,
            }
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    if rank == main_rank:
        print(
            f"[INFO] Loaded {len(samples)} samples from {json_dir}, "
            f"skipped {num_skipped_no_img} items due to missing images."
        )
    return samples


def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except Exception:
        return np.zeros(img.size[::-1], dtype=np.uint8), "", False

    inform = anno.get("shapes", [])
    is_sentence = anno.get("is_sentence", False)
    comments = anno.get("text", "")
    width, height = img.size
    area_list = []
    valid_poly_list = []
    for i in inform:
        if "flag" == i["label"].lower():
            continue
        points = i["points"]
        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        area_list.append(tmp_mask.sum())
        valid_poly_list.append(i)

    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for s_idx in list(sort_index):
        i = valid_poly_list[s_idx]
        label_value = 255 if "ignore" in i["label"].lower() else 1
        cv2.polylines(mask, np.array([i["points"]], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([i["points"]], dtype=np.int32), label_value)
    return mask, comments, is_sentence


# ============================================================
# Visualization utils
# ============================================================
def _get_font(size: int = 18):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
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


def draw_bbox_and_text(img: Image.Image, bbox_xyxy: Optional[List[int]], lines: List[str], bbox_width: int = 3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _get_font(18)

    if bbox_xyxy is not None:
        x0, y0, x1, y1 = bbox_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=bbox_width)

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


def draw_multi_bboxes_and_text(img: Image.Image, bboxes_xyxy: List[List[int]], lines: List[str], bbox_width: int = 5) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _get_font(18)

    for bb in bboxes_xyxy:
        if bb is None or len(bb) != 4:
            continue
        x0, y0, x1, y1 = bb
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=bbox_width)

    if lines:
        text = "\n".join(lines)
        try:
            bbt = draw.multiline_textbbox((0, 0), text, font=font)
            tw, th = bbt[2] - bbt[0], bbt[3] - bbt[1]
        except Exception:
            tw, th = (420, 80)

        pad = 6
        x, y = 8, 8
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
        draw.multiline_text((x, y), text, fill=(255, 255, 255), font=font)

    return out


def make_grid_2x3(imgs: List[Image.Image], cell_wh: Tuple[int, int]) -> Image.Image:
    """
    imgs length must be 6.
    """
    assert len(imgs) == 6
    cell_w, cell_h = cell_wh
    imgs_r = [im.convert("RGB").resize((cell_w, cell_h), Image.BILINEAR) for im in imgs]
    canvas = Image.new("RGB", (cell_w * 3, cell_h * 2), (0, 0, 0))
    # row 0
    canvas.paste(imgs_r[0], (0 * cell_w, 0))
    canvas.paste(imgs_r[1], (1 * cell_w, 0))
    canvas.paste(imgs_r[2], (2 * cell_w, 0))
    # row 1
    canvas.paste(imgs_r[3], (0 * cell_w, 1 * cell_h))
    canvas.paste(imgs_r[4], (1 * cell_w, 1 * cell_h))
    canvas.paste(imgs_r[5], (2 * cell_w, 1 * cell_h))
    return canvas


def gate_heatmap_with_colorbar_center05(
    gate_map_hw: np.ndarray,
    union_mask_hw: np.ndarray,
    out_size_wh: Tuple[int, int],
    title: str = "Gate heatmap",
    center: float = 0.5,
) -> Image.Image:
    """
    gate_map_hw: (Hg,Wg) in [0,1]
    union_mask_hw: (H,W) bool -> only show union; others set to 0
    out_size_wh: (W,H)
    Make 0.5 a clear boundary (TwoSlopeNorm + contour + legend).
    """
    import io
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    W, H = out_size_wh
    if gate_map_hw is None:
        return Image.new("RGB", (W, H), (0, 0, 0))

    gate_big = cv2.resize(gate_map_hw.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    if union_mask_hw is not None:
        gate_big = gate_big.copy()
        gate_big[~union_mask_hw.astype(bool)] = 0.0

    norm = TwoSlopeNorm(vmin=0.0, vcenter=center, vmax=1.0)

    fig = plt.figure(figsize=(W / 120.0, H / 120.0), dpi=120)
    ax = fig.add_subplot(111)
    im = ax.imshow(gate_big, norm=norm, cmap="seismic")

    # explicit boundary at 0.5
    try:
        ax.contour(gate_big, levels=[center], colors="black", linewidths=1.2)
    except Exception:
        pass

    ax.set_title(f"{title} (center=0.5)", fontsize=10)
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

    hm = Image.open(buf).convert("RGB").resize((W, H), Image.BILINEAR)
    return hm


# ============================================================
# SAM3 dual-branch + PixelGate fusion
# ============================================================
def is_empty_masks(x: Optional[torch.Tensor]) -> bool:
    if x is None:
        return True
    if not torch.is_tensor(x):
        return True
    return (x.numel() == 0) or (x.shape[0] == 0)


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


@torch.no_grad()
def run_branch_origin_text_bbox(img: Image.Image, processor, label: str, bbox_xywh_norm: List[float]) -> Dict[str, torch.Tensor]:
    """
    [NEW] ORIGIN baseline: apply BOTH prompts in SAM3 (text + bbox), then read final alloutput.
    """
    state = processor.set_image(img)
    if hasattr(processor, "reset_all_prompts"):
        processor.reset_all_prompts(state)

    # set text prompt first (may or may not return dict)
    if hasattr(processor, "set_text_prompt"):
        _ = processor.set_text_prompt(prompt=label, state=state)
    elif hasattr(processor, "set_text_prompt_alloutput"):
        _ = processor.set_text_prompt_alloutput(prompt=label, state=state)

    # then add bbox and fetch output
    if hasattr(processor, "add_geometric_prompt_alloutput"):
        out = processor.add_geometric_prompt_alloutput(box=bbox_xywh_norm, label=True, state=state)
        return out
    if hasattr(processor, "add_geometric_prompt"):
        out = processor.add_geometric_prompt(box=bbox_xywh_norm, label=True, state=state)
        return out if isinstance(out, dict) else state

    return state


def make_empty_mask(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W), dtype=bool)


def gate_mean_on_gt_foreground(gate_hw: np.ndarray, gt_mask_hw: np.ndarray) -> Optional[float]:
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
def sam3_predict_masks_pixel_gate_fusion_with_branches(
    img: Image.Image,
    label: str,
    bbox_xyxy_image: List[int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Returns:
      text_mask_np, bbox_mask_np, fused_mask_np, avg05_mask_np, info
    """
    info: Dict[str, Any] = {"branch_used": None, "gate_mean": None, "gate_map": None, "gate_hw": None}

    ok_sam = lazy_init_sam3()
    if not ok_sam:
        info["branch_used"] = "sam3_init_failed"
        return None, None, None, None, info
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
        return None, None, None, None, info

    text_mask_np = None
    bbox_mask_np = None
    sel_a = None
    sel_b = None

    if has_text:
        sel_a = select_masks_by_bbox_containment_scaled(out_text, bbox_xyxy_image_f, img_hw=(Himg, Wimg))
        text_mask_np = sel_a["best_mask"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
        text_mask_np = ensure_same_hw(text_mask_np, target_h=Himg, target_w=Wimg)

    if has_bbox:
        sel_b = select_masks_by_bbox_containment_scaled(out_bbox, bbox_xyxy_image_f, img_hw=(Himg, Wimg))
        bbox_mask_np = sel_b["best_mask"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
        bbox_mask_np = ensure_same_hw(bbox_mask_np, target_h=Himg, target_w=Wimg)

    # avg05 fusion (used by DGSeg)
    avg05_mask_np = None
    if has_text and has_bbox and (sel_a is not None) and (sel_b is not None):
        logits_a = sel_a["best_logits"].to(device)
        logits_b = sel_b["best_logits"].to(device)
        logits_a = sigmoid_to_logits(logits_a)
        logits_b = sigmoid_to_logits(logits_b)

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
        avg05_mask_np = bbox_mask_np if has_bbox else text_mask_np

    # PixelGate fused result (used by DGSeg)
    if (not USE_PIXEL_GATE_FUSION) or (not lazy_init_fusion_net()) or (_FUSION_NET is None):
        if has_bbox:
            info["branch_used"] = "bbox_only_no_fusion"
            return text_mask_np, bbox_mask_np, bbox_mask_np, avg05_mask_np, info
        info["branch_used"] = "text_only_no_fusion"
        return text_mask_np, bbox_mask_np, text_mask_np, avg05_mask_np, info

    if not has_text and has_bbox:
        info["branch_used"] = "bbox_only"
        return None, bbox_mask_np, bbox_mask_np, avg05_mask_np, info
    if has_text and not has_bbox:
        info["branch_used"] = "text_only"
        return text_mask_np, None, text_mask_np, avg05_mask_np, info

    pixel_a = out_text.get("pixel_embed", None)
    pixel_b = out_bbox.get("pixel_embed", None)
    if pixel_a is None or pixel_b is None:
        info["branch_used"] = "bbox_only_no_pixel_embed"
        return text_mask_np, bbox_mask_np, bbox_mask_np, avg05_mask_np, info

    assert sel_a is not None and sel_b is not None

    logits_a = sel_a["best_logits"].to(device)
    logits_b = sel_b["best_logits"].to(device)
    logits_a = sigmoid_to_logits(logits_a)
    logits_b = sigmoid_to_logits(logits_b)

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

    try:
        info["_logits_a"] = logits_a.detach().float().cpu()
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
        gate_map = gate.squeeze(0).squeeze(0).detach().float().cpu().numpy()
        info["gate_map"] = gate_map
        info["gate_hw"] = [int(gate_map.shape[0]), int(gate_map.shape[1])]

    return text_mask_np, bbox_mask_np, fused_mask, avg05_mask_np, info


@torch.no_grad()
def sam3_predict_mask_origin_text_bbox(
    img: Image.Image,
    label: str,
    bbox_xyxy_image: List[int],
) -> Optional[np.ndarray]:
    """
    [NEW] ORIGIN: SAM3 with both prompts simultaneously -> select mask by same containment rule.
    Return mask at image (H,W) bool.
    """
    ok_sam = lazy_init_sam3()
    if not ok_sam:
        return None
    assert _SAM3_PROCESSOR is not None

    Himg, Wimg = img.size[1], img.size[0]
    bbox_xywh_norm = xyxy_to_xywh_norm(bbox_xyxy_image, W=Wimg, H=Himg)
    bbox_xyxy_image_f = [float(v) for v in bbox_xyxy_image]

    out = run_branch_origin_text_bbox(img, _SAM3_PROCESSOR, label, bbox_xywh_norm)
    if not (isinstance(out, dict) and ("masks" in out) and (not is_empty_masks(out["masks"]))):
        return None

    sel = select_masks_by_bbox_containment_scaled(out, bbox_xyxy_image_f, img_hw=(Himg, Wimg))
    m = sel["best_mask"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
    m = ensure_same_hw(m, target_h=Himg, target_w=Wimg)
    return m


# ============================================================
# MLLM prompt templates
# ============================================================
model = None
processor = None

QUESTION_TEMPLATE = (
    "{Question} First output the thinking process in <think> </think> tags and then output the final answer in "
    "<answer> </answer> tags. Output the final answer in JSON format."
)
problem_template = "Please provide the bounding box coordinate and the label of the region this sentence describes: <sentence>"


# ============================================================
# Main
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


def main():
    global model, processor
    if rank == main_rank:
        print(f"[INFO] Evaluating ReasonSeg folder: {REASONSEG_JSON_DIR}")

    if model is None or processor is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": local_rank} if torch.cuda.is_available() else "cpu",
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)

    samples = load_reasonseg_json(REASONSEG_JSON_DIR, max_samples=MAX_SAMPLES)
    total_items = len(samples)

    if DEBUG_MODE and os.path.exists(TARGET_JSON_FILE):
        if rank != main_rank:
            dist.barrier()
            return

        print(f"[INFO] Loading cached MLLM output from: {TARGET_JSON_FILE}")
        cached = read_jsonl(TARGET_JSON_FILE)
        if len(cached) < total_items:
            print(f"[WARN] cached outputs ({len(cached)}) < samples ({total_items}). Will evaluate min(len).")
        n_eval = min(len(cached), total_items)

        meta_all = []
        for s in samples[:n_eval]:
            sent = ""
            texts = s.get("sentences", [])
            if isinstance(texts, list) and len(texts) > 0:
                sent = texts[0]
            elif isinstance(texts, str):
                sent = texts
            meta_all.append(
                {
                    "image_path": s["image_path"],
                    "image_file": s.get("image_file", ""),
                    "json_file": s.get("json_file", ""),
                    "sentence": sent,
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

    # DDP MLLM inference (all ranks)
    per_rank = total_items // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else total_items
    my_indices = list(range(start_idx, end_idx))
    my_samples = [samples[i] for i in my_indices]

    messages = []
    meta_rank = []
    for gi, s in zip(my_indices, my_samples):
        image_path = s["image_path"]
        texts = s.get("sentences", [])
        if isinstance(texts, list) and len(texts) > 0:
            sent = texts[0]
        else:
            sent = texts if isinstance(texts, str) else ""
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
                "image_file": s.get("image_file", ""),
                "json_file": s.get("json_file", ""),
                "sentence": sent,
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
            img_ = Image.open(img_path).convert("RGB")
            img_w, img_h = img_.size

            rank_items.append(
                {
                    "global_idx": int(batch_meta[j]["global_idx"]),
                    "meta": batch_meta[j],
                    "model_out": [output_text, input_h, input_w, img_h, img_w],
                }
            )

    if DEBUG_MODE:
        os.makedirs(os.path.dirname(TARGET_JSON_FILE) or ".", exist_ok=True)
        with open(f"{TARGET_JSON_FILE}.rank{rank}.jsonl", "w", encoding="utf-8") as f:
            for it in rank_items:
                f.write(json.dumps(it["model_out"], ensure_ascii=False) + "\n")

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

        if DEBUG_MODE:
            with open(TARGET_JSON_FILE, "w", encoding="utf-8") as f:
                for mo in outputs_all:
                    f.write(json.dumps(mo, ensure_ascii=False) + "\n")
            print(f"[INFO] Wrote merged MLLM outputs to: {TARGET_JSON_FILE}")

        evaluate_and_save(meta_all, outputs_all, len(outputs_all))

    dist.barrier()


# ============================================================
# EVAL + 2x3 VIS
# ============================================================
def evaluate_and_save(meta_all: List[Dict[str, Any]], outputs_all: List[Any], n_eval: int):
    assert rank == main_rank

    os.makedirs(VIS_DIR, exist_ok=True)
    _ = lazy_init_sam3()
    _ = lazy_init_fusion_net()

    final_output = []

    # existing metrics (merged)
    iou_list, intersection_list, union_list = [], [], []
    success_05 = 0

    iou_text_list, inter_text_list, union_text_list = [], [], []
    iou_bbox_list, inter_bbox_list, union_bbox_list = [], [], []

    iou_avg_list, inter_avg_list, union_avg_list = [], [], []
    success_05_avg = 0

    iou_oracle_list, inter_oracle_list, union_oracle_list = [], [], []
    success_05_oracle = 0

    # [NEW] origin baseline metrics
    iou_origin_list, inter_origin_list, union_origin_list = [], [], []
    success_05_origin = 0

    # gate analysis
    gate_fg_list = []
    iou_diff_list = []
    gate_valid_cnt = 0

    vis_saved = 0

    for idx in tqdm(range(n_eval), total=n_eval):
        s = meta_all[idx]
        output_text, input_h, input_w, _, _ = outputs_all[idx]

        image_path = s["image_path"]
        query_text = s.get("sentence", "")

        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        if MAX_IMAGE_SIDE is not None:
            w, h = img.size
            scale = min(MAX_IMAGE_SIDE / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        W, H = img.size

        gt_mask_np, _, _ = get_mask_from_json(s.get("json_file", None), img)
        gt_mask = (gt_mask_np == 1).astype(bool)

        # -------- parse instances --------
        instances = extract_instances_from_output(output_text)
        if len(instances) == 0:
            bbox_in = extract_bbox_answer(output_text)
            label_1 = extract_label_answer(output_text)
            instances = [{"bbox_2d": bbox_in, "label": label_1}]

        # merged predictions (OR over instances)
        text_pred = make_empty_mask(H, W)
        bbox_pred = make_empty_mask(H, W)
        fused_pred = make_empty_mask(H, W)
        avg05_pred = make_empty_mask(H, W)
        origin_pred = make_empty_mask(H, W)  # [NEW]

        # record for visualization
        pred_bboxes_xyxy_img: List[List[int]] = []
        used_labels: List[str] = []

        # oracle-best1 (keep your logic)
        oracle_pred = make_empty_mask(H, W)
        # (optional) debug: record per-instance weights
        oracle_inst_weights = [] 

        # gate fg mean: average across instances that have gate_map
        gate_fg_means_inst = []
        gate_map_for_vis = None  # keep last available gate_map (or you can choose max-area)
        for inst_id, inst in enumerate(instances):
            bbox_in = inst.get("bbox_2d", [0, 0, 0, 0])
            label_i = inst.get("label", None)

            pred_bbox_img_f = resize_bbox_xyxy(bbox_in, input_h, input_w, H, W)
            pred_bbox_xyxy = sanitize_xyxy_int(pred_bbox_img_f, width=W, height=H)
            if pred_bbox_xyxy is None:
                continue

            use_label = label_i if (label_i is not None and len(str(label_i).strip()) > 0) else query_text

            # ---- your adaptive fusion pipeline (unchanged) ----
            tmask, bmask, fmask, amask, fusion_info = sam3_predict_masks_pixel_gate_fusion_with_branches(
                img=img,
                label=use_label,
                bbox_xyxy_image=pred_bbox_xyxy,
            )

            om = sam3_predict_mask_origin_text_bbox(img=img, label=use_label, bbox_xyxy_image=pred_bbox_xyxy)
            if om is not None:
                origin_pred = ensure_same_hw(om, target_h=H, target_w=W)

            if tmask is not None:
                text_pred |= ensure_same_hw(tmask, target_h=H, target_w=W)
            if bmask is not None:
                bbox_pred |= ensure_same_hw(bmask, target_h=H, target_w=W)
            if fmask is not None:
                fused_pred |= ensure_same_hw(fmask, target_h=H, target_w=W)
            if amask is not None:
                avg05_pred |= ensure_same_hw(amask, target_h=H, target_w=W)
            if origin_pred is not None:
                origin_pred |= ensure_same_hw(origin_pred, target_h=H, target_w=W)

            # ---- [NEW] origin baseline: both prompts in SAM3 ----
            oracle_inst_mask = None

            # 1) instance-level IoUs (use instance masks, not merged masks)
            t_i = ensure_same_hw(tmask, target_h=H, target_w=W) if tmask is not None else make_empty_mask(H, W)
            b_i = ensure_same_hw(bmask, target_h=H, target_w=W) if bmask is not None else make_empty_mask(H, W)
            iou_t_i, _, _ = mask_iou(t_i, gt_mask)
            iou_b_i, _, _ = mask_iou(b_i, gt_mask)

            # 2) weighted logits oracle (requires cached processed logits)
            la = fusion_info.get("_logits_a", None) if isinstance(fusion_info, dict) else None  # torch CPU (1,1,Hm,Wm)
            lb = fusion_info.get("_logits_b", None) if isinstance(fusion_info, dict) else None

            if (la is not None) and (lb is not None):
                # softmax on IoUs (temperature via scale)
                # NOTE: your softmax2(a,b,scale) does softmax([a,b]*scale)
                w_t, w_b = softmax2(float(iou_t_i), float(iou_b_i), scale=ORACLE_SOFTMAX_SCALE)  # e.g. 100.0

                oracle_inst_weights.append({
                    "inst": int(inst_id),
                    "iou_t": float(iou_t_i),
                    "iou_b": float(iou_b_i),
                    "w_t": float(w_t),
                    "w_b": float(w_b),
                })

                # weighted sum logits -> mask at logits resolution
                oracle_logits = (w_t * la + w_b * lb)  # torch CPU
                oracle_prob = torch.sigmoid(oracle_logits)
                oracle_small = (oracle_prob > 0.5).squeeze(0).squeeze(0).numpy().astype(bool)  # (Hm,Wm)

                # resize to image resolution and merge
                oracle_inst_mask = ensure_same_hw(oracle_small, target_h=H, target_w=W)

            else:
                # fallback: choose better instance branch mask (if logits not available)
                if tmask is None and bmask is None:
                    oracle_inst_mask = None
                elif tmask is None:
                    oracle_inst_mask = b_i
                elif bmask is None:
                    oracle_inst_mask = t_i
                else:
                    oracle_inst_mask = t_i if (iou_t_i >= iou_b_i) else b_i

            if oracle_inst_mask is not None:
                oracle_pred |= oracle_inst_mask

            # gate fg mean
            if isinstance(fusion_info, dict) and fusion_info.get("gate_map", None) is not None:
                gate_map = fusion_info["gate_map"]
                gate_map_for_vis = gate_map
                gfg = gate_mean_on_gt_foreground(gate_map, gt_mask)
                if gfg is not None:
                    gate_fg_means_inst.append(float(gfg))

        # ---- IoUs (merged) ----
        iou_t, inter_t, uni_t = mask_iou(text_pred, gt_mask)
        iou_b, inter_b, uni_b = mask_iou(bbox_pred, gt_mask)
        iou_f, inter_f, uni_f = mask_iou(fused_pred, gt_mask)
        iou_a, inter_a, uni_a = mask_iou(avg05_pred, gt_mask)
        iou_o, inter_o, uni_o = mask_iou(oracle_pred, gt_mask)

        iou_org, inter_org, uni_org = mask_iou(origin_pred, gt_mask)  # [NEW]

        iou_text_list.append(iou_t); inter_text_list.append(inter_t); union_text_list.append(uni_t)
        iou_bbox_list.append(iou_b); inter_bbox_list.append(inter_b); union_bbox_list.append(uni_b)

        iou_list.append(iou_f); intersection_list.append(inter_f); union_list.append(uni_f)
        if iou_f > 0.5:
            success_05 += 1

        iou_avg_list.append(iou_a); inter_avg_list.append(inter_a); union_avg_list.append(uni_a)
        if iou_a > 0.5:
            success_05_avg += 1

        iou_oracle_list.append(iou_o); inter_oracle_list.append(inter_o); union_oracle_list.append(uni_o)
        if iou_o > 0.5:
            success_05_oracle += 1

        # [NEW] origin metrics
        iou_origin_list.append(iou_org); inter_origin_list.append(inter_org); union_origin_list.append(uni_org)
        if iou_org > 0.5:
            success_05_origin += 1

        # gate fg mean aggregated
        gate_fg_mean = float(np.mean(gate_fg_means_inst)) if len(gate_fg_means_inst) > 0 else None
        if gate_fg_mean is not None:
            gate_valid_cnt += 1
            gate_fg_list.append(gate_fg_mean)
            iou_diff_list.append(iou_t - iou_b)

        # -------- 2x3 visualization --------
        do_save = True
        if VIS_MAX_SAVE is not None and vis_saved >= int(VIS_MAX_SAVE):
            do_save = False
        
        if do_save:
            # pick a standard cell size (avoid too huge)
            cell_w = min(720, W)
            cell_h = int(cell_w * (H / max(1.0, W)))
            cell_h = min(cell_h, 720)
            if cell_h <= 0:
                cell_h = min(720, H)
                cell_w = int(cell_h * (W / max(1.0, H)))

            # --- panels ---
            # Text
            img_text = overlay_mask_on_image(img, text_pred, alpha=VIS_ALPHA)
            label_show = used_labels[0] if len(used_labels) > 0 else ""
            if len(used_labels) > 1:
                label_show = f"{used_labels[0]} (+{len(used_labels)-1})"
            img_text = draw_bbox_and_text(
                img_text,
                bbox_xyxy=None,
                lines=[f"Text (merged)  IoU={iou_t:.3f}", f"label: {label_show}"],
                bbox_width=0,
            )

            # BBox (draw all bboxes thick)
            img_bbox = overlay_mask_on_image(img, bbox_pred, alpha=VIS_ALPHA)
            img_bbox = draw_multi_bboxes_and_text(
                img_bbox,
                pred_bboxes_xyxy_img,
                lines=[f"BBox (merged)  IoU={iou_b:.3f}", f"boxes: {len(pred_bboxes_xyxy_img)}"],
                bbox_width=5,
            )

            # Origin
            img_origin = overlay_mask_on_image(img, origin_pred, alpha=VIS_ALPHA)
            img_origin = draw_bbox_and_text(
                img_origin,
                bbox_xyxy=None,
                lines=[f"Origin(text+bbox)  IoU={iou_org:.3f}"],
                bbox_width=0,
            )

            # GT
            img_gt = overlay_mask_on_image(img, gt_mask, alpha=VIS_ALPHA)
            qshow = query_text if isinstance(query_text, str) else str(query_text)
            if len(qshow) > 80:
                qshow = qshow[:77] + "..."
            img_gt = draw_bbox_and_text(
                img_gt,
                bbox_xyxy=None,
                lines=[f"GT", f"query: {qshow}"],
                bbox_width=0,
            )

            # Fused
            extra = f"gate_fg={gate_fg_mean:.3f}" if gate_fg_mean is not None else "gate_fg=None"
            img_fused = overlay_mask_on_image(img, fused_pred, alpha=VIS_ALPHA)
            img_fused = draw_bbox_and_text(
                img_fused,
                bbox_xyxy=None,
                lines=[f"Fused (merged)  IoU={iou_f:.3f}", extra, f"avg05 IoU={iou_a:.3f}", f"oracle(best1) IoU={iou_o:.3f}"],
                bbox_width=0,
            )

            # Gate heatmap (masked by union of GT/Text/BBox)
            union_mask = (gt_mask | text_pred | bbox_pred).astype(bool)
            gate_img = None
            if gate_map_for_vis is not None:
                gate_img = gate_heatmap_with_colorbar_center05(
                    gate_map_hw=gate_map_for_vis,
                    union_mask_hw=union_mask,
                    out_size_wh=(W, H),
                    title="Gate heatmap (masked)",
                    center=0.5,
                )
            else:
                gate_img = Image.new("RGB", (W, H), (0, 0, 0))
                gate_img = draw_bbox_and_text(gate_img, None, ["Gate heatmap", "None"], bbox_width=0)

            # 2x3: [Text, BBox, Origin; GT, Fused, Gate]
            grid = make_grid_2x3(
                [img_text, img_bbox, img_origin, img_gt, img_fused, gate_img],
                cell_wh=(cell_w, cell_h),
            )

            base = os.path.splitext(os.path.basename(image_path))[0]
            out_png = os.path.join(
                VIS_DIR,
                f"{idx:06d}_{base}_f{iou_f:.3f}_org{iou_org:.3f}_t{iou_t:.3f}_b{iou_b:.3f}.png"
            )
            grid.save(out_png)
            vis_saved += 1

        final_output.append(
            {
                "idx": idx,
                "global_idx": s.get("global_idx", idx),
                "image_file": s.get("image_file", ""),
                "image_path": image_path,
                "sentence": query_text,
                "model_output": output_text,

                "num_instances_parsed": int(len(instances)),
                "num_instances_valid": int(len(pred_bboxes_xyxy_img)),
                "used_labels": used_labels,
                "pred_bboxes_img_xyxy": pred_bboxes_xyxy_img,

                "iou_text": float(iou_t),
                "iou_bbox": float(iou_b),
                "iou_fused": float(iou_f),
                "iou_avg05": float(iou_a),

                "iou_origin": float(iou_org),
                "iou_oracle": float(iou_o),
                "oracle_weights": oracle_inst_weights,   # Optional; remove if the output becomes too large.

                "gate_fg_mean": (float(gate_fg_mean) if gate_fg_mean is not None else None),

                "inter_fused": float(inter_f),
                "union_fused": float(uni_f),
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

    giou_org, ciou_org = compute_giou_ciou(iou_origin_list, inter_origin_list, union_origin_list)
    acc_05_org = float(success_05_origin / max(1, len(iou_origin_list)))

    # gate relation (keep)
    gate_analysis = {
        "num_gate_valid": int(gate_valid_cnt),
        "pearson_corr_gatefg_vs_iou_diff": None,
        "agreement_rate": None,
        "mean_gatefg_when_text_better": None,
        "mean_gatefg_when_bbox_better": None,
    }
    if gate_valid_cnt > 1:
        g = np.array(gate_fg_list, dtype=np.float32)
        d = np.array(iou_diff_list, dtype=np.float32)
        if float(g.std()) > 1e-8 and float(d.std()) > 1e-8:
            gate_analysis["pearson_corr_gatefg_vs_iou_diff"] = float(np.corrcoef(g, d)[0, 1])

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
        f"  Origin(text+bbox): gIoU={giou_org:.4f}, cIoU={ciou_org:.4f}, IoU>0.5={acc_05_org*100:.2f}%\n"
        f"  Text(merged)     : gIoU={giou_t:.4f}, cIoU={ciou_t:.4f}\n"
        f"  BBox(merged)     : gIoU={giou_b:.4f}, cIoU={ciou_b:.4f}\n"
        f"  Avg05(merged)    : gIoU={giou_a:.4f}, cIoU={ciou_a:.4f}, IoU>0.5={acc_05_a*100:.2f}%\n"
        f"  Gate(merged)     : gIoU={giou_f:.4f}, cIoU={ciou_f:.4f}, IoU>0.5={acc_05*100:.2f}%\n"
        f"  Oracle(best1)    : gIoU={giou_o:.4f}, cIoU={ciou_o:.4f}, IoU>0.5={acc_05_o*100:.2f}%\n"
        f"  GateFG analysis  : valid={gate_analysis['num_gate_valid']}, pearson={gate_analysis['pearson_corr_gatefg_vs_iou_diff']}, agree={gate_analysis['agreement_rate']}\n"
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
                "reasonseg_json_dir": REASONSEG_JSON_DIR,
                "num_samples": len(iou_list),
                "metrics": {
                    "origin_text_bbox": {"gIoU": giou_org, "cIoU": ciou_org, "acc_iou_gt_0.5": acc_05_org},
                    "text": {"gIoU": giou_t, "cIoU": ciou_t},
                    "bbox": {"gIoU": giou_b, "cIoU": ciou_b},
                    "avg05": {"gIoU": giou_a, "cIoU": ciou_a, "acc_iou_gt_0.5": acc_05_a},
                    "pixel_gate": {
                        "gIoU": giou_f,
                        "cIoU": ciou_f,
                        "acc_iou_gt_0.5": acc_05,
                        "sum_intersection": float(sum(intersection_list)),
                        "sum_union": float(sum(union_list)),
                        "cIoU_check": float(sum(intersection_list) / max(1e-6, sum(union_list))),
                    },
                    "oracle_best1": {"gIoU": giou_o, "cIoU": ciou_o, "acc_iou_gt_0.5": acc_05_o},
                },
                "gate_analysis": gate_analysis,
                "vis_dir": VIS_DIR,
                "vis_saved": vis_saved,
                "fusion_ckpt": FUSION_CKPT if USE_PIXEL_GATE_FUSION else None,
                "results": final_output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Saved results to {OUTPUT_PATH}")
    print(f"[INFO] Visualization saved to {VIS_DIR}")
    print("-" * 100)


if __name__ == "__main__":
    main()
