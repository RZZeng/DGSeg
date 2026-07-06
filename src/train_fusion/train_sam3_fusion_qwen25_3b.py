# train_sam3_fusion.py
# -*- coding: utf-8 -*-

import os
import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import time
import tempfile
import numpy as np
from PIL import Image, ImageOps
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


# ----------------------------
# Config
# ----------------------------
SAM3_REPO_PATH = os.environ.get("SAM3_REPO_PATH", os.path.join(os.environ.get("DGSEG_ROOT", "."), "sam3"))
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", os.path.join(os.environ.get("MODEL_ROOT", "./models"), "sam3.pt"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

IMAGE_ROOT = os.environ.get(
    "IMAGE_ROOT",
    os.path.join(os.environ.get("DATA_ROOT", "./data"), "refer_seg/images/mscoco/images/train2014"),
)
MLLM_JSON_FILE = os.environ.get(
    "MLLM_JSON_FILE",
    os.path.join(os.environ.get("DGSEG_ROOT", "."), "data", "refcocog_train_predictions_3b.jsonl"),
)
JSON_FILE_PATH = os.environ.get(
    "JSON_FILE_PATH",
    os.path.join(os.environ.get("DATA_ROOT", "./data"), "refcocog_train_dataset.json"),
)

LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
NUM_EPOCHS = 5
NUM_WORKERS = 2
LOG_EVERY = 1
ACCUM_STEPS = 4

DETACH_BRANCH_OUTPUTS = True   # keep True to avoid back-propagating into SAM3
GATE_DROPOUT = 0.1
GATE_HIDDEN = 64

SAVE_INTERN = True


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_script_filename(default: str = "train_sam3_fusion.py") -> str:
    # Some interactive environments do not define __file__
    return os.path.basename(globals().get("__file__", default))

def save_checkpoint_atomic(payload: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write checkpoints atomically through a temporary file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(path))
    os.close(tmp_fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def make_rng_state() -> Dict[str, Any]:
    st = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["torch_cuda_random_all"] = torch.cuda.get_rng_state_all()
    return st


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


def sigmoid_to_logits(sigmoid: torch.Tensor) -> torch.Tensor:
    sigmoid = sigmoid.clamp(1e-6, 1 - 1e-6)
    return torch.log(sigmoid / (1 - sigmoid))


def load_coco_polygon_mask(segmentation: Any, image_w: int, image_h: int) -> Optional[torch.Tensor]:
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        return None
    if segmentation is None:
        return None
    try:
        if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
            m = mask_utils.decode(segmentation)
        else:
            rle = mask_utils.frPyObjects(segmentation, image_h, image_w)
            m = mask_utils.decode(rle)
    except Exception:
        return None

    if m is None:
        return None
    if getattr(m, "ndim", 0) == 3:
        m = (m > 0).any(axis=2)
    m = (m > 0)
    return torch.as_tensor(m, dtype=torch.bool, device="cpu")


def resize_mask_to(mask, H: int, W: int) -> torch.Tensor:
    # mask: (h,w) bool
    t = mask[None, None, ...].to(torch.float32)  # (1,1,h,w)
    # The mask is expected to match the output resolution.
    return t


def bbox_mask_from_xywh_norm(bbox_image: List[float], H: int, W: int, device=None) -> torch.Tensor:
    x1, y1, x2, y2 = bbox_image
    m = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)
    m[:, :, y1:y2, x1:x2] = True
    return m


def mask_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred: (B,N,1,H,W) bool OR (B,1,H,W) bool OR (N,1,H,W)
    gt:   (B,1,1,H,W) bool OR (B,1,H,W) bool OR (1,1,H,W)
    return iou: broadcastable result
    """
    pred = pred.bool()
    gt = gt.bool()
    # Normalize to (..., H, W).
    if pred.dim() == 4:  # (B,1,H,W)
        pred_ = pred.unsqueeze(1)  # (B,1,1,H,W)
    elif pred.dim() == 3:  # (1, H, W) or (H, W), rarely used.
        pred_ = pred.view(1, 1, 1, *pred.shape[-2:])
    else:
        pred_ = pred  # assume (B,N,1,H,W)

    if gt.dim() == 4:   # (B,1,H,W)
        gt_ = gt.unsqueeze(1)      # (B,1,1,H,W)
    elif gt.dim() == 3:
        gt_ = gt.view(1, 1, 1, *gt.shape[-2:])
    else:
        gt_ = gt

    inter = (pred_ & gt_).flatten(-2).sum(dim=-1).squeeze(-1)
    union = (pred_ | gt_).flatten(-2).sum(dim=-1).squeeze(-1)
    return inter.float() / (union.float() + eps)


def select_best_mask_by_bbox_iou(out: Dict[str, torch.Tensor], bbox_image: List[float]) -> Dict[str, torch.Tensor]:
    assert "masks" in out and "masks_logits" in out, "out must contain 'masks' and 'masks_logits'"
    masks = out["masks"].to(torch.bool)        # (N,1,H,W)
    logits = out["masks_logits"]               # (N,1,H,W)

    N, _, H, W = masks.shape
    device = logits.device
    bbox_image = [int(item) for item in bbox_image]
    bbox_m = bbox_mask_from_xywh_norm(bbox_image, H=H, W=W, device=device)  # (1,1,H,W)

    ious = mask_iou(masks, bbox_m)   # Shape can be (N,) or (1, N), depending on broadcasting.
    # Normalize to (N,).
    ious_flat = ious.view(-1)
    best_idx = torch.argmax(ious_flat).item()

    best_mask = masks[best_idx:best_idx+1]    # (1,1,H,W)
    best_logits = logits[best_idx:best_idx+1] # (1,1,H,W)
    best_iou = ious_flat[best_idx:best_idx+1] # (1,)

    return {
        "best_mask": best_mask,
        "best_logits": best_logits,
        "best_idx": torch.tensor([best_idx], device=device, dtype=torch.long),
        "best_iou": best_iou
    }


def is_empty_masks(x: Optional[torch.Tensor]) -> bool:
    if x is None:
        return True
    if not torch.is_tensor(x):
        return True
    return (x.numel() == 0) or (x.shape[0] == 0)


def tensor_stats(x: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    """
    Return summary statistics after flattening the tensor. Empty tensors are handled explicitly.
    """
    if x is None:
        return {f"{prefix}empty": 1.0}
    x = x.detach()
    if x.numel() == 0:
        return {f"{prefix}empty": 1.0}
    xf = x.float().reshape(-1)
    # torch.quantile is available in modern PyTorch releases.
    q = torch.quantile(xf, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=xf.device))
    return {
        f"{prefix}mean": xf.mean().item(),
        f"{prefix}std": xf.std(unbiased=False).item(),
        f"{prefix}min": xf.min().item(),
        f"{prefix}max": xf.max().item(),
        f"{prefix}q10": q[0].item(),
        f"{prefix}q25": q[1].item(),
        f"{prefix}median": q[2].item(),
        f"{prefix}q75": q[3].item(),
        f"{prefix}q90": q[4].item(),
    }


def safe_scalar(x: torch.Tensor) -> float:
    """Convert a tensor to a scalar; mean is used for non-scalar tensors."""
    if x is None:
        return float("nan")
    if not torch.is_tensor(x):
        return float(x)
    if x.numel() == 1:
        return x.item()
    return x.float().mean().item()


# ----------------------------
# Fusion model
# ----------------------------
class PixelGateMaskLogitsFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        gn_groups: int = 8,
        gate_temperature: float = 1.0,
        clamp_gate: Optional[Tuple[float, float]] = (0.01, 0.99),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.gate_temperature = gate_temperature
        self.clamp_gate = clamp_gate

        cin = 2 * in_channels
        self.conv1 = nn.Conv2d(cin, hidden_channels, 3, padding=1)
        g = min(gn_groups, hidden_channels)
        while hidden_channels % g != 0 and g > 1:
            g -= 1
        self.norm1 = nn.GroupNorm(g, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, 1, 3, padding=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, pixel_a, pixel_b, logits_a, logits_b, return_gate: bool = False):
        if logits_a.dim() == 3:
            logits_a = logits_a.unsqueeze(1)
        if logits_b.dim() == 3:
            logits_b = logits_b.unsqueeze(1)

        assert pixel_a.shape == pixel_b.shape
        B, C, H, W = pixel_a.shape
        assert C == self.in_channels

        logH, logW = logits_a.shape[-2:]

        x = torch.cat([pixel_a, pixel_b], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        gate_logits = self.conv2(x)

        gate = torch.sigmoid(gate_logits / max(self.gate_temperature, 1e-6))
        if self.clamp_gate is not None:
            lo, hi = self.clamp_gate
            gate = gate.clamp(lo, hi)

        if (logH, logW) != (H, W):
            gate = F.interpolate(gate, size=(logH, logW), mode="bilinear", align_corners=False)
            gate_logits = F.interpolate(gate_logits, size=(logH, logW), mode="bilinear", align_corners=False)

        fused = gate * logits_a + (1.0 - gate) * logits_b
        if return_gate:
            return fused, gate, gate_logits
        return fused


# ----------------------------
# Dataset
# ----------------------------
class RefCOCOFromJsonlDataset(Dataset):
    def __init__(self, ann_json_path: str, image_root: str, mllm_jsonl_path: str, max_items: int = -1):
        self.image_root = image_root
        self.samples = self._load(ann_json_path, mllm_jsonl_path, max_items)

    def _load(self, ann_json_path: str, mllm_jsonl_path: str, max_items: int):
        mllm_outputs = []
        with open(mllm_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                mllm_outputs.append(json.loads(line))

        with open(ann_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        num_sample = 0
        for item in data:
            sentences = item.get("sentences", [])
            if len(sentences) == 0:
                continue
            sent = sentences[0]

            image_file = item.get("image_file", "")
            image_path = os.path.join(self.image_root, image_file)
            if not os.path.exists(image_path):
                continue

            segmentation = item.get("segmentation", None)

            mo = mllm_outputs[num_sample]
            if isinstance(mo, (list, tuple)) and len(mo) >= 5:
                output_text, input_h, input_w, img_h, img_w = mo[:5]
            elif isinstance(mo, dict):
                output_text = mo.get("output_text", mo.get("model_output", ""))
                input_h = mo.get("input_h", mo.get("input_height", 1))
                input_w = mo.get("input_w", mo.get("input_width", 1))
                img_h = mo.get("img_h", mo.get("img_height", 1))
                img_w = mo.get("img_w", mo.get("img_width", 1))
            else:
                num_sample += 1
                continue

            bbox_xyxy = extract_bbox_answer(output_text)
            label = extract_label_answer(output_text)

            x1, y1, x2, y2 = bbox_xyxy
            if input_w <= 0 or input_h <= 0:
                num_sample += 1
                continue

            x1n = float(x1 / input_w)
            y1n = float(y1 / input_h)
            x2n = float(x2 / input_w)
            y2n = float(y2 / input_h)
            x1n, y1n, x2n, y2n = [max(0.0, min(1.0, v)) for v in [x1n, y1n, x2n, y2n]]
            w, h = max(0.0, x2n - x1n), max(0.0, y2n - y1n)

            bbox_xywh_norm = [x1n + 0.5 * w, y1n + 0.5 * h, w, h]
            bbox_image = [x1n * img_w, y1n * img_h, x2n * img_w, y2n * img_h]

            samples.append({
                "image_path": image_path,
                "query": sent,
                "label": label if label is not None else sent,
                "bbox_xywh_norm": bbox_xywh_norm,
                "bbox_image": bbox_image,
                "segmentation": segmentation,
            })

            num_sample += 1
            if num_sample >= len(mllm_outputs):
                break
            if max_items is not None and max_items > 0 and len(samples) >= max_items:
                break

        print(f"[INFO] Loaded {len(samples)} samples from {ann_json_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ----------------------------
# Loss
# ----------------------------
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)

def teacher_weight(step: int,
                   warmup_steps: int = 0,
                   total_steps: int = 20000,
                   init_w: float = 0.5,
                   final_w: float = 0.0,
                   mode: str = "cosine") -> float:
    if step < warmup_steps:
        return init_w
    t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    if mode == "linear":
        w = init_w * (1.0 - t) + final_w * t
    elif mode == "exp":
        if final_w <= 0:
            final_w = 1e-6
        k = math.log(init_w / final_w)
        w = init_w * math.exp(-k * t)
    else:
        w = final_w + 0.5 * (init_w - final_w) * (1.0 + math.cos(math.pi * t))
    return float(w)

def gate_teacher_loss_dense(
    gate_logits: torch.Tensor,   # (B,1,H,W)
    iou_a: torch.Tensor,         # scalar tensor or (B,)
    iou_b: torch.Tensor,         # scalar tensor or (B,)
) -> torch.Tensor:
    B, _, H, W = gate_logits.shape
    T = 100.0
    ia = iou_a.view(B)
    ib = iou_b.view(B)
    w = torch.softmax(torch.stack([ia, ib], dim=-1) * T, dim=-1)[..., 0]  # (B,)
    tm = w.view(B, 1, 1, 1).expand_as(gate_logits)
    teacher_loss = F.binary_cross_entropy_with_logits(gate_logits, tm)
    return teacher_loss


# ----------------------------
# Run branches (frozen SAM3)
# ----------------------------
@torch.no_grad()
def run_branch_text(img: Image.Image, processor: Sam3Processor, sam3_model, label: str) -> Dict[str, torch.Tensor]:
    state = processor.set_image(img)
    out = processor.set_text_prompt_alloutput(prompt=label, state=state)
    return out

@torch.no_grad()
def run_branch_bbox(img: Image.Image, processor: Sam3Processor, sam3_model, bbox_xywh_norm: List[float]) -> Dict[str, torch.Tensor]:
    state = processor.set_image(img)
    processor.reset_all_prompts(state)
    out = processor.add_geometric_prompt_alloutput(box=bbox_xywh_norm, label=True, state=state)
    return out


def freeze_module(m: nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

def get_joint_spatial_weight(logits_a, logits_b, fg_weight=1.2, bg_weight=0.2):
    """
    Build a spatial weight map that emphasizes foreground regions.
    """
    # Get predictions from the two branches, shaped as (B, 1, H, W).
    pred_a = (logits_a > 0)
    pred_b = (logits_b > 0)
    
    # Union mask: a pixel is active if either branch or the GT marks it as foreground.
    # This keeps gradients on missed foreground regions.
    interest_mask = (pred_a | pred_b).float()
    
    # Map weights to [bg_weight, fg_weight].
    # Example: foreground weight 10 and background weight 1.
    weight_map = interest_mask * (fg_weight - bg_weight) + bg_weight
    
    return weight_map, interest_mask

def compute_all_losses(
    fused_logits, gate, gate_logits, targets, 
    logits_a, logits_b, iou_a, iou_b, 
    lam, polarization_weight=0.1
):
    # 1. Prepare base tensors.
    gt_bool = (targets > 0.5).bool()
    weight_map, interest_mask = get_joint_spatial_weight(logits_a, logits_b)
    
    # --- A. Weighted segmentation loss (BCE + Dice) ---
    # Weighted BCE.
    bce = F.binary_cross_entropy_with_logits(fused_logits, targets, reduction='none')
    weighted_bce = (bce * weight_map).mean()
    
    # Weighted Dice.
    probs = torch.sigmoid(fused_logits)
    num = 2 * (probs * targets * weight_map).sum(dim=(2, 3))
    den = (probs * weight_map + targets * weight_map).sum(dim=(2, 3)) + 1e-6
    weighted_dice = (1 - (num / den)).mean()
    
    seg_loss = weighted_bce + weighted_dice

    # --- B. Weighted teacher loss ---
    # Polarized teacher target based on IoU differences.
    diff = iou_a - iou_b  # (B,)
    teacher_target = torch.sigmoid(diff * 100).view(-1, 1, 1, 1).expand_as(gate_logits)
    
    gate_bce = F.binary_cross_entropy_with_logits(gate_logits, teacher_target, reduction='none')
    # Apply teacher supervision only where informative pixels exist.
    weighted_teacher_loss = (gate_bce * weight_map).sum() / (weight_map.sum() + 1e-6)

    # --- C. Weighted polarization entropy loss ---
    p = gate.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    # Encourage confident gates in informative regions while allowing uncertain background gates.
    weighted_polarization_loss = (entropy * weight_map).sum() / (weight_map.sum() + 1e-6)

    # --- Total loss ---
    total_loss = seg_loss + (lam * weighted_teacher_loss) + (polarization_weight * weighted_polarization_loss)
    
    return total_loss, {
        "seg": seg_loss.item(),
        "teacher": weighted_teacher_loss.item(),
        "polar": weighted_polarization_loss.item()
    }


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    if os.environ.get("DGSEG_DRY_RUN", "0") == "1":
        print("DGSEG_DRY_RUN=1")
        print(f"SAM3_REPO_PATH={SAM3_REPO_PATH}")
        print(f"SAM3_CHECKPOINT={SAM3_CHECKPOINT}")
        print(f"IMAGE_ROOT={IMAGE_ROOT}")
        print(f"MLLM_JSON_FILE={MLLM_JSON_FILE}")
        print(f"JSON_FILE_PATH={JSON_FILE_PATH}")
        print("fusion training entry is importable; skipping weight/data loading.")
        return
    script_name = get_script_filename()  # e.g. train_sam3_fusion.py
    ckpt_dir = os.path.join("src", "train_fusion", "checkpoint_3b", script_name.replace(".", "_"))
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[INFO] Checkpoints will be saved to: {ckpt_dir}")

    if SAM3_REPO_PATH not in os.sys.path:
        os.sys.path.append(SAM3_REPO_PATH)

    sam3_model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT).to(DEVICE)
    sam3_model.eval()
    freeze_module(sam3_model)
    processor = Sam3Processor(sam3_model)

    ds = RefCOCOFromJsonlDataset(JSON_FILE_PATH, IMAGE_ROOT, MLLM_JSON_FILE, max_items=-1)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    C = 256
    fusion_net = PixelGateMaskLogitsFusion(
        in_channels=C,
        hidden_channels=GATE_HIDDEN,
        dropout=GATE_DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(fusion_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    fusion_net.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    optim_step = 0

    for epoch in range(NUM_EPOCHS):
        for sample in dl:


            image_path = sample["image_path"][0]
            label = sample["label"][0]
            bbox_xywh_norm = [float(x) for x in sample["bbox_xywh_norm"]]
            bbox_image = [float(x) for x in sample["bbox_image"]]
            segmentation = [[float(x) for x in sample["segmentation"][0]]]

            img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            img_w, img_h = img.size

            gt = load_coco_polygon_mask(segmentation, img_w, img_h)
            if gt is None:
                continue

            out_text = run_branch_text(img, processor, sam3_model, label)
            out_bbox = run_branch_bbox(img, processor, sam3_model, bbox_xywh_norm)

            if ("masks" not in out_text) or ("masks" not in out_bbox):
                continue
            if is_empty_masks(out_text["masks"]) or is_empty_masks(out_bbox["masks"]):
                continue
            if ("pixel_embed" not in out_text) or ("pixel_embed" not in out_bbox):
                continue
            if ("masks_logits" not in out_text) or ("masks_logits" not in out_bbox):
                continue

            pixel_a = out_text["pixel_embed"].to(DEVICE)
            pixel_b = out_bbox["pixel_embed"].to(DEVICE)

            sel_a = select_best_mask_by_bbox_iou(out_text, bbox_image)
            sel_b = select_best_mask_by_bbox_iou(out_bbox, bbox_image)

            logits_a = sel_a["best_logits"].to(DEVICE)  # (1,1,H,W)
            logits_b = sel_b["best_logits"].to(DEVICE)  # (1,1,H,W)
            logits_a = sigmoid_to_logits(logits_a)
            logits_b = sigmoid_to_logits(logits_b)

            if DETACH_BRANCH_OUTPUTS:
                pixel_a = pixel_a.detach()
                pixel_b = pixel_b.detach()
                logits_a = logits_a.detach()
                logits_b = logits_b.detach()

            # Preserve the resolution alignment logic.
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

            fused_logits, gate, gate_logits = fusion_net(pixel_a, pixel_b, logits_a, logits_b, return_gate=True)

            B, _, H, W = fused_logits.shape
            targets = resize_mask_to(gt, H, W).to(DEVICE)  # (1,1,h,w)
            if targets.shape[-2:] != (H, W):
                targets = F.interpolate(targets, size=(H, W), mode="nearest")

            # --- IoU_a / IoU_b between the selected mask and GT.
            # sel_a['best_mask']:(1,1,H,W) bool on sam3 device; targets:(1,1,H,W) float
            pred_a = sel_a["best_mask"].to(DEVICE).bool()
            pred_b = sel_b["best_mask"].to(DEVICE).bool()
            gt_bool = (targets > 0.5).bool()

            iou_a_t = mask_iou(pred_a, gt_bool)  # -> tensor
            iou_b_t = mask_iou(pred_b, gt_bool)
            iou_a_s = safe_scalar(iou_a_t)
            iou_b_s = safe_scalar(iou_b_t)

            lam = teacher_weight(
                step=optim_step,  # Use optimizer steps for the schedule.
                warmup_steps=0,
                total_steps=max(1, int(NUM_EPOCHS * len(dl) / ACCUM_STEPS)),
                init_w=0.5,
                final_w=0.0,
                mode="cosine",
            )

            # teacher_loss expects iou_a/iou_b to be shaped as (B,).
            iou_a_vec = torch.tensor([iou_a_s], device=DEVICE, dtype=torch.float32)
            iou_b_vec = torch.tensor([iou_b_s], device=DEVICE, dtype=torch.float32)
            # --- (1) Foreground gate mean and statistics.
            # Foreground mask: gt_bool.
            fg_mask = gt_bool  # (1,1,H,W)
            fg_count = int(fg_mask.sum().item())
            if fg_count > 0:
                gate_fg = gate[fg_mask]  # 1D
                gate_fg_mean = gate_fg.float().mean().item()
                fg_stats = tensor_stats(gate_fg, prefix="gate_fg_")
            else:
                gate_fg_mean = float("nan")
                fg_stats = {"gate_fg_empty": 1.0}

            # --- (3) Global gate statistics.
            gate_stats = tensor_stats(gate, prefix="gate_")

            global_step += 1

            total_loss, loss_dict = compute_all_losses(
                fused_logits, gate, gate_logits, targets, 
                logits_a, logits_b, iou_a_vec, iou_b_vec, 
                lam, polarization_weight=0.01
            )

            (total_loss / ACCUM_STEPS).backward()

            if global_step % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(fusion_net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

            if SAVE_INTERN:
                if global_step % 3000 == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                    payload = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "optim_step": optim_step,
                        "fusion_net": fusion_net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": {
                            "LR": LR,
                            "WEIGHT_DECAY": WEIGHT_DECAY,
                            "BATCH_SIZE": BATCH_SIZE,
                            "NUM_EPOCHS": NUM_EPOCHS,
                            "ACCUM_STEPS": ACCUM_STEPS,
                            "DETACH_BRANCH_OUTPUTS": DETACH_BRANCH_OUTPUTS,
                            "GATE_DROPOUT": GATE_DROPOUT,
                            "GATE_HIDDEN": GATE_HIDDEN,
                            "SEED": SEED,
                        },
                        "rng_state": make_rng_state(),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    save_checkpoint_atomic(payload, ckpt_path)
                    print(f"[INFO] Saved checkpoint: {ckpt_path}")

            if global_step % LOG_EVERY == 0:
                with torch.no_grad():
                    # (2) Per-loss logging.
                    msg = (
                        f"[ep{epoch}] gstep={global_step} optstep={optim_step} "
                        f"loss={total_loss.item():.6f} | "
                        f"seg={loss_dict['seg']:.6f} "
                        f"teacher={loss_dict['teacher']:.6f} "
                        f"lam={lam:.4f} "
                        f"entropy={loss_dict['polar']:.6f} | "
                        f"IoUa={iou_a_s:.4f} IoUb={iou_b_s:.4f} | "
                        f"fg_gate_mean={gate_fg_mean:.4f} fg_pixels={fg_count}"
                    )
                    print(msg)

                    # (3) Gate statistics, global and foreground.
                    def fmt_stats(d: Dict[str, float]) -> str:
                        keys = [k for k in d.keys() if "empty" not in k]
                        keys_sorted = sorted(keys)
                        return " ".join([f"{k}={d[k]:.4f}" for k in keys_sorted]) + (" " + " ".join([k for k in d.keys() if "empty" in k]) if any("empty" in k for k in d.keys()) else "")

                    print("[gate_stats]   ", fmt_stats(gate_stats))
                    print("[gate_fg_stats]", fmt_stats(fg_stats))

        print(f"Epoch {epoch} done.")

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "optim_step": optim_step,
            "fusion_net": fusion_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "BATCH_SIZE": BATCH_SIZE,
                "NUM_EPOCHS": NUM_EPOCHS,
                "ACCUM_STEPS": ACCUM_STEPS,
                "DETACH_BRANCH_OUTPUTS": DETACH_BRANCH_OUTPUTS,
                "GATE_DROPOUT": GATE_DROPOUT,
                "GATE_HIDDEN": GATE_HIDDEN,
                "SEED": SEED,
            },
            "rng_state": make_rng_state(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_checkpoint_atomic(payload, ckpt_path)

        # Also write last.pt for convenient checkpoint lookup.
        save_checkpoint_atomic(payload, os.path.join(ckpt_dir, "last.pt"))

        print(f"[INFO] Saved checkpoint: {ckpt_path}")

    save_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint_atomic({"fusion_net": fusion_net.state_dict()}, save_path)
    print(f"Saved fusion net to: {save_path}")


if __name__ == "__main__":
    main()
