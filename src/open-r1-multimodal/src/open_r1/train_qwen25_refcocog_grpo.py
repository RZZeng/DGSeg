# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GRPO training main script adapted for Qwen REC with 3 rewards:
  - format        (from Qwen2VLModule_seg)
  - accuracy/IoU  (bbox IoU, from Qwen2VLModule_seg)
  - segmentation  (mask IoU via SAM3, from Qwen2VLModule_seg)

Key:
- Set is_reward_customized_from_vlm_module=True so reward funcs come from vlm_module.select_reward_func.
- Ensure dataset keeps `image_path`, `problem`, `solution`, and `segmentation` (optional but needed for seg reward).
"""

import os
import re
import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

from datasets import Dataset
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig

logger = logging.get_logger(__name__)

# ---- Optional Qwen2.5 monkey patches ----
try:
    from open_r1.qwen2_5vl_monkey_patch import (
        monkey_patch_qwen2_5vl_flash_attn,
        monkey_patch_qwen2_5vl_forward,
        monkey_patch_torch_load,
    )

    monkey_patch_qwen2_5vl_flash_attn()
    monkey_patch_torch_load()
except Exception as e:
    logger.warning(f"[WARN] Qwen2.5 monkey patch not applied: {e}")
    monkey_patch_qwen2_5vl_forward = None

from open_r1.vlm_modules import *  # type: ignore

tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

MAX_SAMPLES = 9000 # we collect 3000 instances from refcocog


# ---------------------------
# Script / Model arguments
# ---------------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files (jsonl), separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )

    # >>> Modified defaults for this REC+SEG mode <<<
    reward_funcs: List[str] = field(
        default_factory=lambda: ["format", "accuracy", "segmentation"],
        metadata={"help": "Reward funcs from VLM module: format / accuracy(iou) / segmentation(mask_iou)"},
    )
    task_type: Optional[str] = field(
        default="rec",
        metadata={"help": "Task type, should be 'rec' for referring expression localization/seg"},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=True,
        metadata={"help": "Use reward funcs from vlm module (required for this mode)"},
    )

    # Image processor knobs (passed to trainer, used by qwen processor)
    max_pixels: Optional[int] = field(
        default=1024*1024,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of dynamic image blocks"},
    )

    # Kept for compatibility with existing configurations; unused in this mode.
    reward_method: Optional[str] = field(
        default=None,
        metadata={"help": "Legacy field (unused in this mode)."},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def get_vlm_module(model_name_or_path: str):
    name = model_name_or_path.lower()
    if "qwen" in name:
        return Qwen2VLModule_seg
    raise ValueError(f"Unsupported model for this release: {model_name_or_path}. Expected a Qwen2.5-VL checkpoint.")


def _fix_image_path(p: str) -> str:
    """Normalize duplicated train2014 fragments used by some REFER exports."""
    if p is None or os.path.exists(p):
        return p

    p2 = p.replace("images/train2014/train2014/", "")
    if os.path.exists(p2):
        return p2

    p3 = p.replace("train2014/train2014/", "train2014/")
    if os.path.exists(p3):
        return p3

    return p2


def build_dataset_from_jsonl(script_args: GRPOScriptArguments, question_prompt: str, reward_method = None) -> Dataset:
    """
    Read jsonl(s), attach image_path, problem, solution, and KEEP segmentation if present.
    Output columns used by VLMGRPOTrainer:
      - prompt (chat template input)
      - image_path (list[str])  # multi-image supported
      - problem (str)
      - solution (str, must contain <answer>...</answer> with JSON bbox list/dict)
      - segmentation (optional; COCO polygon list or RLE dict)
    """
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    problem_template = "<image>Please provide the bounding box coordinate and the label of the region this sentence describes: <sentence>"

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    all_data: List[Dict[str, Any]] = []
    image_file_set = set()

    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if len(all_data) >= MAX_SAMPLES:
                    break
                if "image_file" in item:
                    if isinstance(item["image_file"], str):
                        p = os.path.join(image_folder, item["image_file"])
                        item["image_path"] = [_fix_image_path(p)]
                    elif isinstance(item["image_file"], list):
                        item["image_path"] = [_fix_image_path(os.path.join(image_folder, im)) for im in item["image_file"]]
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image_file'])}")
                    del item["image_file"]
                    
                sentences = item["sentences"]

                for path in item["image_path"]:
                    if path in image_file_set:
                        continue
                    image_file_set.add(path)

                for idx, sentence in enumerate(sentences):
                    if idx > 0:
                        continue
                    new_item = dict()
                    new_item["image_path"] = item["image_path"]
                    new_item["problem"] = problem_template.replace("<sentence>", sentence).replace('<image>', '')
                    new_item["solution"] = str(item["bbox"])
                    new_item["segmentation"] = str(item.get("segmentation", None))
                    new_item["accu_reward_method"] = accu_reward_method

                    all_data.append(new_item)

    dataset = Dataset.from_list(all_data)

    def make_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
        # Prepare prompt structure (multi-modal chat template style)
        if "image_path" in example and example["image_path"] is not None:
            paths = [_fix_image_path(p) for p in example["image_path"]]
            assert all(os.path.exists(p) for p in paths), f"Image paths do not exist: {paths}"

            return {
                "image_path": paths,  # list[str]
                "problem": example["problem"],
                # trainer passes this to reward as `solution`
                "solution": f"<answer>{example['solution']}</answer>",
                "accu_reward_method": example['accu_reward_method'],
                # keep segmentation for seg reward
                "segmentation": example.get("segmentation", None),
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            *({"type": "image", "text": None} for _ in range(len(paths))),
                            {"type": "text", "text": question_prompt.format(Question=example["problem"])},
                        ],
                    }
                ],
            }
        else:
            return {
                "problem": example["problem"],
                "solution": f"<answer>{example['solution']}</answer>",
                "segmentation": example.get("segmentation", None),
                'accu_reward_method': example['accu_reward_method'],
                "prompt": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question_prompt.format(Question=example["problem"])}],
                    }
                ],
            }

    dataset = dataset.map(make_conversation, num_proc=8)
    return dataset


# ---------------------------
# Main
# ---------------------------
def main(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: GRPOModelConfig):
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Use template from module, but for REC we further constrain the answer format
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.task_type)

    # Rewards must come from the selected VLM module.
    if not script_args.is_reward_customized_from_vlm_module:
        raise ValueError(
            "This training mode requires --is_reward_customized_from_vlm_module True "
            "so rewards are loaded from Qwen2VLModule_seg."
        )

    reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in script_args.reward_funcs]
    print("reward funcs:", script_args.reward_funcs)
    print("reward funcs callables:", reward_funcs)

    # Build dataset (keeps segmentation if present)
    dataset = build_dataset_from_jsonl(script_args, question_prompt)

    # Split validation if needed
    splits = {"train": dataset}
    if script_args.val_split_ratio and script_args.val_split_ratio > 0:
        train_val = dataset.train_test_split(test_size=script_args.val_split_ratio)
        splits["train"] = train_val["train"]
        splits["validation"] = train_val["test"]

    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    # Initialize trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits["train"],
        eval_dataset=splits.get("validation") if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    # Train (resume if checkpoints exist)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Apply the compatible forward patch when using DeepSpeed ZeRO-3.
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        if monkey_patch_qwen2_5vl_forward is not None:
            print("zero3 is used, qwen2_5vl forward monkey patch is applied")
            monkey_patch_qwen2_5vl_forward()

    main(script_args, training_args, model_args)
