#!/usr/bin/env python3
"""Build DGSeg RefCOCO-family JSON files from REFER-style annotations.

The released training/evaluation code expects a compact JSON list where each
item merges REFER expressions with the corresponding COCO instance annotation:

{
  "ref_id": int,
  "ann_id": int,
  "image_id": int,
  "image_file": str,
  "category_id": int,
  "category_name": str,
  "bbox": [x, y, w, h],
  "segmentation": COCO polygon or RLE,
  "sentences": [str, ...],
  "split": str
}

Inputs are the standard REFER files:
  refer_seg/refcoco/instances.json + refs(unc).p
  refer_seg/refcoco+/instances.json + refs(unc).p
  refer_seg/refcocog/instances.json + refs(umd).p
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_SPLITS = {
    "refcoco": ["train", "val", "testA", "testB"],
    "refcoco+": ["train", "val", "testA", "testB"],
    "refcocog": ["train", "val", "test"],
}

DEFAULT_SPLIT_BY = {
    "refcoco": "unc",
    "refcoco+": "unc",
    "refcocog": "umd",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_refs(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        return pickle.load(f)


def sentence_texts(ref: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for sent in ref.get("sentences", []):
        if isinstance(sent, dict):
            text = sent.get("sent") or sent.get("raw")
            if text is not None:
                texts.append(str(text))
        elif isinstance(sent, str):
            texts.append(sent)
    return texts


def build_dataset(dataset_dir: Path, split: str, split_by: str) -> List[Dict[str, Any]]:
    instances_path = dataset_dir / "instances.json"
    refs_path = dataset_dir / f"refs({split_by}).p"
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing instances file: {instances_path}")
    if not refs_path.exists():
        raise FileNotFoundError(f"Missing refs file: {refs_path}")

    instances = load_json(instances_path)
    refs = load_refs(refs_path)

    anns_by_id = {ann["id"]: ann for ann in instances["annotations"]}
    images_by_id = {img["id"]: img for img in instances["images"]}
    cats_by_id = {cat["id"]: cat for cat in instances["categories"]}

    items: List[Dict[str, Any]] = []
    for ref in refs:
        if ref.get("split") != split:
            continue
        ann_id = ref["ann_id"]
        image_id = ref["image_id"]
        ann = anns_by_id.get(ann_id)
        image = images_by_id.get(image_id)
        if ann is None:
            raise KeyError(f"ann_id={ann_id} from ref_id={ref.get('ref_id')} not found in instances.json")
        if image is None:
            raise KeyError(f"image_id={image_id} from ref_id={ref.get('ref_id')} not found in instances.json")

        category_id = ref.get("category_id", ann.get("category_id"))
        category = cats_by_id.get(category_id, {})
        items.append(
            {
                "ref_id": ref.get("ref_id"),
                "ann_id": ann_id,
                "image_id": image_id,
                "image_file": image["file_name"],
                "category_id": category_id,
                "category_name": category.get("name", ""),
                "bbox": ann.get("bbox"),
                "segmentation": ann.get("segmentation"),
                "sentences": sentence_texts(ref),
                "split": ref.get("split"),
            }
        )
    return items


def parse_dataset_specs(values: Iterable[str] | None) -> List[Tuple[str, str]]:
    if not values:
        return list(DEFAULT_SPLIT_BY.items())
    specs: List[Tuple[str, str]] = []
    for value in values:
        if ":" in value:
            dataset, split_by = value.split(":", 1)
        else:
            dataset, split_by = value, DEFAULT_SPLIT_BY[value]
        specs.append((dataset, split_by))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DGSeg RefCOCO-family JSON files.")
    parser.add_argument(
        "--refer-root",
        required=True,
        help="Directory containing refcoco, refcoco+, and refcocog REFER-style folders.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated JSON files.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset to build, optionally with splitBy as DATASET:SPLITBY. May be repeated.",
    )
    parser.add_argument(
        "--split",
        action="append",
        help="Split to build. If omitted, standard splits for each dataset are generated.",
    )
    parser.add_argument("--indent", type=int, default=None, help="Pretty-print JSON with this indent.")
    args = parser.parse_args()

    refer_root = Path(args.refer_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, split_by in parse_dataset_specs(args.dataset):
        dataset_dir = refer_root / dataset
        splits = args.split or DEFAULT_SPLITS[dataset]
        for split in splits:
            items = build_dataset(dataset_dir, split, split_by)
            output_path = output_dir / f"{dataset}_{split}_dataset.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=args.indent)
            print(f"wrote {output_path} ({len(items)} samples)")


if __name__ == "__main__":
    main()
