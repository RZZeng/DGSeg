# DGSeg

Official code release for **DGSeg: Dynamic Gating of Semantic-Spatial Guided Predictions for Reasoning Segmentation**.

DGSeg addresses reasoning segmentation, where a model must segment the target object implied by a complex language query. Instead of compressing the reasoning result into a single prompt, DGSeg asks the MLLM to produce complementary target cues: a concise semantic description of what the target is and a spatial box describing where it is. SAM3 then processes these cues in separate semantic and spatial branches. A lightweight dynamic gating module estimates pixel-wise fusion weights from branch features and combines the two mask logits, suppressing noisy or conflicting regions while preserving the MLLM's reasoning intent.

![DGSeg method overview](assets/method.png)


## Environment Setup

```bash
conda create -n dgseg python=3.10 -y
conda activate dgseg

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Install the local training package in editable mode:

```bash
cd /path/to/DGSeg/src/open-r1-multimodal
pip install -e .
```

Install the bundled SAM3 package:

```bash
cd /path/to/DGSeg/sam3
pip install -e .
```

Set common paths before training or evaluation:

```bash
export DGSEG_ROOT=/path/to/DGSeg
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/model_weights
export SAM3_REPO_PATH=${DGSEG_ROOT}/sam3
export SAM3_CHECKPOINT=${MODEL_ROOT}/sam3.pt
```

## Data Preparation


### Step 1: Prepare Raw Data

Download RefCOCO and ReasonSeg from their official sources. The RefCOCO-family annotations should be organized in the REFER-style format with `instances.json` and `refs(...).p` files:

```text
${DATA_ROOT}/refer_seg/
|-- images/mscoco/images/train2014/        # COCO train2014 images
|-- refcoco/
|   |-- instances.json
|   `-- refs(unc).p
|-- refcoco+/
|   |-- instances.json
|   `-- refs(unc).p
`-- refcocog/
    |-- instances.json
    `-- refs(umd).p
```

ReasonSeg can be placed under:

```text
${DATA_ROOT}/ReasonSeg/
|-- val/
`-- test/
```

### Step 2: Build DGSeg JSON Files

The `*_dataset.json` files used by DGSeg are derived files, not native RefCOCO files. They merge REFER expressions with COCO instance annotations so each sample contains image metadata, category information, segmentation masks, and all referring expressions.

Generate them with:

```bash
cd ${DGSEG_ROOT}
python tools/build_refcoco_datasets.py \
  --refer-root ${DATA_ROOT}/refer_seg \
  --output-dir ${DATA_ROOT}
```

## Training

Before training, set the common paths used by the scripts, especially `DGSEG_ROOT`, `DATA_ROOT`, `MODEL_PATH`, `SAM3_CHECKPOINT`, and the relevant data paths.

### Stage 1: RL Finetuning

Run the RefCOCOg-9000 GRPO finetuning script for the desired Qwen2.5-VL scale:

```bash
bash run_scripts/run_grpo_rec_lora_refcocog_9000_3b.sh
bash run_scripts/run_grpo_rec_lora_refcocog_9000_7b.sh
```

### Stage 2: Fusion Training

After obtaining cached Qwen2.5-VL predictions, train the SAM3 fusion module:

```bash
bash run_scripts/train_fusion_3b.sh
bash run_scripts/train_fusion_7b.sh
```

## Evaluation


For RefCOCO-family evaluation, configure the split-related variables in the shell or script (`DATASET`, `SPLIT`, `REFCOCO_JSON_DIR`, `IMAGE_FOLDER`, `MODEL_PATH`, `SAM3_CHECKPOINT`, and `FUSION_CKPT`) and run:

```bash
cd ${DGSEG_ROOT}/src/reasonseg_eval
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 eval_refcoco.py
```

For ReasonSeg evaluation, configure `REASONSEG_JSON_DIR`, `MODEL_PATH`, `SAM3_CHECKPOINT`, and `FUSION_CKPT`, then run:

```bash
cd ${DGSEG_ROOT}/src/reasonseg_eval
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 eval_reasonseg.py
```

## Acknowledgements

This project has referenced some excellent open-source repositories ([VLM-R1](https://github.com/om-ai-lab/VLM-R1), [SAM3](https://github.com/facebookresearch/sam3)). Thanks for their wonderful works and contributions to the community.

## Citation

If you find this repository useful, please cite the DGSeg paper. 
```
@article{zeng2026dgseg,
  title={DGSeg: Dynamic Gating of Semantic-Spatial Guided Predictions for Reasoning Segmentation},
  author={Zeng, Ruizhe and Cao, Siyu and Zhang, Lu and Liu, Zhiyong},
  journal={arXiv preprint arXiv:2607.04779},
  year={2026}
}

```
