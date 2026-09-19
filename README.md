# Self-Critical Inference (SCI) and DRBench: Scaling Test-Time Robustness of Vision-Language Models

[![Paper](https://img.shields.io/badge/arXiv-2603.07659-b31b1b.svg)](https://arxiv.org/abs/2603.07659)
[![Conference](https://img.shields.io/badge/CVPR-2026-1f6feb.svg)](https://arxiv.org/abs/2603.07659)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f.svg)](https://kaihuatang.github.io/Self-Critical-Inference-Framework/)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey.svg)](LICENSE)

Official implementation of the **CVPR 2026** paper
**[Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](https://arxiv.org/abs/2603.07659)**
by [Kaihua Tang](https://kaihuatang.github.io/), Jiaxin Qi, Jinli Ou, Yuhua Zheng, and Jianqiang Huang.

**[Paper (arXiv:2603.07659)](https://arxiv.org/abs/2603.07659)** | **[Project Page](https://kaihuatang.github.io/Self-Critical-Inference-Framework/)** | **[ViLP.tsv for VLMEvalKit](https://github.com/KaihuaTang/Custom-Dataset-for-VLMEvalKit)** | **[中文简介](#中文简介)**

## TL;DR

- **Self-Critical Inference (SCI)** is a decoding-time framework for Large Vision-Language Models (LVLMs). It runs several *counterfactual* forward passes (perturbed prompts and perturbed images) and combines their next-token **logits**, so that one method mitigates both **language bias** (answering from language priors instead of the image, which also shows up as **object hallucination**) and **language sensitivity** (answers that flip under semantically equivalent prompts).
- SCI generalizes **Visual Contrastive Decoding (VCD)** and the TIE-based debiasing of **Counterfactual VQA (CF-VQA)**: both are special cases of the SCI formulation.
- Robustness **scales with the number of counterfactual inference rounds** (SCI3 → SCI5 → SCI7): a test-time scaling direction that adds inference rounds rather than longer reasoning chains.
- **Dynamic Robustness Benchmark (DRBench)** is a model-specific benchmark: for a given LVLM it automatically extracts that model's own non-robust samples (Bias Subset, Sensitivity Subset, and their union, the BS Subset) from existing datasets such as MMBench, MME, MMStar, CCBench, and ViLP.
- SCI only changes how logits are combined at inference. This repository contains no training code; it evaluates the Hugging Face checkpoints `Qwen2-VL-7B-Instruct` and `llama3-llava-next-8b-hf` on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Key Results

Top-1 accuracy (%) on the **BS Subset of DRBench** (Overall, 80% test split; Table 2 of the paper):

| Base LVLM | Base | TIE | VCD | M3ID | SCI3 | SCI5 | SCI7 |
|---|---|---|---|---|---|---|---|
| LLaVA-NeXT-8B | 18.75 | 27.31 | 27.89 | 29.05 | 32.72 | 34.19 | **34.92** |
| Qwen2-VL-7B | 14.52 | 22.32 | 23.12 | 25.68 | 26.94 | 29.50 | **31.72** |

- More counterfactual rounds give higher robustness: accuracy on the BS Subset increases monotonically from SCI3 to SCI5 to SCI7 for both models.
- On the six original datasets (Table 4), SCI5 keeps or slightly improves overall accuracy (LLaVA-NeXT 70.46 → 70.79, Qwen2-VL 80.58 → 80.98), so the gains on DRBench do not come at the cost of standard benchmark performance.
- Non-robust samples are model-specific: 24.68% of the test samples are hard for LLaVA-NeXT, but only 7.34% are shared with Qwen2-VL. On the BS Subset built from LLaVA-NeXT, LLaVA-NeXT scores 18.75% while Qwen2-VL scores 60.31% (Table 3).
- With batch inference, SCI3 / SCI5 / SCI7 cost about 1.29× / 1.81× / 2.48× the base inference time (2.96× / 5.01× / 6.68× when the rounds run sequentially), measured on MMStar with one NVIDIA A800 GPU.

Full tables are in the [Results](#results) section.

## Method

### From VCD to TIE

VCD decodes from contrasted logits $(1+\alpha)\,Z(v,q)-\alpha\,Z(v^{*},q)$, where $v^{*}$ is a noisy image. Rewriting this in the $\exp(\cdot)$ domain gives

$$p(y\mid v,v^{*},q)\;\propto\;\exp\big(Z(v,q)\big)\cdot\exp\big(\mathrm{TIE}/\tau\big),\qquad \mathrm{TIE}=Z(v,q)-Z(v^{*},q),\quad \tau=1/\alpha .$$

So VCD reweights the original token distribution with the **Total Indirect Effect (TIE)** logits used by CF-VQA and Unbiased Scene Graph Generation, and $1/\alpha$ plays the role of a temperature.

### Self-Critical Inference

SCI combines a **Textual Counterfactual (TC)** term for prompt consistency and a **Visual Counterfactual (VC)** term for visual grounding:

$$p_{\text{SCI}}(y\mid \boldsymbol{v},\boldsymbol{q})\;\propto\;\exp(\mathrm{TC}/\tau_{1})\cdot\exp(\mathrm{VC}/\tau_{2})$$

$$\mathrm{TC}_{k}=\max_{i}\,Z_{k}(v^{0},q^{i}),\; i=0,\dots,N \qquad\qquad \mathrm{VC}=Z(v^{0},q^{0})-\mathbb{E}_{j}\big[Z(v^{j},q^{0})\big],\; j=1,\dots,M$$

- $v^{0}, q^{0}$ are the original image and prompt; $q^{i}$ are semantically equivalent but lexically different prompts; $v^{j}$ are content-removed images.
- TC takes the element-wise maximum over the $N+1$ prompt variants; VC averages over $M$ counterfactual images for a more stable TIE estimate.
- Following VCD, an **Adaptive Plausibility Constraint** masks tokens whose (temperature-scaled) TC logit is below $\max_k(\cdot)+\log\beta$ before decoding.
- **VCD** is the special case $N=0, M=1$. **CF-VQA** is the special case with a constant TC term and $M=1$.
- SCI3, SCI5, SCI7 denote $M+N+1 = 3, 5, 7$ total inference rounds, i.e. $M=N=1, 2, 3$.

### Counterfactual inputs

| Name | Type | Construction (see `generate_ids_or_logits` and `prompt_variation*`) | Used by |
|---|---|---|---|
| VC-Color0 | visual | every pixel set to RGB (0, 0, 0), a fully black image | SCI3, SCI5, SCI7 |
| VC-Noise500 | visual | diffusion forward-process noise at step 500 (noise function from VCD) | SCI5, SCI7, and the TIE / VCD / M3ID baselines |
| VC-Noise400 | visual | diffusion forward-process noise at step 400 | SCI7 |
| TC-V1 | textual | adds an instruction to focus on the details of the given image | SCI3, SCI5, SCI7 |
| TC-V2 | textual | same intent as TC-V1, with the instruction language switched (English ↔ Chinese) | SCI5, SCI7 |
| TC-V3 | textual | injects an identity ("a smart student who is good at answering ... questions") | SCI7 |

### Hyperparameters: paper notation ↔ code arguments

| Paper | Code argument (`vlmeval/config.py`) | Value |
|---|---|---|
| $\tau_{1}$, temperature of TC | `gamma` | 1.5 (SCI3), 2.0 (SCI5), 2.5 (SCI7) |
| $\tau_{2}$, temperature of VC | `beta` | 0.2 |
| $\beta$, Adaptive Plausibility Constraint threshold | `theta` | 0.3 on DRBench; 0.8 on the original datasets |
| $\alpha$ of VCD / M3ID | `alpha` | 1.0 (VCD), 0.02 (M3ID) |

The suffix in a registered model name encodes these values, e.g. `Qwen2-VL-7B-SCI5-b02a1g2t03` means `beta=0.2, alpha=1, gamma=2, theta=0.3`. Hyperparameters were selected on the validation split of the Qwen2-VL BS Subset and applied to LLaVA-NeXT unchanged.

## Dynamic Robustness Benchmark (DRBench)

DRBench converts any existing LVLM dataset into a robustness benchmark for one specific model, in two steps: (1) evaluate the dataset with the model under the original input, two visual counterfactual inputs, and two textual counterfactual inputs ($M=N=2$); (2) filter the samples:

- **Bias Subset (B)**: samples on which the model gives the same *incorrect* prediction under the original and the dummy (content-removed) visual inputs, indicating reliance on spurious language priors.
- **Sensitivity Subset (S)**: samples whose prediction changes under subtle, non-causal prompt variations.
- **BS Subset**: the union of the two.

Results are reported separately for **MCQ** (multiple-choice questions) and **Others** (Yes/No questions of MME and open-ended QA of ViLP). The exact matching rule is `get_biased_data` in [`tools/generate_dataset.py`](tools/generate_dataset.py).

The paper builds DRBench from six benchmarks: **MME, MMStar, CCBench, ViLP, MMBench-DEV-EN-V11, MMBench-DEV-CN-V11**, split into 20% validation (3,315 samples) and 80% test (13,251 samples; 10,632 MCQ and 2,619 Others). Test-split subset sizes (Table 1):

| Construction model | B Subset | S Subset | BS Subset | Overlap |
|---|---|---|---|---|
| LLaVA-NeXT (MCQ / Others / Overall) | 1810 / 345 / 2155 | 1005 / 582 / 1587 | 2476 / 794 / 3270 | 339 / 133 / 472 |
| Qwen2-VL (MCQ / Others / Overall) | 1080 / 327 / 1407 | 252 / 311 / 563 | 1243 / 513 / 1756 | 89 / 125 / 214 |

Generated split files follow this naming (written to the VLMEvalKit data root, `~/LMUData`):

| File suffix | Meaning |
|---|---|
| `{Dataset}_Custom_Val.tsv`, `{Dataset}_Custom_Test.tsv` | 20% / 80% split of the original dataset |
| `{Dataset}_{Model}_VCF_{Val,Test}.tsv` | Bias Subset |
| `{Dataset}_{Model}_TCF_{Val,Test}.tsv` | Sensitivity Subset |
| `{Dataset}_{Model}_Biased_{Val,Test}.tsv` | BS Subset (union) |

## Repository Structure

This repository is a modified copy of [VLMEvalKit v0.2](https://github.com/open-compass/VLMEvalKit/tree/v0.2). The SCI- and DRBench-specific parts are:

| Path | Content |
|---|---|
| [`vlmeval/vlm/qwen2_vl/model.py`](vlmeval/vlm/qwen2_vl/model.py) | Qwen2-VL wrapper: visual/textual counterfactual input construction, and the TIE / VCD / M3ID / SCI3 / SCI5 / SCI7 inference rounds |
| [`vlmeval/vlm/qwen2_vl/modeling_qwen2_vl.py`](vlmeval/vlm/qwen2_vl/modeling_qwen2_vl.py) | Logit-level aggregation inside `forward` (TC element-wise max, VC averaging, adaptive plausibility constraint) |
| [`vlmeval/vlm/llava/llava.py`](vlmeval/vlm/llava/llava.py), [`vlmeval/vlm/llava/modeling_llava_next.py`](vlmeval/vlm/llava/modeling_llava_next.py) | The same for LLaVA-NeXT |
| [`vlmeval/config.py`](vlmeval/config.py) | Registered models: `*-Original`, `*-VCF-Color0`, `*-VCF-Noise400`, `*-VCF-Noise500`, `*-TCF-V1/V2/V3`, `*-TIE`, `*-VCD`, `*-M3ID`, `*-SCI3/5/7-*` |
| [`tools/generate_dataset.py`](tools/generate_dataset.py), [`tools/generate_dataset.yaml`](tools/generate_dataset.yaml) | DRBench construction |
| [`tools/evaluate_dataset.py`](tools/evaluate_dataset.py) | Accuracy on B / S / BS subsets, with MCQ / Others breakdown |
| [`tools/validation.py`](tools/validation.py) | Hyperparameter search on dumped logits of the validation split |
| [`tools/find_dataset_overlap.py`](tools/find_dataset_overlap.py) | Overlap of non-robust samples between models |
| [`tools/batch_inference_speed.py`](tools/batch_inference_speed.py) | Timing of batched inference (batch sizes 1 / 3 / 5 / 7) |
| [`visualization/`](visualization/) | Notebooks and figures for the test-time scaling curves |

## Getting Started

### 0. Environment (创建环境)

Based on [VLMEvalKit v0.2](https://github.com/open-compass/VLMEvalKit/tree/v0.2).

```bash
conda create -n vlmeval python=3.10
conda activate vlmeval

git clone https://github.com/KaihuaTang/Self-Critical-Inference-Framework.git
cd Self-Critical-Inference-Framework
pip install -e .

MAX_JOBS=64 pip -v install flash-attn==2.2.0 --no-build-isolation
pip install accelerate
pip install qwen-vl-utils
```

The experiments in the paper were run on a single NVIDIA A800 (80GB) GPU with PyTorch 2.6, Transformers 4.49, and Flash Attention 2.7. Qwen2-VL uses bfloat16 with its default top-k sampling; LLaVA-NeXT uses float16 with greedy decoding.

Before running, edit [`vlmeval/config.py`](vlmeval/config.py): set `model_path` to your local `Qwen2-VL-7B-Instruct` / `llama3-llava-next-8b-hf` checkpoints, and set `dump_path` (where logits are saved for validation).

### 1. Datasets (数据集)

MME, MMStar, CCBench, and MMBench are standard VLMEvalKit datasets. `ViLP.tsv` is provided at [KaihuaTang/Custom-Dataset-for-VLMEvalKit](https://github.com/KaihuaTang/Custom-Dataset-for-VLMEvalKit); place it in the VLMEvalKit data root (`~/LMUData`).

### 2. Run the base model and its counterfactual variants (执行初始推理)

```bash
bash step0_run_basemodel.sh
```

This runs `*-Original`, `*-VCF-Color0`, `*-VCF-Noise500`, `*-TCF-V1`, and `*-TCF-V2` on the six datasets.

### 3. Build DRBench (处理数据)

Point the result paths in [`tools/generate_dataset.yaml`](tools/generate_dataset.yaml) to your step-0 outputs, then:

```bash
bash step1_generate_data.sh
```

### 4. Re-run the base models on the generated subsets (重新推理划分好的 BS Subset)

```bash
bash step2_test_basemodel_a.sh
bash step2_test_basemodel_b.sh
```

### 5. Evaluate (运行测试脚本)

```bash
python ./tools/evaluate_dataset.py --result-path ./outputs_test/Qwen2-VL-7B-Original/T20250708_Gc4a387d4/ --model-name Qwen2-VL-7B-Original --split-name Qwen2-VL-7B_Biased_Test
```

`--result-path` is the run folder created by VLMEvalKit; replace the timestamped folder name with your own.

### 6. Select hyperparameters on the validation split (运行 validation 调参)

```bash
bash step3_validation_a.sh
bash step3_validation_b.sh

python tools/validation.py --model-name Qwen2-VL-7B --split-name Qwen2-VL-7B_Biased_Val --logit-path ./dump_tensors/ --result-path ./outputs_val/
```

### 7. Run TIE / VCD / M3ID / SCI with the selected hyperparameters (基于超参跑实验)

```bash
bash step4_test_algorithm_a.sh
bash step4_test_algorithm_b.sh
```

The `--model` argument must be a name registered in [`vlmeval/config.py`](vlmeval/config.py), for example `Qwen2-VL-7B-SCI3-b02a1g15t03`, `Qwen2-VL-7B-SCI5-b02a1g2t03`, `Qwen2-VL-7B-SCI7-b02a1g25t03`, or the `LLaVA-NeXT-8B-*` counterparts. `*-SCI5-b02a1g2t08` (`theta=0.8`) is the setting for the original datasets.

## Results

### DRBench (Table 2)

Top-1 accuracy (%) on the 80% test split. B = Bias Subset, S = Sensitivity Subset, BS = their union.

| Method | B: MCQ | B: Others | B: Overall | S: MCQ | S: Others | S: Overall | BS: MCQ | BS: Others | BS: Overall |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-NeXT | 0.0 | 0.0 | 0.0 | 39.2 | 37.63 | 38.63 | 15.91 | 27.58 | 18.75 |
| LLaVA-NeXT-TIE | 12.98 | 23.48 | 14.66 | 39.00 | 57.56 | 45.81 | 21.89 | 44.21 | 27.31 |
| LLaVA-NeXT-VCD | 12.65 | 25.51 | 14.71 | 40.50 | 56.53 | 46.38 | 22.54 | 44.58 | 27.89 |
| LLaVA-NeXT-M3ID | 16.91 | 25.22 | 18.24 | 39.90 | 56.36 | 45.94 | 24.15 | 44.33 | 29.05 |
| LLaVA-NeXT-SCI3 (ours) | 21.22 | 35.36 | 23.48 | 39.60 | 60.31 | 47.20 | 27.14 | 50.13 | 32.72 |
| LLaVA-NeXT-SCI5 (ours) | 23.81 | 37.97 | 26.08 | **40.60** | **60.65** | **47.95** | 28.80 | 51.01 | 34.19 |
| LLaVA-NeXT-SCI7 (ours) | **24.86** | **38.26** | **27.01** | 40.10 | **60.65** | 47.64 | **29.68** | **51.26** | **34.92** |
| Qwen2-VL | 5.37 | 8.56 | 6.11 | 38.10 | 34.41 | 36.06 | 10.78 | 23.59 | 14.52 |
| Qwen2-VL-TIE | 16.20 | 16.82 | 16.35 | 45.63 | 36.66 | 40.67 | 20.27 | 27.29 | 22.32 |
| Qwen2-VL-VCD | 15.74 | 21.71 | 17.13 | 46.83 | 40.84 | 43.52 | 20.11 | 30.41 | 23.12 |
| Qwen2-VL-M3ID | 19.81 | 21.71 | 20.26 | **47.22** | 41.16 | 43.87 | 23.65 | 30.6 | 25.68 |
| Qwen2-VL-SCI3 (ours) | 21.67 | 26.30 | 22.74 | 44.05 | 42.44 | 43.16 | 24.54 | 32.75 | 26.94 |
| Qwen2-VL-SCI5 (ours) | 24.91 | 25.69 | 25.09 | **47.22** | 42.44 | 44.58 | 28.00 | 33.14 | 29.50 |
| Qwen2-VL-SCI7 (ours) | **27.04** | **29.66** | **27.65** | **47.22** | **45.98** | **46.54** | **29.61** | **36.84** | **31.72** |

Why is the base accuracy on the Bias Subset (close to) zero? By definition the Bias Subset collects samples that the base model answers incorrectly, so its expected accuracy is 0.0. LLaVA-NeXT uses greedy decoding and scores exactly 0.0; Qwen2-VL uses top-k sampling by default, so its accuracy is slightly above zero.

### Original datasets (Table 4)

Top-1 accuracy (%) on the 80% test splits. MMB-C / MMB-E = MMBench-DEV-CN-V11 / MMBench-DEV-EN-V11, CCB = CCBench, MMS = MMStar. MME scores are converted to accuracy.

| Method | MMB-C | MMB-E | MME | CCB | MMS | ViLP | MCQ | Others | Overall |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-NeXT | 78.0 | 79.72 | 79.57 | 47.0 | 44.75 | 51.53 | 70.12 | 71.86 | 70.46 |
| LLaVA-NeXT-TIE | 78.28 | 80.28 | 77.30 | 45.65 | 46.00 | 53.19 | 70.36 | 70.68 | 70.42 |
| LLaVA-NeXT-VCD | 78.38 | 80.28 | 78.09 | 46.63 | 45.00 | 54.31 | 70.44 | 71.55 | 70.66 |
| LLaVA-NeXT-M3ID | 78.31 | 80.18 | 78.62 | 45.89 | 45.92 | 54.03 | 70.36 | 71.86 | 70.66 |
| LLaVA-NeXT-SCI5 (ours) | 78.21 | 80.08 | 80.15 | 46.20 | 45.75 | 53.06 | 70.32 | 72.70 | 70.79 |
| Qwen2-VL | 85.26 | 86.36 | 87.89 | 73.22 | 59.50 | 56.53 | 80.91 | 79.27 | 80.58 |
| Qwen2-VL-TIE | 86.00 | 86.59 | 86.52 | 73.84 | 59.00 | 57.08 | 81.30 | 78.43 | 80.73 |
| Qwen2-VL-VCD | 86.05 | 86.56 | 86.41 | 73.77 | 60.08 | 57.92 | 81.42 | 78.58 | 80.86 |
| Qwen2-VL-M3ID | 85.69 | 86.46 | 86.10 | 73.96 | 59.75 | 57.78 | 81.25 | 78.31 | 80.67 |
| Qwen2-VL-SCI5 (ours) | 85.97 | 86.67 | 87.36 | 73.59 | 59.92 | 58.06 | 81.39 | 79.31 | 80.98 |

### Cross-model evaluation on the BS Subset (Table 3)

| BS Subset constructed by | Evaluated method | MCQ | Others | Overall |
|---|---|---|---|---|
| LLaVA-NeXT | LLaVA-NeXT-Original | 15.91 | 27.58 | 18.75 |
| LLaVA-NeXT | LLaVA-NeXT-SCI5 | 28.80 | 51.01 | 34.19 |
| LLaVA-NeXT | Qwen2-VL-Original | 59.29 | 63.48 | 60.31 |
| LLaVA-NeXT | Qwen2-VL-SCI5 | 61.15 | 67.88 | 62.78 |
| Qwen2-VL | Qwen2-VL-Original | 10.78 | 23.59 | 14.52 |
| Qwen2-VL | Qwen2-VL-SCI5 | 28.00 | 33.14 | 29.50 |
| Qwen2-VL | LLaVA-NeXT-Original | 30.25 | 39.18 | 32.86 |
| Qwen2-VL | LLaVA-NeXT-SCI5 | 34.59 | 41.33 | 36.56 |

SCI still improves a model on the vulnerable set derived from the *other* model, so the gains are not tailored to the model's own DRBench.

### Inference time (Table 7)

Average time per sample on MMStar, one NVIDIA A800 GPU, Qwen2-VL:

| | Qwen2-VL | SCI3 | SCI5 | SCI7 |
|---|---|---|---|---|
| Sequential rounds | 540.47 ms | 1599.65 ms (2.96×) | 2707.16 ms (5.01×) | 3611.18 ms (6.68×) |
| Batch inference | 540.47 ms | 697.24 ms (1.29×) | 978.14 ms (1.81×) | 1342.86 ms (2.48×) |

## Citation

If you find this project helps your research, please kindly consider citing our paper in your publications.

```bibtex
@inproceedings{tang2026scaling,
  title={Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework},
  author={Tang, Kaihua and Qi, Jiaxin and Ou, Jinli and Zheng, Yuhua and Huang, Jianqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgements

This codebase is built on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) (the original VLMEvalKit README is kept as [`VLMEval_README.md`](VLMEval_README.md)). The diffusion-noise function for visual counterfactual images follows the official [VCD](https://github.com/DAMO-NLP-SG/VCD) implementation. The paper was supported by the Double First-Class Initiative Fund, Disciplinary Development Program of the Institute of AI for Engineering, Tongji University.

## 中文简介

本仓库是 **CVPR 2026** 论文《[Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](https://arxiv.org/abs/2603.07659)》的官方代码，作者：Kaihua Tang、Jiaxin Qi、Jinli Ou、Yuhua Zheng、Jianqiang Huang。

- **自我批判推理框架（Self-Critical Inference, SCI）**：一种作用于推理（解码）阶段的大型视觉语言模型（LVLM，多模态大模型）鲁棒性方法。它对同一输入执行多轮**反事实推理**（文本反事实 TC：语义等价但措辞不同的提示词；视觉反事实 VC：去除内容的全黑图像或扩散加噪图像），在 **logit 层面**聚合与对比各轮输出，同时缓解**语言偏见**（language bias，模型依赖语言先验而忽视图像，表现为**物体幻觉**）与**语言敏感性**（language sensitivity，提示词微小变化导致答案改变）。
- SCI 统一并推广了**视觉对比解码（VCD）**与反事实 VQA（CF-VQA）中的 TIE 去偏方法，二者均为 SCI 的特例。
- **测试时扩展（test-time scaling）的新方向**：增加反事实推理轮数（SCI3 → SCI5 → SCI7）即可持续提升鲁棒性，而不是增加单次推理中的思考 token 长度。
- **动态鲁棒性基准（Dynamic Robustness Benchmark, DRBench）**：针对具体模型，从 MMBench、MME、MMStar、CCBench、ViLP 等现有数据集中自动筛选该模型自身的偏见子集（Bias Subset）、敏感性子集（Sensitivity Subset）及其并集（BS Subset）。
- 实验基于 VLMEvalKit，基座模型为 Qwen2-VL-7B-Instruct 与 Llama3-LLaVA-NeXT-8B。在 DRBench 的 BS 子集上，SCI7 将 LLaVA-NeXT 的总体准确率从 18.75% 提升到 34.92%，将 Qwen2-VL 从 14.52% 提升到 31.72%，均优于 TIE、VCD 和 M3ID。

使用步骤见上文 [Getting Started](#getting-started)。

## Keywords

Large Vision-Language Models (LVLMs), multimodal large language models, robustness, language bias, language prior, language sensitivity, prompt sensitivity, object hallucination, counterfactual inference, counterfactual reasoning, Total Indirect Effect (TIE), Counterfactual VQA, contrastive decoding, Visual Contrastive Decoding (VCD), M3ID, test-time scaling, inference-time scaling, logit aggregation, adaptive plausibility constraint, dynamic benchmark, model-specific benchmark, DRBench, Qwen2-VL, LLaVA-NeXT, VLMEvalKit, MMBench, MME, MMStar, CCBench, ViLP, CVPR 2026.

关键词：视觉语言模型、多模态大模型、鲁棒性、语言偏见、语言先验、语言敏感性、提示词敏感性、物体幻觉、反事实推理、对比解码、视觉对比解码、测试时扩展、推理时扩展、动态基准、CVPR 2026。
