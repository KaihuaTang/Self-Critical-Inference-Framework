# Self-Critical Inference Framework

Our paper [Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework]() is accepted to CVPR 2026.

If you find this project helps your research, please kindly consider citing our paper in your publications.
```
@inproceedings{tang2026scaling,
  title={Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework},
  author={Tang, Kaihua and Qi, Jiaxin and Ou, Jinli and Zheng, Yuhua and Huang, Jianqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

### 创建环境
使用[VLMEvalKit-0.2版](https://github.com/open-compass/VLMEvalKit/tree/v0.2)

```
conda create -n myenv python=3.10

conda activate vlmeval

cd VLMEvalKit

pip install -e .

MAX_JOBS=64 pip -v install flash-attn==2.2.0 --no-build-isolation

pip install accelerate

pip install qwen-vl-utils
```

### 数据集

ViLP.tsv 下载路径 https://github.com/KaihuaTang/Custom-Dataset-for-VLMEvalKit


### 执行初始推理
```
bash step0_run_basemodel.sh
```

### 处理数据
```
bash step1_generate_data.sh
```

### 重新推理划分好的BS-Subset
```
bash step2_test_basemodel_a.sh
bash step2_test_basemodel_b.sh
```

### 运行测试脚本
```
python ./tools/evaluate_dataset.py --result-path ./outputs_test/Qwen2-VL-7B-Original/T20250708_Gc4a387d4/ --model-name Qwen2-VL-7B-Original --split-name Qwen2-VL-7B_Biased_Test
```

### 运行validation调参
```
bash step3_validation_a.sh
bash step3_validation_b.sh

python tools/validation.py --model-name Qwen2-VL-7B --split-name Qwen2-VL-7B_Biased_Val --logit-path ./dump_tensors/ --result-path ./outputs_val/
```

### 基于超参跑实验
```
base step4_test_algorithm.sh
```