import os
import re
import csv
import json
import yaml
import pandas as pd
from tqdm import tqdm, trange
from argparse import ArgumentParser

ROOT_PATH = "/home/couser/datasets/LMUData"
DATASET_LIST = ["MMStar", "MME", "ViLP", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11"]

MODEL_A = "LLaVA-NeXT-8B"
MODEL_B = "Qwen2-VL-7B"

SPLIT_NAME = "Biased_Test"

count_a = 0
count_b = 0
count_joint = 0

def load_raw_data(source_root, current_dataset):
    index_list = []
    # load source data
    df = pd.read_csv(os.path.join(source_root, f"{current_dataset}.tsv"), sep="\t")
    # add column names
    index_id = df.columns.tolist().index('index')
    # read data line by line
    for _, row in df.iterrows():
        index_list.append(row.tolist()[index_id])
    return index_list

for dataset_name in DATASET_LIST:
    mode_a_dataname = f"{dataset_name}_{MODEL_A}_Biased_Test"
    mode_b_dataname = f"{dataset_name}_{MODEL_B}_Biased_Test"

    model_a_index = load_raw_data(ROOT_PATH, mode_a_dataname)
    model_b_index = load_raw_data(ROOT_PATH, mode_b_dataname)

    count_a += len(model_a_index)
    count_b += len(model_b_index)
    count_joint += len(set(model_a_index) & set(model_b_index))

print(f"{MODEL_A} Size: {count_a}")
print(f"{MODEL_B} Size: {count_b}")
print(f"Intersection Size: {count_joint}")