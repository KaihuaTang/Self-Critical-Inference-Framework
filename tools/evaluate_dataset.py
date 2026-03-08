import os
import re
import csv
import json
import pandas as pd
from tqdm import tqdm, trange
from argparse import ArgumentParser

DATASET_LIST = ["MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11", "MME", "CCBench", "MMStar", "ViLP"]
TYPE_MCQ = ["MMStar", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11"]
TYPE_OTHER = ["MME", "ViLP"]

def load_results(root_path, model_name, split_name, dataset_name):
    file_name = f"{model_name}_{dataset_name}_{split_name}.xlsx"
    if not os.path.exists(os.path.join(root_path, file_name)):
        raise ValueError(f"Dataset({dataset_name}) result is not available: no file {file_name}")
    
    results = []
    for _, row in pd.read_excel(os.path.join(root_path, file_name)).iterrows():
        results.append({'index': row['index'], 'answer': row['answer'], 'prediction': row['prediction']})
    return results, file_name


def load_candidate_index(dataset_name, subset_name):
    index_list = []
    # load source data
    df = pd.read_csv(os.path.join("~/LMUData", f"{dataset_name}_{subset_name}.tsv"), sep="\t")
    # add column names
    index_index = df.columns.tolist().index('index')
    # read data line by line
    for _, row in df.iterrows():
        index_list.append(int(row.tolist()[index_index]))
    return index_list


def load_subset_results(subset_name, root_path, model_name, split_name, dataset_name):
    file_name = f"{model_name}_{dataset_name}_{split_name}.xlsx"
    if not os.path.exists(os.path.join(root_path, file_name)):
        raise ValueError(f"Dataset({dataset_name}) result is not available: no file {file_name}")
    
    index_list = load_candidate_index(dataset_name, subset_name)

    results = []
    for _, row in pd.read_excel(os.path.join(root_path, file_name)).iterrows():
        if int(row['index']) in index_list:
            results.append({'index': row['index'], 'answer': row['answer'], 'prediction': row['prediction']})
    return results, file_name

def evaluate_data(results_list):
    accuracy = []
    for item in results_list:
        answer = str(item['answer']).lower()
        prediction = str(item['prediction']).lower()
        if len(prediction) < len(answer):
            accuracy.append(0.0)
        elif len(prediction) == len(answer):
            accuracy.append(float(prediction == answer))
        else:
            # 判断第一个超出 answer 长度的字符，如果这个字符不是一个字母（比如数字、标点或预测没有超长），则继续判断预测是否匹配答案。
            if not bool(re.fullmatch(r"[A-Za-z]", prediction[len(answer)])):
                accuracy.append(float(prediction[:len(answer)] == answer))
            # 如果这个字符是一个字母，说明组成了一个非gt的单词，因此预测错误。
            else:
                accuracy.append(0.0)
    return accuracy
            

def main(args):
    container_all = []
    all_names = []
    container_mcq = []
    mcq_names = []
    container_other = []
    other_names = []
    # process dataset one by one
    for current_dataset in DATASET_LIST:
        if args.eval_subset == "all":
            dataset_results, file_name = load_results(args.result_path, args.model_name, args.split_name, current_dataset)
        else:
            dataset_results, file_name = load_subset_results(args.eval_subset, args.result_path, args.model_name, args.split_name, current_dataset)
        if dataset_results is None:
            raise ValueError(f"Dataset results is None")
        accuracy = evaluate_data(dataset_results)
        print(f"Accuracy: {file_name} = {sum(accuracy) / len(accuracy)} ({len(accuracy)} items)")
        
        if current_dataset in TYPE_MCQ:
            container_mcq += accuracy
            mcq_names.append(file_name)
        
        if current_dataset in TYPE_OTHER:
            container_other += accuracy
            other_names.append(file_name)
        
        container_all += accuracy
        all_names.append(file_name)

    print("=" * 25)
    print(f"Multi-Choice Question Accuracy: = {sum(container_mcq) / len(container_mcq)} ({len(container_mcq)} items)")
    print(f"Multi-Choice Question Datasets: {mcq_names}")
    print("=" * 25)
    print(f"Other Question Accuracy: = {sum(container_other) / len(container_other)} ({len(container_other)} items)")
    print(f"Other Question Datasets: {other_names}")
    print("=" * 25)
    print(f"Overall Accuracy: = {sum(container_all) / len(container_all)} ({len(container_all)} items)")
    print(f"Overall Datasets: {all_names}")
    print("Finished")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--split-name", type=str, required=True)
    parser.add_argument("--eval-subset", default="all", type=str)
    args = parser.parse_args()

    main(args)

# python ./tools/evaluate_dataset.py --result-path xxx --model-name xxx --split-name xxx