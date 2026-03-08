import os
import re
import csv
import json
import yaml
import pandas as pd
from tqdm import tqdm, trange
from argparse import ArgumentParser

DATASET_LIST = ["MMStar", "MME", "ViLP", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11"]

def load_results(model_name, root_path, dataset_name):
    results = {}
    for _, row in pd.read_excel(os.path.join(root_path, f"{model_name}_{dataset_name}.xlsx")).iterrows():
        results[row['index']] = {'answer': row['answer'], 'prediction': row['prediction']}
    return results

def load_all_results(model_configs, dataset_name):
    base_results = load_results(model_configs['base_name'], model_configs['base_result_path'], dataset_name)
    tcf1_results = load_results(model_configs['tcf1_name'], model_configs['tcf1_result_path'], dataset_name)
    tcf2_results = load_results(model_configs['tcf2_name'], model_configs['tcf2_result_path'], dataset_name)
    vcf1_results = load_results(model_configs['vcf1_name'], model_configs['vcf1_result_path'], dataset_name)
    vcf2_results = load_results(model_configs['vcf2_name'], model_configs['vcf2_result_path'], dataset_name)
    print(f"Load {len(base_results)} items from {dataset_name}")
    assert len(base_results) == len(tcf1_results)
    assert len(base_results) == len(tcf2_results)
    assert len(base_results) == len(vcf1_results)
    assert len(base_results) == len(vcf2_results)
    return base_results, tcf1_results, tcf2_results, vcf1_results, vcf2_results

def get_biased_data(args, configs, dataset_name, print_filter=False):
    model_configs = configs[args.model_name]
    base_results, tcf1_results, tcf2_results, vcf1_results, vcf2_results = load_all_results(model_configs, dataset_name)
    
    all_data_val = []
    all_data_test = []
    vcf_data_val = []
    vcf_data_test = []
    tcf_data_val = []
    tcf_data_test = []

    filtered_data = []

    val_devider = 5 # equal to set 1/5 as val data 

    for i, (key, base_item) in enumerate(base_results.items()):
        vcf1_item = vcf1_results[key]
        vcf2_item = vcf2_results[key]
        tcf1_item = tcf1_results[key]
        tcf2_item = tcf2_results[key]
        assert base_item['answer'] == vcf1_item['answer']
        assert base_item['answer'] == vcf2_item['answer']
        assert base_item['answer'] == tcf1_item['answer']
        assert base_item['answer'] == tcf2_item['answer']
        answer = base_item['answer'].lower()
        base_pred = str(base_item['prediction']).lower()
        vcf1_pred = str(vcf1_item['prediction']).lower()
        vcf2_pred = str(vcf2_item['prediction']).lower()
        tcf1_pred = str(tcf1_item['prediction']).lower()
        tcf2_pred = str(tcf2_item['prediction']).lower()
        ans_len = len(answer)

        current_item = {'dataset': dataset_name, 'index': key, 
                        'answer': base_item['answer'],
                        'base_pred': base_item['prediction'],
                        'vcf1_pred': vcf1_item['prediction'],
                        'vcf2_pred': vcf2_item['prediction'],
                        'tcf1_pred': tcf1_item['prediction'],
                        'tcf2_pred': tcf2_item['prediction'],}

        if i % val_devider == 0:
            all_data_val.append(current_item)
        else:
            all_data_test.append(current_item)

        # blind model
        if ((base_pred[:ans_len] == vcf1_pred[:ans_len]) or (base_pred[:ans_len] == vcf2_pred[:ans_len])) and (base_pred[:ans_len] != answer):
            if (len(base_pred) > ans_len) and (dataset_name not in ["ViLP", "MME"]) and (base_pred[ans_len] not in [' ', '.', ',', ';', '。', '，']):
                filtered_data.append(current_item)
                if print_filter:
                    print(f"Filter the following data: { str(current_item) }")
            else:
                if i % val_devider == 0:
                    vcf_data_val.append(current_item)
                else:
                    vcf_data_test.append(current_item)
        
        # language in-consistancy
        if (base_pred[:ans_len] != tcf1_pred[:ans_len]) or (base_pred[:ans_len] != tcf2_pred[:ans_len]):
            if i % val_devider == 0:
                tcf_data_val.append(current_item)
            else:
                tcf_data_test.append(current_item)

    all_splits = {'all_data_val': all_data_val,
                  'all_data_test': all_data_test,
                  'vcf_data_val': vcf_data_val,
                  'vcf_data_test': vcf_data_test,
                  'tcf_data_val': tcf_data_val,
                  'tcf_data_test': tcf_data_test,
                  'filtered_data': filtered_data,}

    return all_splits


def merge_selection(selection_a, selection_b):
    index_container = []
    merged_selection = []
    for item in selection_a:
        if item['index'] not in index_container:
            index_container.append(item['index'])
            merged_selection.append(item)
    for item in selection_b:
        if item['index'] not in index_container:
            index_container.append(item['index'])
            merged_selection.append(item)
    return merged_selection

def load_raw_data(source_root, current_dataset):
    data_list = []
    # load source data
    df = pd.read_csv(os.path.join(source_root, f"{current_dataset}.tsv"), sep="\t")
    # add column names
    data_list.append(df.columns.tolist())
    # read data line by line
    for _, row in df.iterrows():
        data_list.append(row.tolist())
    return data_list


def generate_our_pd_dataset(custom_data, raw_data, suffix=""):
    saving_index_to_item = {}
    for item in custom_data:
        saving_index_to_item[item['index']] = item

    df_keys = raw_data[0]
    index_index = df_keys.index('index')
    answer_index = df_keys.index('answer')
    image_index = df_keys.index('image')

    pure_data_items = raw_data[1:]
    # all image mapping
    image_map = {item[index_index]: item[image_index] for item in pure_data_items}

    saving_items = []

    for i, item in enumerate(pure_data_items):
        if item[index_index] in saving_index_to_item:
            assert saving_index_to_item[item[index_index]]['answer'] == item[answer_index]
            # The image field can store the base64 encoded image or another question index (for saving space)
            if len(item[image_index]) <= 64:
                idx = int(item[image_index])
                assert idx in image_map and len(image_map[idx]) > 64
                item[image_index] = image_map[idx]
            saving_items.append(item)
    
    print(f"Data Type ({suffix}): Saving {len(saving_items)} out of {len(raw_data) - 1} raw items, expected to be {len(saving_index_to_item)}")

    df_data = {key: [] for key in df_keys}
    for i, item in enumerate(saving_items):
        for k, key in enumerate(df_keys):
            df_data[key].append(item[k])


    pd_dataset = pd.DataFrame(df_data)    
    return pd_dataset


def main(args):
    # load configs
    with open(args.config_path, 'r') as file:
        configs = yaml.safe_load(file)
    # process dataset one by one
    for current_dataset in DATASET_LIST:
        all_splits = get_biased_data(args, configs, current_dataset)
        
        all_data_val = all_splits['all_data_val']
        all_data_test = all_splits['all_data_test']
        vcf_data_val = all_splits['vcf_data_val']
        vcf_data_test = all_splits['vcf_data_test']
        tcf_data_val = all_splits['tcf_data_val']
        tcf_data_test = all_splits['tcf_data_test']
        filtered_data = all_splits['filtered_data']

        print(f"====== {current_dataset} ======")
        print(f"Splite {current_dataset} into Val={len(all_data_val)}, Test={len(all_data_test)}, all={len(all_data_val)}+{len(all_data_test)}={len(all_data_val)+len(all_data_test)}")

        biased_data_val = merge_selection(vcf_data_val, tcf_data_val)
        biased_data_test = merge_selection(vcf_data_test, tcf_data_test)
        print(f"Get biased {current_dataset} with Val-Visual={len(vcf_data_val)}, Test-Visual={len(vcf_data_test)}")
        print(f"Get biased {current_dataset} with Val-Textual={len(tcf_data_val)}, Test-Textual={len(tcf_data_test)}")
        print(f"Get biased {current_dataset} with Val-Combined={len(biased_data_val)}, Test-Combined={len(biased_data_test)}")

        
        print(f"Save filtered data and biased data")
        with open(os.path.join(args.source_root, f"{current_dataset}_{args.model_name}_filtered_data.json"), 'w') as f:
            json.dump(filtered_data, f)
        
        raw_data = load_raw_data(args.source_root, current_dataset)
        print(f"Loading raw {current_dataset}: {len(raw_data) - 1} items")

        # save custom all data
        if args.save_custom:
            pd_total_data_val = generate_our_pd_dataset(all_data_val, raw_data, suffix="total_val")
            pd_total_data_val.to_csv(os.path.join(args.source_root, f"{current_dataset}_Custom_Val.tsv"), sep='\t', index=False)
        
            pd_total_data_test = generate_our_pd_dataset(all_data_test, raw_data, suffix="total_test")
            pd_total_data_test.to_csv(os.path.join(args.source_root, f"{current_dataset}_Custom_Test.tsv"), sep='\t', index=False)

        # save biased data
        pd_vcf_data_val = generate_our_pd_dataset(vcf_data_val, raw_data, suffix="VisualCF_val")
        pd_vcf_data_val.to_csv(os.path.join(args.source_root, f"{current_dataset}_{args.model_name}_VCF_Val.tsv"), sep='\t', index=False)

        pd_vcf_data_test = generate_our_pd_dataset(vcf_data_test, raw_data, suffix="VisualCF_test")
        pd_vcf_data_test.to_csv(os.path.join(args.source_root, f"{current_dataset}_{args.model_name}_VCF_Test.tsv"), sep='\t', index=False)

        pd_tcf_data_val = generate_our_pd_dataset(tcf_data_val, raw_data, suffix="TextualCF_val")
        pd_tcf_data_val.to_csv(os.path.join(args.source_root, f"{current_dataset}_{args.model_name}_TCF_Val.tsv"), sep='\t', index=False)

        pd_tcf_data_test = generate_our_pd_dataset(tcf_data_test, raw_data, suffix="TextualCF_test")
        pd_tcf_data_test.to_csv(os.path.join(args.source_root, f"{current_dataset}_{args.model_name}_TCF_Test.tsv"), sep='\t', index=False)

        pd_biased_data_val = generate_our_pd_dataset(biased_data_val, raw_data, suffix="Biased_val")
        pd_biased_data_val.to_csv(os.path.join(args.source_root, f"{current_dataset}_{args.model_name}_Biased_Val.tsv"), sep='\t', index=False)

        pd_biased_data_test = generate_our_pd_dataset(biased_data_test, raw_data, suffix="Biased_test")
        pd_biased_data_test.to_csv(os.path.join(args.source_root, f"{current_dataset}_{args.model_name}_Biased_Test.tsv"), sep='\t', index=False)

    print("Finished")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--save-custom", action="store_true", default=False)
    args = parser.parse_args()
    main(args)


# python ./tools/generate_dataset.py --model-name xxx --config-path xxx --source-root xxx --save-custom