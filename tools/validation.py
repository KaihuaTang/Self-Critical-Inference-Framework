import os
import statistics
from tqdm import tqdm
import torch
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer

DATASET_LIST = ["MMStar", "MME", "ViLP", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11"]
MODEL_LIST = ["Original", "TCF-V1", "TCF-V2", "VCF-Color0", "VCF-Noise500"]
#MODEL_LIST = ["Original", "TCF-V1", "TCF-V2", "TCF-V3", "VCF-Color0", "VCF-Color255", "VCF-Noise400", "VCF-Noise500"]

# temperature
# small variation

def confidence(logits, type='constant'):
    if type == 'max':
        return logits.unsqueeze(0).softmax(-1).max(-1).values
    elif type == 'entropy':
        prob = logits.unsqueeze(0).softmax(-1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        return entropy
    elif type == 'neg_energy':
        return torch.logsumexp(logits.unsqueeze(0), dim=-1)
    elif type == 'constant':
        return 1.0


def get_consistent_logits_type2(orig_logits, tcf1_logits, alpha, gamma, type="max"):
    if type == "max":
        consistent_logits = torch.cat([orig_logits.unsqueeze(0) * confidence(orig_logits / alpha),
                                    tcf1_logits.unsqueeze(0) * confidence(tcf1_logits / alpha)], dim=0).max(0).values 
        return consistent_logits / gamma
    elif type == "add":
        consistent_logits = orig_logits.unsqueeze(0) * confidence(orig_logits / alpha) + \
                            tcf1_logits.unsqueeze(0) * confidence(tcf1_logits / alpha)
        return consistent_logits / gamma


def get_consistent_logits_type4(orig_logits, tcf1_logits, tcf2_logits, tcf3_logits, alpha, gamma, type="max"):
    if type == "max":
        consistent_logits = torch.cat([orig_logits.unsqueeze(0) * confidence(orig_logits / alpha),
                                    tcf1_logits.unsqueeze(0) * confidence(tcf1_logits / alpha),
                                    tcf2_logits.unsqueeze(0) * confidence(tcf2_logits / alpha),
                                    tcf3_logits.unsqueeze(0) * confidence(tcf3_logits / alpha)], dim=0).max(0).values 
        return consistent_logits / gamma
    elif type == "add":
        consistent_logits = orig_logits.unsqueeze(0) * confidence(orig_logits / alpha) + \
                            tcf1_logits.unsqueeze(0) * confidence(tcf1_logits / alpha) + \
                            tcf2_logits.unsqueeze(0) * confidence(tcf2_logits / alpha) + \
                            tcf3_logits.unsqueeze(0) * confidence(tcf3_logits / alpha)
        return consistent_logits / gamma

def get_consistent_logits_type3(orig_logits, tcf1_logits, tcf2_logits, alpha, gamma, type="max"):
    if type == "max":
        consistent_logits = torch.cat([orig_logits.unsqueeze(0) * confidence(orig_logits / alpha),
                                    tcf1_logits.unsqueeze(0) * confidence(tcf1_logits / alpha),
                                    tcf2_logits.unsqueeze(0) * confidence(tcf2_logits / alpha)], dim=0).max(0).values 
        return consistent_logits / gamma
    elif type == "add":
        consistent_logits = orig_logits.unsqueeze(0) * confidence(orig_logits / alpha) + \
                            tcf1_logits.unsqueeze(0) * confidence(tcf1_logits / alpha) + \
                            tcf2_logits.unsqueeze(0) * confidence(tcf2_logits / alpha)
        return consistent_logits / gamma
    elif type == "inv1":
        #  inverse-variance weighting
        mean_logits = (orig_logits + tcf1_logits + tcf2_logits) / 3.0
        consistent_logits = orig_logits.unsqueeze(0) * ( - ((orig_logits - mean_logits) ** 2).mean() / alpha).exp() + \
                            tcf1_logits.unsqueeze(0) * ( - ((tcf1_logits - mean_logits) ** 2).mean() / alpha).exp() + \
                            tcf2_logits.unsqueeze(0) * ( - ((tcf2_logits - mean_logits) ** 2).mean() / alpha).exp()
        return consistent_logits / gamma
    elif type == "inv2":
        mean_logits = (orig_logits + tcf1_logits + tcf2_logits) / 3.0
        consistent_logits = orig_logits.unsqueeze(0) / alpha / (((orig_logits - mean_logits) ** 2).mean().item() + 1e-10) + \
                            tcf1_logits.unsqueeze(0) / alpha / (((tcf1_logits - mean_logits) ** 2).mean().item() + 1e-10) + \
                            tcf2_logits.unsqueeze(0) / alpha / (((tcf2_logits - mean_logits) ** 2).mean().item() + 1e-10)
        return consistent_logits / gamma
    elif type == "variance":
        mean_logits = (orig_logits + tcf1_logits + tcf2_logits) / 3.0
        consistent_logits = ((orig_logits.unsqueeze(0) - mean_logits.unsqueeze(0)) / alpha) ** 2 + \
                            ((tcf1_logits.unsqueeze(0) - mean_logits.unsqueeze(0)) / alpha) ** 2 + \
                            ((tcf2_logits.unsqueeze(0) - mean_logits.unsqueeze(0)) / alpha) ** 2
        return gamma / (consistent_logits / 3.0 + 1e-5)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    all_gt_tokens = []
    all_pred_logits = {model_type : [] for model_type in MODEL_LIST}
    
    for dataset_type in DATASET_LIST:
        # load base results
        result_path = os.path.join(args.result_path, f"{args.model_name}-Original", f"{args.result_folder}", f"{args.model_name}-Original_{dataset_type}_{args.split_name}.xlsx")
        result_df = pd.read_excel(result_path)

        num_samples = len(os.listdir(os.path.join(args.logit_path, f"{args.model_name}-Original", f"{dataset_type}_{args.split_name}")))
        # load all logits and gt tokens
        for index in tqdm(range(num_samples)):
            gt = result_df.iloc[index]['answer']
            gt_tokens = [tokenizer(gt).input_ids[-1], tokenizer(gt.lower()).input_ids[-1], tokenizer(gt.upper()).input_ids[-1], tokenizer(gt.capitalize()).input_ids[-1]]
            #gt_tokens = [tokenizer(gt).input_ids[0], tokenizer(gt.lower()).input_ids[0], tokenizer(gt.upper()).input_ids[0], tokenizer(gt.capitalize()).input_ids[0]]
            all_gt_tokens.append(gt_tokens)
            for model_type in MODEL_LIST:
                logits = torch.load(os.path.join(args.logit_path, f"{args.model_name}-{model_type}", f"{dataset_type}_{args.split_name}", f"logits_{index}.pt")).float().cuda()
                all_pred_logits[model_type].append(logits[0])
    
    print(f"Samples with gt tokens: {len(all_gt_tokens)}")
    for model_type in MODEL_LIST:
        print(f"{args.model_name}_{model_type} samples: {len(all_pred_logits[model_type])}")
        assert len(all_pred_logits[model_type]) == len(all_gt_tokens)

    best_hyper = []
    best_acc = 0.0
    #for beta in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for beta in [0.2]:
        #for alpha in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for alpha in [1.0]:
            #for gamma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4]:
            for gamma in [1.5]:
                #for theta in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
                for theta in [0.3]:
                    accuracy = []
                    for i in range(len(all_gt_tokens)):
                        orig_logits = all_pred_logits["Original"][i]
                        tcf1_logits = all_pred_logits["TCF-V1"][i]
                        tcf2_logits = all_pred_logits["TCF-V2"][i]
                        #tcf3_logits = all_pred_logits["TCF-V3"][i]
                        vcf1_logits = all_pred_logits["VCF-Color0"][i]
                        vcf2_logits = all_pred_logits["VCF-Noise500"][i]
                        #vcf3_logits = all_pred_logits["VCF-Noise400"][i]
                        #vcf3_logits = all_pred_logits["VCF-Color255"][i]
                        consistent_logits = get_consistent_logits_type3(orig_logits, tcf1_logits, tcf2_logits, alpha, gamma)
                        unbiased_weights = (orig_logits - (vcf1_logits + vcf2_logits) / 2.0) / beta
                        
                        #consistent_logits = get_consistent_logits_type2(orig_logits, tcf1_logits, alpha, gamma)
                        #unbiased_weights = (orig_logits - vcf1_logits) / beta

                        #consistent_logits = get_consistent_logits_type4(orig_logits, tcf1_logits, tcf2_logits, tcf3_logits, alpha, gamma)
                        #unbiased_weights = (orig_logits - (vcf1_logits + vcf2_logits + vcf3_logits) / 3.0) / beta

                        #final_logits = orig_logits
                        #final_logits = consistent_logits
                        #final_logits = orig_logits + unbiased_weights
                        #final_logits = orig_logits - vcf2_logits # TIE
                        #final_logits = (1 + beta) * orig_logits - beta * vcf2_logits # VCD
                        #final_logits = orig_logits + (1 - beta) / beta * (orig_logits - vcf2_logits) # M3ID
                        #cutoff = torch.log(torch.tensor([theta])).to(orig_logits.device) + orig_logits.max(dim=-1, keepdim=True).values
                        #final_logits = final_logits.masked_fill(orig_logits < cutoff, -float("inf"))

                        final_logits = consistent_logits + unbiased_weights
                        cutoff = torch.log(torch.tensor([theta])).to(consistent_logits.device) + consistent_logits.max(dim=-1, keepdim=True).values
                        final_logits = final_logits.masked_fill(consistent_logits < cutoff, -float("inf"))

                        final_pred = final_logits.max(-1).indices.item()
                        accuracy.append(float(final_pred in all_gt_tokens[i]))
                    if sum(accuracy) / len(accuracy) > best_acc:
                        best_acc = sum(accuracy) / len(accuracy)
                        best_hyper = [(beta, alpha, gamma, theta)]
                        print(f"Update best hyper-parameters. beta: {beta}; alpha: {alpha}; gamma: {gamma}; theta: {theta}; accuracy: {best_acc} ({sum(accuracy)} / {len(accuracy)})")
                    elif sum(accuracy) / len(accuracy) == best_acc:
                        best_hyper.append((beta, alpha, gamma, theta))
    print(f"==========================")
    print(f"Best hyper-parameters (beta, alpha, gamma, theta) has {len(best_hyper)} groups with accuracy: {best_acc}")
    print(f"Best hyper-parameters (beta, alpha, gamma, theta): {best_hyper}")
    print(f"Finished")


    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--split-name", type=str, required=True)
    parser.add_argument("--logit-path", type=str, required=True)
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--result-folder", default="T20250715_G2dd739c9", type=str)
    #parser.add_argument("--tokenizer-path", default="/home/couser/checkpoints/Qwen2-VL-7B-Instruct", type=str)
    parser.add_argument("--tokenizer-path", default="/home/couser/checkpoints/llama3-llava-next-8b-hf", type=str)
    args = parser.parse_args()
    main(args)


# python ./tools/validation.py --model-name xxx --split-name xxx --logit-path xxx --result-path xxx

# python tools/validation.py --model-name Qwen2-VL-7B --split-name Qwen2-VL-7B_TCF_Val --logit-path ./dump_tensors/ --result-path ./outputs_val/