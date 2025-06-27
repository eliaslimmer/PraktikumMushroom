from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import evaluate
from datasets import load_dataset
import torch.nn.functional as F
import gc
import json
from tqdm import tqdm
from scipy.stats import spearmanr #For the spearman correlation computation



def convert_reference_to_dict(ref_list):
    hard = {}
    soft = {}
    for sample in ref_list:
        sid = sample["id"]
        hard[sid] = sample["hard_labels"]
        soft[sid] = sample["soft_labels"]
    return hard, soft


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
