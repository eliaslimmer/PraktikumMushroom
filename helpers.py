from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import evaluate
from datasets import load_dataset
import torch.nn.functional as F
import gc
import json
from tqdm import tqdm
from scipy.stats import spearmanr #For the spearman correlation computation




def prepare_dataset(language):
    # Load the MKQA dataset
    dataset = load_dataset("mkqa")
    train_set = dataset['train']

    # Initialize a dictionary to hold language-specific queries and answers
    language_data = {}

    # Extract queries and answers for the specified language
    language_queries = [item['queries'][language] for item in train_set]
    language_answers = [[answer['text'] for answer in item['answers'][language]] for item in train_set]

    # Store the extracted data in the dictionary
    language_data['queries'] = language_queries
    language_data['answers'] = language_answers
    
    return language_data



def generate_answers(model, tokenizer, lang_queries, language, file_path):
    batch_size = 4
    max_new_tokens = 64
    
    def system_msg(language):
        if language == 'en':
            return {
            "role": "system", 
            "content": "Answer concisely. Don't explain the background/reason."
            }
        if language == 'fr':
            return {
                "role": "system",
                "content": "Répondez de manière concise. N'expliquez pas le contexte ni la raison."
            }
        if language == 'de':
            return {
            "role": "system",
            "content": "Antworte kurzgefasst. Erkläre nicht den Hintergrund/Begründung."
            }
    
    model.to("cuda")
    output_scores = []
    output_list = []
    
    with open(file_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(lang_queries), batch_size), desc="Generating", total=(len(lang_queries) + batch_size - 1) // batch_size):
            batch = lang_queries[i:i+batch_size]
            batch_messages = [[system_msg(language), {"role": "user", "content":query}] for query in batch]
            batch_prompts = [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in batch_messages
            ]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,max_length=1024)
            inputs = inputs.to("cuda")
            

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens = max_new_tokens, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id)

            gen_only_ids = output_ids.sequences[:, inputs.input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)

            selected_token_probs_batch = []
            for step, scores in enumerate(output_ids.scores):
                probs = torch.softmax(scores, dim=-1)
                selected_probs = []
                for b in range(probs.size(0)):
                    token_index = inputs.input_ids.shape[1] + step
                    try:
                        token_id = output_ids.sequences[b][token_index].item()
                        prob = probs[b][token_id].item()
                    except IndexError:
                        prob = 0.0
                    selected_probs.append(prob)
                selected_token_probs_batch.append(selected_probs)

            for b_idx, (query, prompt_text, full_output) in enumerate(zip(batch, batch_prompts, decoded)):
                if full_output.startswith(prompt_text):
                    cleaned_output = full_output[len(prompt_text):].strip()
                else:
                    cleaned_output = full_output.strip()

                selected_probs = [selected_token_probs_batch[step][b_idx] for step in range(len(selected_token_probs_batch))]

                f.write(json.dumps({
                    "model_input": query,
                    "model_output_text": cleaned_output,
                    "token_scores": selected_probs
                    }, ensure_ascii=False) + "\n")

                output_list.append(cleaned_output)
                output_scores.append(selected_probs)

            del inputs, output_ids, decoded, selected_token_probs_batch
            gc.collect()
            torch.cuda.empty_cache()

def span_mapping(file_path, output_path, tokenizer):
    def find_hallucination_spans(text, probabilities, tokenizer, threshold=0.4):
        encoding = tokenizer(text, return_offsets_mapping = True, return_tensors="pt", truncation=True)
        offsets = encoding["offset_mapping"][0].tolist()
        hallucinated_spans = []
        current_span = {}
        current_span_num = 0
        for i, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if i >= len(probabilities):
                break
            prob = probabilities[i]
            if current_span_num > 0:
                if prob < threshold:
                    current_span['end'] = end
                    avg_probs = (current_span['prob'] * current_span_num + prob) / (current_span_num + 1)
                    current_span['prob'] = avg_probs
                    current_span_num += 1
                else:
                    avg_probs = (current_span['prob'] * current_span_num + prob) / (current_span_num + 1)
                    if avg_probs < threshold:
                        current_span['end'] = end
                        current_span['prob'] = avg_probs
                        current_span_num += 1
                    else:
                        hallucinated_spans.append(current_span)
                        current_span = {}
                        current_span_num = 0
            else:
                if prob < threshold:
                    current_span['start'] = start
                    current_span['end'] = end
                    current_span['prob'] = prob
                    current_span_num = 1
                else:
                    continue
            if current_span:
                    hallucinated_spans.append(current_span)
            return hallucinated_spans
    
    with open(file_path, "r", encoding = "utf-8") as f:
        data = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            text = item["model_output_text"]
            probabilities = item["token_scores"]
            spans = find_hallucination_spans(text, probabilities, tokenizer)
            item["soft_labels"] = spans
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def generate_hard_labels(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for item in data:
        text = item.get("model_output_text", "")
        text_len = len(text)
        soft_spans = item.get("soft_labels", [])
        
        if soft_spans is None:
            soft_spans = []
            
        soft_mask = [0] * text_len
        for span in soft_spans:
            for i in range(span["start"], span["end"]):
                if 0 <= i < text_len:
                    soft_mask[i] = 1

        hard_spans = []
        start = None
        for i, v in enumerate(soft_mask):
            if v == 0 and start is None:
                start = i
            elif v == 1 and start is not None:
                hard_spans.append([start, i])
                start = None
        if start is not None:  
            hard_spans.append([start, text_len])

        item["hard_labels"] = hard_spans

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def validation_reformat(file_path:str, output_path:str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    converted = []
    for item in data:
        converted.append({
            "id": item.get("id"),
            "model_output_text": item.get("model_output_text"),
            "hard_labels": item.get("hard_labels"),
            "soft_labels": item.get("soft_labels")
        })
    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def tokenize_and_map_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['model_output_text'],
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    offset_mappings = tokenized_inputs['offset_mapping']
    all_labels = examples['hard_labels']
    tok_labels_batch = []

    for batch_idx in range(len(offset_mappings)):
        offset_mapping = offset_mappings[batch_idx]
        hard_labels = all_labels[batch_idx]
        tok_labels = [0] * len(offset_mapping)
        for idx, (start, end) in enumerate(offset_mapping):
            for label_start, label_end in hard_labels:
                if start >= label_start and end <= label_end:
                    tok_labels[idx] = 1
        tok_labels_batch.append(tok_labels)

    tokenized_inputs['labels'] = tok_labels_batch
    return tokenized_inputs

def train_model(MODEL_NAME, LABEL_LIST, data_files, output_dir="./results_xlmr"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)

    raw_dataset = load_dataset("json", data_files=data_files)

    def keep_relevant(example):
        return {
            "model_output_text": example["model_output_text"],
            "hard_labels": example["hard_labels"],
            "soft_labels": example["soft_labels"]
        }

    dataset = raw_dataset.map(keep_relevant)

    tokenized_datasets = dataset.map(lambda x: tokenize_and_map_labels(x, tokenizer), batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=2, #Less GPU usage
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy='epoch',
        push_to_hub=False,
        report_to='none',
        fp16=True #Less GPU usage
    )

    metric = evaluate.load('seqeval')

    def compute_metrics(p):
      predictions = torch.argmax(torch.tensor(p.predictions), dim=2).tolist()
      labels = p.label_ids.tolist()

      true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
      true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
      ]

      results = metric.compute(predictions=true_predictions, references=true_labels)
      return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
      }
        
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_dir)
    print(f"Training completed. Checkpoints saved to: {output_dir}")

def test_model(MODEL_NAME, test_lang, model_path, data_path):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    # Load the test dataset
    test_dataset = load_dataset('json', data_files={'test': data_path})['test'] #Change the file name
    # Tokenize test dataset
    inputs = tokenizer(test_dataset['model_output_text'], padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")

    # Get predictions for the test set
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    preds = torch.argmax(outputs.logits, dim=2)
    probs = F.softmax(outputs.logits, dim=2)
    # map predictions to character spans
    hard_labels_all = {}
    soft_labels_all = {}
    predictions_all = []
    for i, pred in enumerate(preds):
        hard_labels_sample = []
        soft_labels_sample = []
        positive_indices = torch.nonzero(pred == 1, as_tuple=False)
        offset_mapping = inputs['offset_mapping'][i]
        for j, offset in enumerate(offset_mapping):
            if offset[0].item() < offset[1].item():
                soft_labels_sample.append({'start': offset[0].item(), 'end': offset[1].item(), 'prob': probs[i][j][1].item()})
            if j in positive_indices:
                if offset[0].item() < offset[1].item():
                    hard_labels_sample.append((offset[0].item(), offset[1].item()))
        soft_labels_all[test_dataset['id'][i]] = soft_labels_sample
        hard_labels_all[test_dataset['id'][i]] = hard_labels_sample
        predictions_all.append({'id': test_dataset['id'][i], 'hard_labels': hard_labels_sample, 'soft_labels': soft_labels_sample})
    with open(f"{test_lang}-hard_labels.json", 'w') as f:
        json.dump(hard_labels_all, f)
    with open(f"{test_lang}-soft_labels.json", 'w') as f:
        json.dump(soft_labels_all, f)
    with open(f"{test_lang}-pred.jsonl", 'w') as f:
        for pred_dict in predictions_all:
            print(json.dumps(pred_dict), file=f)
    print(f"Labels saved to {test_lang}-hard_labels.json and {test_lang}-soft_labels.json")
    print(f"Prediction file saved to {test_lang}-pred.jsonl")

def iou(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0.0

def average_iou(preds, gts):
    ious = []
    for gt_span in gts:
        max_iou = max([iou(gt_span, pred_span) for pred_span in preds], default=0.0)
        ious.append(max_iou)
    return sum(ious) / len(ious) if ious else 0.0

def span_to_dict(span_list):
    return {(s['start'], s['end']): s['prob'] for s in span_list}

def spearman_score(soft_pred, soft_gt):
    pred_dict = {(s['start'], s['end']): s['prob'] for s in soft_pred}
    gt_dict = {(s['start'], s['end']): s['prob'] for s in soft_gt}
    
    common_keys = list(set(pred_dict.keys()) & set(gt_dict.keys()))
    if not common_keys:
        return None

    pred_scores = [pred_dict[k] for k in common_keys]
    gt_scores = [gt_dict[k] for k in common_keys]

    if len(set(gt_scores)) <= 1 or len(set(pred_scores)) <= 1:
        return None #if all the probabilities are same then it doesn't work

    corr, _ = spearmanr(pred_scores, gt_scores)
    return corr

def evaluate_predictions(predictions, hard_references, soft_references):
    all_ious = []
    all_spearman = []

    for sample_pred in predictions:
        sample_id = sample_pred['id']
        pred_spans = sample_pred['hard_labels']
        soft_pred = sample_pred['soft_labels']

        gt_spans = hard_references[sample_id]
        soft_gt = soft_references[sample_id]

        iou_score = average_iou(pred_spans, gt_spans)
        all_ious.append(iou_score)

        spearman = spearman_score(soft_pred, soft_gt)
        if spearman is not None:
            all_spearman.append(spearman)

    return {
        "mean_iou": sum(all_ious) / len(all_ious),
        "mean_spearman": sum(all_spearman) / len(all_spearman)
    }

def convert_reference_to_dict(ref_list):
    hard = {}
    soft = {}
    for sample in ref_list:
        sid = sample["id"]
        hard[sid] = sample["hard_labels"]
        soft[sid] = sample["soft_labels"]
    return hard, soft
