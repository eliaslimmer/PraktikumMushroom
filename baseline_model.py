from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import evaluate
from datasets import load_dataset
import torch.nn.functional as F
import json


'''adapted by https://github.com/Helsinki-NLP/mu-shroom/blob/main/participant_kit/baseline_model.py'''


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
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
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
