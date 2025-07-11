import torch
from datasets import load_dataset
import gc
import json
from tqdm import tqdm



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
            item.pop("model_input", None)
            item.pop("token_scores", None)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def validation_reformat(file_path:str, output_path:str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    converted = []
    for item in data:
        converted.append({
            "model_output_text": item.get("model_output_text"),
            "hard_labels": item.get("hard_labels"),
            "soft_labels": item.get("soft_labels")
        })
    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
