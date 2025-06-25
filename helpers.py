from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from datasets import load_dataset

import gc
import json
from tqdm import tqdm



def load_dataset(language):
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

    # Print the length of the dataset and a sample query and answer
    print(f"Length of {language} dataset: {len(language_data['queries'])}, "
        f"sample {language} query: {language_data['queries'][0]}, "
        f"sample {language} answer: {', '.join(language_data['answers'][0])}")



def generate_answers(model, tokenizer, lang_queries):
    batch_size = 4
    max_new_tokens = 64

    system_msg = {
        "role": "system", 
        "content": "Answer concisely. Don't explain the background/reason."
    }

    output_scores = []
    output_list = []
    with open("outputs_partial.jsonl", "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(lang_queries), batch_size), desc="Generating", total=(len(lang_queries) + batch_size - 1) // batch_size):
            batch = lang_queries[i:i+batch_size]
            batch_messages = [[system_msg, {"role": "user", "content":query}] for query in batch]
            batch_prompts = [
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
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