import os
import sys
from datasets import load_from_disk

sys.path.append(f"{os.getenv('HOME')}/{os.getenv('PROJECT_DIR')}/llm-distillation")
from prompt.prompt import create_chat_prompt
from prompt.prompt import create_prompt

def tokenize(item, tokenizer, encoder_decoder=False):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "summary_news"

    if tokenizer.name_or_path == "meta-llama/Llama-2-7b-chat-hf":
        shot = 2
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
        shot = 2
    elif tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
        shot = 2
    elif tokenizer.name_or_path == "meta-llama/Meta-Llama-3-70B-Instruct":
        shot = 2

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            context = item['context'],
            sys_user = True if "mistralai/Mistral-7B-Instruct-v0.2" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            context = item['context'],
        )

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    if not encoder_decoder:
        if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
            context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
            if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
                answer_tokens = tokenizer.encode(f" {item['summary_generated']}", add_special_tokens=False)
            else:
                answer_tokens = tokenizer.encode(f"{item['summary_generated']}", add_special_tokens=False)
        else:
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
            answer_tokens = tokenizer.encode(f" {item['summary_generated']}{tokenizer.eos_token}", add_special_tokens=False)

        prompt_tokens = context_tokens+answer_tokens if len(item['summary_generated']) > 0 else [-200]
        labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

        combined_tokens = {
            "input_ids": prompt_tokens,
            "labels": labels_tokens
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(item['summary_generated'], add_special_tokens=True, return_tensors="pt")[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1]*len(input_ids)
        }

def get_split(dataset_config, tokenizer, split):
    dataset = load_from_disk(f"/project/yzhao010_1246/llm-hallucination/generated/{dataset_config.generated_by.split('/')[-1]}/cnn_dailymail/train/{split}")
    # dataset = load_from_disk(f"{os.getenv('HOME')}/{os.getenv('PROJECT_DIR')}/llm-distillation-test/datasets/generated/Llama-2-7b-chat-hf/cnn_dailymail/{split}")
    if dataset_config.training_size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.training_size)))
    dataset = dataset.map(lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder), remove_columns=list(dataset.features))
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 1)
    return dataset