import json
import sys
import os
import argparse
import re
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

system_prompt = """You are a helpful assistant and your task is to answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]. The response must stricly follow this format: Answer: [[x]] where x is the letter answer such as A, B, C, D."""

prompt_template = """Question: {question}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n

Answer: [["""

partial_answer = """Answer: [["""

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", bool, lambda v: v.lower() == "true")
    parser.add_argument('--judge_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True) # output of LLM judge -- jsonl
    return parser

def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    judge_name = kwargs['judge_name']
    output_path = kwargs['output_path']
    for k,v in kwargs.items():
        print(k, v)

    # data_path 
    data = load_dataset("cais/mmlu", "all", split='test')
    assert len(data) == 14042

    # Load model directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(judge_name)
    model = AutoModelForCausalLM.from_pretrained(
        judge_name, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    abc_mapping = {}
    for x in ["A", "B", "C", "D"]:
        mystr = f"[[{x}"
        tokens = tokenizer(mystr, add_special_tokens=False)['input_ids']
        assert len(tokens) == 2 # [[ and x
        abc_mapping[x] = tokens[-1]


    if os.path.exists(output_path):
        temp = []
        with open(output_path, "r") as f:
            for line in f:
                x = json.loads(line)
                temp.append(x)
        num_done = len(temp)
    else:
        num_done = 0
    print("start from IDX = {}".format(num_done))

    correct, total = 0, 0

    for i in tqdm(range(num_done, len(data))):
        x = data[i] # ['question', 'subject', 'choices', 'answer']
        prompt = prompt_template.format(
            question=x['question'],
            a=x['choices'][0],
            b=x['choices'][1],
            c=x['choices'][2],
            d=x['choices'][3]
        )
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        messages_with_special_tokens = tokenizer.apply_chat_template(messages, tokenize=False)
        ii_ = messages_with_special_tokens.rfind(partial_answer) # rfind --> last index
        messages_with_special_tokens = messages_with_special_tokens[:ii_] +  partial_answer

        encodeds = tokenizer(messages_with_special_tokens, return_tensors="pt", add_special_tokens=False)
        model_inputs = encodeds.to(device)

        # generation
        # generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
        # decoded = tokenizer.batch_decode(generated_ids)

        # simple forward pass
        last_logits = model(**model_inputs)['logits'][0, -1] # [batch_size, num_tokens, vocab_size]
        logit_a = last_logits[abc_mapping['A']].tolist()
        logit_b = last_logits[abc_mapping['B']].tolist()
        logit_c = last_logits[abc_mapping['C']].tolist()
        logit_d = last_logits[abc_mapping['D']].tolist()

        logits_arr = [logit_a, logit_b, logit_c, logit_d]
        pred = int(np.argmax(logits_arr))
        label = x['answer']

        if pred == label:
            correct += 1
        total += 1
        
        if i % 100 == 0:
            print("===> [{}] Accuracy: {:.2f}".format(judge_name, correct/total*100))

        item = {
            'i': i,
            'llm_judge': judge_name,
            'logits': logits_arr,
            'pred': pred,
            'label': label
        }
        with open(output_path, 'a') as f:
            f.write(json.dumps(item) + '\n')

    print("finish llm judge run")
    print("deleting cached model...")
    model_org, model_name = judge_name.split("/")
    cache_path = f"/cache/.cache/huggingface/hub/models--{model_org}--{model_name}"
    shutil.rmtree(cache_path)
    print("deleted cached model:", cache_path)
    
if __name__ == "__main__":
    with torch.no_grad():
        main()