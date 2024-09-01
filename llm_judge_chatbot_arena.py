"""
template from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl
{
    "name": "pair-v2", 
    "type": "pairwise", 

    "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", 
    
    "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", 
    
    "description": "Prompt for general questions", 
    "category": "general", 
    "output_format": "[[A]]"

}
"""

import json
import sys
import os
import argparse
import re
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Do not provide any explanation, please provide your final verdict after \"Verdict:\" by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."""

prompt_template = """[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"""

partial_answer = """Verdict: [["""

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
    data_path = "/workspace/aggregation-data/lmsys-chatbot-arena/train.single-turn.json"
    with open(data_path) as f:
        data = json.load(f)

    # Load model directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(judge_name)
    # model = AutoModelForCausalLM.from_pretrained(judge_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    # model.to(device)

    model = AutoModelForCausalLM.from_pretrained(
        judge_name, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    abc_mapping = {}
    for x in ["A", "B", "C"]:
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

    correct, total, bad = 0, 0, 0

    for i in tqdm(range(num_done, len(data))):
        x = data[i]
        question = x['question']
        answer_a = x['answer_a']
        answer_b = x['answer_b']
        prompt = prompt_template.format(question=question, answer_a=answer_a, answer_b=answer_b)

        assert x['winner_model_a'] + x['winner_model_b'] + x['winner_tie'] == 1
        if x['winner_model_a'] == 1:
            winner_gt = "A"
        elif x['winner_model_b'] == 1:
            winner_gt = "B"
        elif x['winner_tie'] == 1:
            winner_gt = "tie"

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": partial_answer}
        ]

        messages_with_special_tokens = tokenizer.apply_chat_template(messages, tokenize=False)
        ii_ = messages_with_special_tokens.find(partial_answer)
        messages_with_special_tokens = messages_with_special_tokens[:ii_] +  partial_answer

        # try:

        encodeds = tokenizer(messages_with_special_tokens, return_tensors="pt", add_special_tokens=False)
        model_inputs = encodeds.to(device)

        # generation
        # generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
        # decoded = tokenizer.batch_decode(generated_ids)

        # simple forward pass
        last_logits = model(**model_inputs)['logits'][0, -1] # [batch_size, num_tokens, vocab_size]
        logit_A = last_logits[abc_mapping['A']].tolist()
        logit_B = last_logits[abc_mapping['B']].tolist()
        logit_C = last_logits[abc_mapping['C']].tolist()
        # except:
        #     logit_A, logit_B, logit_C = 5, 5, 5

        if logit_A > logit_B and logit_A > logit_C:
            winner_pred = "A"
        elif logit_B > logit_A and logit_B > logit_C:
            winner_pred = "B"
        elif logit_C > logit_A and logit_C > logit_B:
            winner_pred = "tie"
        else:
            print("Warning: bad values [{:.2f}]".format(bad/(total+1)*100))
            bad += 1
            winner_pred = "tie"

        if winner_gt ==  winner_pred:
            correct += 1
        total += 1
        if i % 100 == 0:
            print("===> [{}] Accuracy: {:.2f}".format(judge_name, correct/total*100))

        item = {
            'id': x['id'],
            'llm_judge': judge_name,
            'winner_gt': winner_gt,
            'winner_pred': winner_pred,
            'logit_A': logit_A,
            'logit_B': logit_B,
            'logit_C': logit_C
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