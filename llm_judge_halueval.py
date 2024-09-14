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

system_prompt = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the true entity in the response is replaced with a highly similar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: [[No]]
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Steven Spielberg was the director. He also directed insomnia and inception.
#Your Judgement#: [[Yes]]

You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: [[No]]
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Batman Begins was the director. He also directed insomnia and inception.
#Your Judgement#: [[Yes]]

You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: [[No]]
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: United States of America was the director. He also directed insomnia and inception.
#Your Judgement#: [[Yes]]

You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be [[Yes]] or [[No]]."""

prompt_template = """#Dialogue History#: {dislogue_history}\n#Response#: {response}\n#Your Judgement#: [["""
partial_answer = """#Your Judgement#: [["""

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
    file_path = 'halueval/dialogue_data.jsonl'
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the line as JSON and add it to the list
            data.append(json.loads(line))

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
    for x in ["Yes", "No"]:
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
        x = data[i]
        dialogue_history = x['dialogue_history']

        this_example = [
            {"response": x['right_response'], "label": "no"}, # does it have hallucination?
            {"response": x['hallucinated_response'], "label": "yes"}, # does it have hallucination?
        ]

        ex_outputs = []
        for ex in this_example:
            model_response = ex['response']
            label = ex['label']

            prompt = prompt_template.format(dislogue_history=dialogue_history, response=model_response)
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
            logit_Yes = last_logits[abc_mapping['Yes']].tolist()
            logit_No  = last_logits[abc_mapping['No']].tolist()

            if logit_Yes > logit_No:
                pred = "yes"
            else:
                pred = "no"

            if pred == label:
                correct += 1
            total += 1
            if i % 100 == 0:
                print("===> [{}] Accuracy: {:.2f}".format(judge_name, correct/total*100))

            ex_outputs.append({
                'response': model_response,
                'pred': pred,
                'ref': label,
                'logit_yes': logit_Yes,
                'logit_no': logit_No
            })

        item = {
            'i': i,
            'llm_judge': judge_name,
            'dialogue_history': dialogue_history,
            'outputs': ex_outputs,
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