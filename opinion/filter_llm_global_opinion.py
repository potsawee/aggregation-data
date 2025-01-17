import json
import sys
import os
import argparse
import re
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

system_prompt = "You are a helpful assistant."
prompt_template = """
Evaluate if this survey question is appropriate for Thai workers who may have varying levels of education and exposure to global affairs. The question is:

"###QUESTION###"

Please evaluate using these criteria:
1. Universal Relevance: Does this question address experiences or phenomena that exist across cultures?
2. Required Knowledge: Does the question require specific knowledge about foreign politics, institutions, or events?
3. Observable Impact: Can respondents answer this based on their daily life experiences or provided context?
4. Cultural Transferability: Is the concept being asked about meaningful in a Thai context?
5. Time Sensitivity: Does the question rely on understanding specific current events?

For each criterion, provide a YES/NO and brief explanation.

Also, keep only questions that can be answered using a range scale (Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree), or binary scale (Yes/no).

Then provide a final verdict:
- KEEP: Question is appropriate as is
- MODIFY: Question could work with modifications (suggest how)
- REJECT: Question is inappropriate
If MODIFY, provide a suggested revision that maintains the core concept but makes it more appropriate for Thai workers.

First, provide a brief explaination of your evaluation, then outputing a paragraph break [VERDICT], before providing your final verdict.

Your final verdict must be in the following JSON format where revision is optional (only MODIFY verdict):
{
    "verdict": "KEEP/MODIFY/REJECT",
    "revision": "",
}

Don't include the answer scale (e.g., Strongly Disagree, Disagree, or Yes/No) in the revision. The revision needs to have just the question only. The final verdict will be parsed by json.loads(), so please do not include additional text and start the final verdict with "{" and end with "}".

""".strip()

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", bool, lambda v: v.lower() == "true")
    parser.add_argument('--judge_name', type=str, default="gpt-4o")
    parser.add_argument('--output_path', type=str, required=True) # output of inference
    return parser

def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    judge_name = kwargs['judge_name']
    output_path = kwargs['output_path']
    for k,v in kwargs.items():
        print(k, v)

    client = OpenAI()
    data = load_dataset("Anthropic/llm_global_opinions")["train"]
    print("len(data):", len(data))

    outputs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                x = json.loads(line)
                outputs.append(x)
        num_done = len(outputs)
    else:
        num_done = 0
    print("num_done = {}".format(num_done))

    for i in tqdm(range(num_done, len(data))):
        x = data[i]
        try:
            question = x['question']
            prompt = prompt_template.replace("###QUESTION###", question)
            response = client.chat.completions.create(
                model=judge_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.00001,
            )
            response = response.choices[0].message.content.strip()
            if "[VERDICT]" in response:
                explanation, verdict = response.split("[VERDICT]")
                explanation = explanation.strip()
                verdict = verdict.strip()
                if verdict.startswith('```json'):
                    verdict = verdict[7:-3].strip()

                verdict = verdict.strip('```')
                verdict = json.loads(verdict)
                output = {
                    "explanation": explanation,
                    "verdict": verdict,
                }
            else:
                print("Invalid JSON response")
                print("response:", response)

        except Exception as e:
            print("error:", e)
            output = {
                "explanation": "ERROR",
                "verdict": {"verdict": "ERROR", "revision": ""}
            }

        print(i, output)
        with open(output_path, 'a') as f:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    print("finish llm judge run")

if __name__ == "__main__":
    main()