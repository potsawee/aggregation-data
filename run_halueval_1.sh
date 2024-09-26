#!/bin/bash
#$ -S /bin/bash

# Set up your working dir here
export CUDA_VISIBLE_DEVICES=1
# conda activate exp-pp1
export TRANSFORMERS_CACHE=/scratch/cache/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/cache/.cache/huggingface/datasets


# 1) Llama-3-70B
python llm_judge_halueval.py --judge_name meta-llama/Meta-Llama-3-70B-Instruct --output_path outputs/halueval/Meta-Llama-3-70B-Instruct.jsonl

# 2) Mixtral
# python llm_judge_halueval.py --judge_name mistralai/Mixtral-8x7B-Instruct-v0.1 --output_path outputs/halueval/Mixtral-8x7B-Instruct-v0.1.jsonl

# 3) Qwen2-72B
# python llm_judge_halueval.py --judge_name Qwen/Qwen2-72B-Instruct --output_path outputs/halueval/Qwen2-72B-Instruct.jsonl

# 4) Hermes3-70B
# python llm_judge_halueval.py --judge_name NousResearch/Hermes-3-Llama-3.1-70B --output_path outputs/halueval/Hermes-3-Llama-3.1-70B.jsonl

# # 5) Athene-70B
# python llm_judge_halueval.py --judge_name Nexusflow/Athene-70B --output_path outputs/halueval/Athene-70B.jsonl

# 6) Dolphin-mixtral
# python llm_judge_halueval.py --judge_name cognitivecomputations/dolphin-2.5-mixtral-8x7b --output_path outputs/halueval/dolphin-2.5-mixtral-8x7b.jsonl





