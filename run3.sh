#!/bin/bash
#$ -S /bin/bash

# Set up your working dir here
export CUDA_VISIBLE_DEVICES=4,5
# conda activate exp-pp1
export HF_DATASETS_CACHE=/cache/.cache/huggingface/datasets
export HF_HOME=/cache/.cache/huggingface
export HF_HUB_CACHE=/cache/.cache/huggingface/hub


# 1) Llama-3-70B
# python llm_judge_mmlu.py --judge_name meta-llama/Meta-Llama-3-70B-Instruct --output_path outputs/mmlu/Meta-Llama-3-70B-Instruct.jsonl

# 3) Qwen2-72B
python llm_judge_mmlu.py --judge_name Qwen/Qwen2-72B-Instruct --output_path outputs/mmlu/Qwen2-72B-Instruct.jsonl

# # 5) Athene-70B
python llm_judge_mmlu.py --judge_name Nexusflow/Athene-70B --output_path outputs/mmlu/Athene-70B.jsonl

# 4) Hermes3-70B
python llm_judge_mmlu.py --judge_name NousResearch/Hermes-3-Llama-3.1-70B --output_path outputs/mmlu/Hermes-3-Llama-3.1-70B.jsonl


# 2) Mixtral
python llm_judge_mmlu.py --judge_name mistralai/Mixtral-8x7B-Instruct-v0.1 --output_path outputs/mmlu/Mixtral-8x7B-Instruct-v0.1.jsonl

# 6) Dolphin-mixtral
python llm_judge_mmlu.py --judge_name cognitivecomputations/dolphin-2.5-mixtral-8x7b --output_path outputs/mmlu/dolphin-2.5-mixtral-8x7b.jsonl

# ----------------------------------------------------- #

# 1) Llama-3-70B
python llm_judge_chatbot_arena_reverse.py --judge_name meta-llama/Meta-Llama-3-70B-Instruct --output_path outputs/lmsys-chatbot-arena-train-reverse/Meta-Llama-3-70B-Instruct.jsonl

# 4) Hermes3-70B
python llm_judge_chatbot_arena_reverse.py --judge_name NousResearch/Hermes-3-Llama-3.1-70B --output_path outputs/lmsys-chatbot-arena-train-reverse/Hermes-3-Llama-3.1-70B.jsonl

# # 5) Athene-70B
python llm_judge_chatbot_arena_reverse.py --judge_name Nexusflow/Athene-70B --output_path outputs/lmsys-chatbot-arena-train-reverse/Athene-70B.jsonl
