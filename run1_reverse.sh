#!/bin/bash
#$ -S /bin/bash

# Set up your working dir here
export CUDA_VISIBLE_DEVICES=0,7
# conda activate exp-pp1
export HF_DATASETS_CACHE=/cache/.cache/huggingface/datasets
export HF_HOME=/cache/.cache/huggingface
export HF_HUB_CACHE=/cache/.cache/huggingface/hub


# 3) Qwen2-72B
python llm_judge_chatbot_arena_reverse.py --judge_name Qwen/Qwen2-72B-Instruct --output_path outputs/lmsys-chatbot-arena-train-reverse/Qwen2-72B-Instruct.jsonl

# 1) Llama-3-70B
# python llm_judge_chatbot_arena_reverse.py --judge_name meta-llama/Meta-Llama-3-70B-Instruct --output_path outputs/lmsys-chatbot-arena-train-reverse/Meta-Llama-3-70B-Instruct.jsonl

# 2) Mixtral
python llm_judge_chatbot_arena_reverse.py --judge_name mistralai/Mixtral-8x7B-Instruct-v0.1 --output_path outputs/lmsys-chatbot-arena-train-reverse/Mixtral-8x7B-Instruct-v0.1.jsonl

# 4) Hermes3-70B
# python llm_judge_chatbot_arena_reverse.py --judge_name NousResearch/Hermes-3-Llama-3.1-70B --output_path outputs/lmsys-chatbot-arena-train-reverse/Hermes-3-Llama-3.1-70B.jsonl

# # 5) Athene-70B
# python llm_judge_chatbot_arena_reverse.py --judge_name Nexusflow/Athene-70B --output_path outputs/lmsys-chatbot-arena-train-reverse/Athene-70B.jsonl

# 6) Dolphin-mixtral
python llm_judge_chatbot_arena_reverse.py --judge_name cognitivecomputations/dolphin-2.5-mixtral-8x7b --output_path outputs/lmsys-chatbot-arena-train-reverse/dolphin-2.5-mixtral-8x7b.jsonl

export CUDA_VISIBLE_DEVICES=0

python llm_judge_chatbot_arena_reverse.py --judge_name mistralai/Mistral-7B-Instruct-v0.2 --output_path outputs/lmsys-chatbot-arena-train-reverse/Mistral-7B-Instruct-v0.2.jsonl
python llm_judge_chatbot_arena_reverse.py --judge_name meta-llama/Meta-Llama-3-8B --output_path outputs/lmsys-chatbot-arena-train-reverse/Meta-Llama-3-8B.jsonl
python llm_judge_chatbot_arena_reverse.py --judge_name stabilityai/StableBeluga-7B --output_path outputs/lmsys-chatbot-arena-train-reverse/StableBeluga-7B.jsonl
python llm_judge_chatbot_arena_reverse.py --judge_name HuggingFaceH4/zephyr-7b-beta --output_path outputs/lmsys-chatbot-arena-train-reverse/zephyr-7b-beta.jsonl
python llm_judge_chatbot_arena_reverse.py --judge_name berkeley-nest/Starling-LM-7B-alpha --output_path outputs/lmsys-chatbot-arena-train-reverse/Starling-LM-7B-alpha.jsonl
python llm_judge_chatbot_arena_reverse.py --judge_name meta-llama/Meta-Llama-3-8B-Instruct --output_path outputs/lmsys-chatbot-arena-train-reverse/Meta-Llama-3-8B-Instruct.jsonl
python llm_judge_chatbot_arena_reverse.py --judge_name Open-Orca/Mistral-7B-OpenOrca --output_path outputs/lmsys-chatbot-arena-train-reverse/Mistral-7B-OpenOrca
python llm_judge_chatbot_arena_reverse.py --judge_name cognitivecomputations/dolphin-2.1-mistral-7b --output_path outputs/lmsys-chatbot-arena-train-reverse/dolphin-2.1-mistral-7b
python llm_judge_chatbot_arena_reverse.py --judge_name mistralai/Mistral-7B-Instruct-v0.1 --output_path outputs/lmsys-chatbot-arena-train-reverse/Mistral-7B-Instruct-v0.1
python llm_judge_chatbot_arena_reverse.py --judge_name teknium/OpenHermes-2-Mistral-7B --output_path outputs/lmsys-chatbot-arena-train-reverse/OpenHermes-2-Mistral-7B
python llm_judge_chatbot_arena_reverse.py --judge_name teknium/OpenHermes-2.5-Mistral-7B --output_path outputs/lmsys-chatbot-arena-train-reverse/OpenHermes-2.5-Mistral-7B








