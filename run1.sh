#!/bin/bash
#$ -S /bin/bash

# Set up your working dir here
export CUDA_VISIBLE_DEVICES=0,1
# conda activate exp-pp1
export HF_DATASETS_CACHE=/cache/.cache/huggingface/datasets
export HF_HOME=/cache/.cache/huggingface
export HF_HUB_CACHE=/cache/.cache/huggingface/hub


# python llm_judge_chatbot_arena.py --judge_name mistralai/Mistral-7B-Instruct-v0.2 --output_path outputs/lmsys-chatbot-arena-train/Mistral-7B-Instruct-v0.2.jsonl
# python llm_judge_chatbot_arena.py --judge_name meta-llama/Meta-Llama-3-8B --output_path outputs/lmsys-chatbot-arena-train/Meta-Llama-3-8B.jsonl
# python llm_judge_chatbot_arena.py --judge_name stabilityai/StableBeluga-7B --output_path outputs/lmsys-chatbot-arena-train/StableBeluga-7B.jsonl
# python llm_judge_chatbot_arena.py --judge_name HuggingFaceH4/zephyr-7b-beta --output_path outputs/lmsys-chatbot-arena-train/zephyr-7b-beta.jsonl
# python llm_judge_chatbot_arena.py --judge_name berkeley-nest/Starling-LM-7B-alpha --output_path outputs/lmsys-chatbot-arena-train/Starling-LM-7B-alpha.jsonl
# python llm_judge_chatbot_arena.py --judge_name meta-llama/Meta-Llama-3-8B-Instruct --output_path outputs/lmsys-chatbot-arena-train/Meta-Llama-3-8B-Instruct.jsonl
# python llm_judge_chatbot_arena.py --judge_name Open-Orca/Mistral-7B-OpenOrca --output_path outputs/lmsys-chatbot-arena-train/Mistral-7B-OpenOrca
# python llm_judge_chatbot_arena.py --judge_name cognitivecomputations/dolphin-2.1-mistral-7b --output_path outputs/lmsys-chatbot-arena-train/dolphin-2.1-mistral-7b
# python llm_judge_chatbot_arena.py --judge_name mistralai/Mistral-7B-Instruct-v0.1 --output_path outputs/lmsys-chatbot-arena-train/Mistral-7B-Instruct-v0.1
# python llm_judge_chatbot_arena.py --judge_name teknium/OpenHermes-2-Mistral-7B --output_path outputs/lmsys-chatbot-arena-train/OpenHermes-2-Mistral-7B
# python llm_judge_chatbot_arena.py --judge_name teknium/OpenHermes-2.5-Mistral-7B --output_path outputs/lmsys-chatbot-arena-train/OpenHermes-2.5-Mistral-7B


# python llm_judge_truthful_qa.py --judge_name mistralai/Mistral-7B-Instruct-v0.2 --output_path outputs/truthful_qa/Mistral-7B-Instruct-v0.2.jsonl
# python llm_judge_truthful_qa.py --judge_name meta-llama/Meta-Llama-3-8B --output_path outputs/truthful_qa/Meta-Llama-3-8B.jsonl
# python llm_judge_truthful_qa.py --judge_name stabilityai/StableBeluga-7B --output_path outputs/truthful_qa/StableBeluga-7B.jsonl
# python llm_judge_truthful_qa.py --judge_name HuggingFaceH4/zephyr-7b-beta --output_path outputs/truthful_qa/zephyr-7b-beta.jsonl
# python llm_judge_truthful_qa.py --judge_name berkeley-nest/Starling-LM-7B-alpha --output_path outputs/truthful_qa/Starling-LM-7B-alpha.jsonl
# python llm_judge_truthful_qa.py --judge_name meta-llama/Meta-Llama-3-8B-Instruct --output_path outputs/truthful_qa/Meta-Llama-3-8B-Instruct.jsonl
# python llm_judge_truthful_qa.py --judge_name Open-Orca/Mistral-7B-OpenOrca --output_path outputs/truthful_qa/Mistral-7B-OpenOrca.jsonl
# python llm_judge_truthful_qa.py --judge_name cognitivecomputations/dolphin-2.1-mistral-7b --output_path outputs/truthful_qa/dolphin-2.1-mistral-7b.jsonl
# python llm_judge_truthful_qa.py --judge_name mistralai/Mistral-7B-Instruct-v0.1 --output_path outputs/truthful_qa/Mistral-7B-Instruct-v0.1.jsonl
# python llm_judge_truthful_qa.py --judge_name teknium/OpenHermes-2-Mistral-7B --output_path outputs/truthful_qa/OpenHermes-2-Mistral-7B.jsonl
# python llm_judge_truthful_qa.py --judge_name teknium/OpenHermes-2.5-Mistral-7B --output_path outputs/truthful_qa/OpenHermes-2.5-Mistral-7B.jsonl

# 1 Sep 2024
# 1) Llama-3-70B
# python llm_judge_truthful_qa.py --judge_name meta-llama/Meta-Llama-3-70B-Instruct --output_path outputs/truthful_qa/Meta-Llama-3-70B-Instruct.jsonl
# python llm_judge_chatbot_arena.py --judge_name meta-llama/Meta-Llama-3-70B-Instruct --output_path outputs/lmsys-chatbot-arena-train/Meta-Llama-3-70B-Instruct.jsonl

# 2) Mixtral
# python llm_judge_truthful_qa.py --judge_name mistralai/Mixtral-8x7B-Instruct-v0.1 --output_path outputs/truthful_qa/Mixtral-8x7B-Instruct-v0.1.jsonl
# python llm_judge_chatbot_arena.py --judge_name mistralai/Mixtral-8x7B-Instruct-v0.1 --output_path outputs/lmsys-chatbot-arena-train/Mixtral-8x7B-Instruct-v0.1.jsonl

# 3) Qwen2-72B
# python llm_judge_truthful_qa.py --judge_name Qwen/Qwen2-72B-Instruct --output_path outputs/truthful_qa/Qwen2-72B-Instruct.jsonl
# python llm_judge_chatbot_arena.py --judge_name Qwen/Qwen2-72B-Instruct --output_path outputs/lmsys-chatbot-arena-train/Qwen2-72B-Instruct.jsonl

# 4) Hermes3-70B
# python llm_judge_truthful_qa.py --judge_name NousResearch/Hermes-3-Llama-3.1-70B --output_path outputs/truthful_qa/Hermes-3-Llama-3.1-70B.jsonl
python llm_judge_chatbot_arena.py --judge_name NousResearch/Hermes-3-Llama-3.1-70B --output_path outputs/lmsys-chatbot-arena-train/Hermes-3-Llama-3.1-70B.jsonl

# # 5) Athene-70B
# python llm_judge_truthful_qa.py --judge_name Nexusflow/Athene-70B --output_path outputs/truthful_qa/Athene-70B.jsonl
# python llm_judge_chatbot_arena.py --judge_name Nexusflow/Athene-70B --output_path outputs/lmsys-chatbot-arena-train/Athene-70B.jsonl

# 6) Dolphin-mixtral
# python llm_judge_truthful_qa.py --judge_name cognitivecomputations/dolphin-2.5-mixtral-8x7b --output_path outputs/truthful_qa/dolphin-2.5-mixtral-8x7b.jsonl
# python llm_judge_chatbot_arena.py --judge_name cognitivecomputations/dolphin-2.5-mixtral-8x7b --output_path outputs/lmsys-chatbot-arena-train/dolphin-2.5-mixtral-8x7b.jsonl





