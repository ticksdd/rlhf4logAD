
import re
from datasets import load_dataset, Dataset
import os

# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from unsloth import FastLanguageModel
import torch

max_seq_length = 2560 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "RLHF/model/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    truncation=True,          # 强制截断
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
    skip_special_tokens=True,
)





