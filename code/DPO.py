
import os
from datasets import load_dataset, Dataset
# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from unsloth import FastLanguageModel
import torch


max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/RLHF/code/save_lora/llama_dpo", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


from dpo_data import *


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
instr2="You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes, security violations). Evaluate each log independently for abnormalities, Ignoring sequence-level patterns. As long as one log is considered abnormal, the final judgment result is abnormal. If all logs are normal, the final judgment result is normal.  Don't be too sensitive, no obvious errors will be considered normal(e.g alignment exceptions, generating core, are considered as normal), and corrected errors will also be considered normal). Output Requirements: Firstly, anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]).  Analysis of Log sequence: a. Given a brief summery of this log sequnce. b.  Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score for this log sequence.(1-10). And the given log sequence is : "


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('../data/cpt_dpo', 'default')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': x['conversations'][0]['value'],
        'chosen': x['chosen']['value'],
        'rejected': x['rejected']['value'],
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()
print(dataset[0])
print("type:       ",type(dataset))


# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 32,
#     lora_dropout = 0, # Currently only supports dropout = 0
#     bias = "none",    # Currently only supports bias = "none"
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )


# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 10,
        learning_rate = 5e-6,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
    beta = 0.1,
    train_dataset = dataset,
    # eval_dataset = raw_datasets["test"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 4096,
)

dpo_trainer.train()

print(type(model),"+++++++++++++++++++++++++++")
print(type(model),"+++++++++++++++++++++++++++")

model.save_pretrained("./save_lora/llama_dpo2")
tokenizer.save_pretrained("./save_lora/llama_dpo2")

#test for lora
# text = tokenizer.apply_chat_template([
#     {"role" : "system", "content" : SYSTEM_PROMPT+instr2},
#     {"role" : "user", "content" : dataset[0]['conversations'][0]['value']},
# ], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.2,
    top_p = 0.95,
    max_tokens = 1024,
)
# output_ = model.fast_generate(
#     dataset[0]['conversations'][0]['value'],
#     sampling_params = sampling_params,
#     lora_request = model.load_lora("./save_lora/llama_dpo2"),
# )[0].outputs[0].text
# print(output_)


if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./save_lora/llama_dpo2", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(dataset[0]['conversations'][0]['value'], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)
out=tokenizer.batch_decode(outputs)
print(out)
# 仅显卡0可见（其他卡对程序不可见）
# CUDA_VISIBLE_DEVICES=2 python DPO.py

