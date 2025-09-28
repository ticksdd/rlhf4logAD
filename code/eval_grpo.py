
import re
from datasets import load_dataset, Dataset
import os

# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()
from peft import PeftModelForCausalLM
from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer

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

# max_seq_length = 8196 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# tokenizer=AutoTokenizer.from_pretrained("/code/save_lora/grpo_save_fixed_all",)
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = "RLHF/code/save_lora/llama_dpo2",
    # model_name="/home/hanchunhui/projects/hhr/rlhf/model/unsloth/Qwen3-8B-unsloth-bnb-4bit" ,
    max_seq_length=8192,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    gpu_memory_utilization = 0.6, # Reduce if out of memory
    skip_special_tokens=True,
)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.2,
    top_p = 0.95,
    max_tokens = 1024,
)
# from unsloth.chat_templates import get_chat_template
from tqdm import *


import pandas as pd

list_res=[]
# instruction="You are a log analysis expert tasked with determining whether there are any abnormal logs in the given log sequence by analyse the given log sequence. **Output Requirements**: Firstly, return anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]). Secondly, Detailed reasoning including: a. Given a brief summery of this log sequnce. b. Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score for this log sequence.(1-10). And the given log sequence is : "
# instruction="You are a log analysis expert tasked with determining whether there are any abnormal logs in the given log sequence by analyse the given log sequence. **Output Requirements**: Firstly, return anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]). Secondly, Detailed reasoning including: a. Given a brief summery of this log sequnce. b. Identify triggered security protocol rules and anomaly patterns.(Don't be too sensitive, no obvious errors will be considered normal, and corrected errors will also be considered normal) c. Analyze and provide the anomaly severity score for this log sequence.(1-10). And the given log sequence is : "
#instruction="You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes, security violations). Evaluate each log independently for abnormalities, ignoring sequence-level patterns. Don't be too sensitive, no obvious errors will be considered normal, and corrected errors will also be considered normal). Output Requirements: Final Status: [anomaly status: #Abnormal#] if any single log is abnormal. [anomaly status: #Normal#] only if all logs are normal. Analysis of Log sequence: a. Given a brief summery of this log sequnce. b.  Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score for this log sequence.(1-10). And the given log sequence is : "
instruction="You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes, security violations). Evaluate each log independently for abnormalities, Ignoring sequence-level patterns. As long as one log is considered abnormal, the final judgment result is abnormal. If all logs are normal, the final judgment result is normal.  Don't be too sensitive, no obvious errors will be considered normal(e.g floating point alignment exceptions, generating core, corrected errors , are considered as normal). Output Requirements: Firstly, anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]).  Analysis of Log sequence accroding to the steps: a. Given a brief summery of this log sequnce. b.  Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score.(1-10). And the given log sequence is : "
# for i in tqdm(range(len(dataset))):
#test for lora

from build_gp import *
# tree = SequenceTree(seq_length=100)  # 测试使用短长度
tree = Tree()  # 测试使用短长度
# tree = SequenceTree.load("sequence20_tree_ad_20000.json")
df = pd.read_csv('RLHF/data/window_ad/bgl_20l_optimized_fixed.csv')
for i in tqdm(range(210000,220000)):
    list_temp=eval(df['eventlist'][i])
    list_uniq=list(set(list_temp))
    list_ = sorted(list_uniq, key=lambda x: int(x[1:]))

    if tree.exists(list_)&tree.is_endpoint(list_) : 
        if tree.get_status(list_)=="normal":
            list_res.append({"output": "#Normal#", "label": df['label'][i]})
            continue
        else: 
            list_res.append({"output": "#Abnormal#", "label": df['label'][i]})
            continue

    if len(df['content'][i])>20000:
        text_slice=df['content'][i][:20000]
    else: text_slice=df['content'][i]

    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT + instruction},
        {"role" : "user", "content" : text_slice},
    ], tokenize = False, add_generation_prompt = True)

    output_ = model.fast_generate(
        text,
        sampling_params = sampling_params,
    )[0].outputs[0].text

    if "#Abnormal#" in output_: 
        tree.insert_list(list_)
        tree.set_status(list_,"abnormal")
    else: 
        tree.insert_list(list_)
        tree.set_status(list_,"normal")

    new_dict={"output": output_ , "label": df['label'][i]}
    list_res.append(new_dict)
tree.save_to_json("./result/dpo_sft_210000.json")
import json
# 将列表保存为JSON文件
with open("./result/dpo_sft_len20_210000.json", "w") as file:
    json.dump(list_res, file, indent=4)
# print(output_)

# 仅显卡0可见（其他卡对程序不可见）
# CUDA_VISIBLE_DEVICES=2 python eval_grpo.py



