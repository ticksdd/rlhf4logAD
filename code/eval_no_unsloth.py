import re
from datasets import load_dataset, Dataset
import os

# One must patch the DPO Trainer first!
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

from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig


tokenizer = AutoTokenizer.from_pretrained("rlhf/code/save_lora/qwen_dpo", max_length=1024, truncation=True, padding=False)
model = AutoModelForCausalLM.from_pretrained("rlhf/code/save_lora/qwen_dpo", device_map="cuda", load_in_4bit=True)

from tqdm import *

import pandas as pd

list_res=[]
instruction="You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes, security violations). Evaluate each log independently for abnormalities, Ignoring sequence-level patterns. As long as one log is considered abnormal, the final judgment result is abnormal. If all logs are normal, the final judgment result is normal.  Don't be too sensitive, no obvious errors will be considered normal(e.g floating point alignment exceptions, generating core, corrected errors , are considered as normal). Output Requirements: Firstly, anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]).  Analysis of Log sequence accroding to the steps: a. Given a brief summery of this log sequnce. b.  Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score.(1-10). And the given log sequence is : "

from build_gp import *
tree = Tree()  # 测试使用短长度
df = pd.read_csv('/home/hanchunhui/projects/hhr/rlhf/code/window_ad/BGL/bgl_20l_optimized_fixed.csv')
for i in tqdm(range(210000,210300)):
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

    # text = tokenizer.apply_chat_template([
    #     {"role" : "system", "content" : SYSTEM_PROMPT + instruction},
    #     {"role" : "user", "content" : text_slice},
    # ], tokenize = False, add_generation_prompt = True)

    text_ = SYSTEM_PROMPT + instruction +text_slice

    inputs = tokenizer(text_, return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, temperature=0.2,max_new_tokens=512,num_beams=4,top_p = 0.95)
    out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)


    if "#Abnormal#" in out_text: 
        tree.insert_list(list_)
        tree.set_status(list_,"abnormal")
    else: 
        tree.insert_list(list_)
        tree.set_status(list_,"normal")

    new_dict={"output": out_text , "label": df['label'][i]}
    list_res.append(new_dict)
tree.save_to_json("./result/dpo_sft_210300.json")
import json
# 将列表保存为JSON文件
with open("./result/dpo_sft_len20_210300.json", "w") as file:
    json.dump(list_res, file, indent=4)
# print(output_)

# 仅显卡0可见（其他卡对程序不可见）
# CUDA_VISIBLE_DEVICES=2 python eval_no_unsloth.py



