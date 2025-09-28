import re
from datasets import load_dataset, Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Optional set GPU device ID
# print(os.listdir('../data/gsm8k'))  # 确认目录中有数据文件

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

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# print(dataset[0])




# # Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content'] 
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards=[0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    # print("zheshi int:+++++++++",rewards)
    return rewards

# def strict_abnormal_reward_func(completions,**kwargs) -> list[float]:
#     """
#     严格模式：要求文本中精确包含 [#Abnormal#] 或 [#Normal#] 标记
#     - 奖励权重:1.0(高优先级）
#     - 必须完整匹配方括号和#号，大小写敏感
#     """
#     pattern = r'#(Abnormal|Normal)#'  # 精确匹配标记
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.search(pattern, r) for r in responses]
#     return [2.0 if match else 0.0 for match in matches]

def consistency_reward_func(prompts, completions, answer,**kwargs) -> list[float]:
    """验证生成文本与标签的一致性奖励函数
    规则：
    1. 如果标签包含"#Abnormal#"，生成的文本必须包含"#Abnormal#"
    2. 如果标签包含"#Normal#"，生成的文本必须包含"#Normal#"
    3. 其他情况返回0分
    
    参数：
        prompts: 输入提示列表（保持与示例函数相同结构）
        completions: 模型生成的完整响应列表，格式为 [ [{'content': str}], ... ]
        answer: 标签列表，每个元素应为包含"#Abnormal#"或"#Normal#"的字符串
    """
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for response, label in zip(responses, answer):
        # 关键检查逻辑
        if "#Abnormal#" in label:
            reward = 2.0 if "#Abnormal#" in response else 0.0
        elif "#Normal#" in label:
            reward = 2.0 if "#Normal#" in response else 0.0
        else:
            # 处理意外标签类型
            reward = 0.0
        
        # 可选调试信息（参考示例函数）
        # if kwargs.get('verbose', False):
        #     print('-'*20, 
        #           f"Label: {label}",
        #           f"\nGenerated: {response}", 
        #           f"\nReward: {reward}")
        # print("本次得分异常检测得分为：", reward)
        # print("本次得分异常检测得分为：", reward, "\ngenerated+++++++++: ",response)
        
        rewards.append(reward)
    print("step score: ",rewards)
    
    return rewards



def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from unsloth import FastLanguageModel
import torch
max_seq_length = 4196 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "RLHF/model/unsloth/Meta-Llama-31-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    truncation=True,          # 强制截断
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    gpu_memory_utilization = 0.8, # Reduce if out of memory
    skip_special_tokens=True,
)


instr2="You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes, security violations). Evaluate each log independently for abnormalities, Ignoring sequence-level patterns. As long as one log is considered abnormal, the final judgment result is abnormal. If all logs are normal, the final judgment result is normal.  Don't be too sensitive, no obvious errors will be considered normal(e.g alignment exceptions, generating core, are considered as normal), and corrected errors will also be considered normal). Output Requirements: Firstly, anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]).  Analysis of Log sequence: a. Given a brief summery of this log sequnce. b.  Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score for this log sequence.(1-10). And the given log sequence is : "
# instruction="You are an expert in the field of log analysis. And the BGL dataset is one of the most famous benchmark datasets in the field of log anomaly detection, collected by Lawrence Livermore National Laboratory (LLNL) during the operation of the IBM BlueGene/L supercomputer, recording the working status and abnormal events of system components. I will now provide you with a log template, you need analyse the log template. Please firstly reply [#normal# / #abnormal#], and then describe this log template and explain the reason for the normal/abnormal . Strictly limit reply to no more than 128 tokens.  "
instr3="You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes). Evaluate each log independently for abnormalitie. One log is considered abnormal, the log sequence will be regard as abnormal. Don't be too sensitive, no obvious errors will be considered normal(e.g alignment exceptions, corrected errors). Output Requirements: Firstly, a. anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]). b. Analysis of Log sequence and Identify anomaly patterns. c. Given the anomaly severity score.(1-10). And the given log sequence is :  "
# uncomment middle messages for 1-shot prompting

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('../data/grpo_data', 'default')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content':  SYSTEM_PROMPT+instr3},
            {'role': 'user', 'content': x['input'] + EOS_TOKEN}
        ],
        'answer':x['output']
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()
print(dataset[0]['prompt'][1]['content'])
print("type:       ",type(dataset))


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.



max_prompt_length = 4196

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 100,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = 1024,
    num_train_epochs = 10, # Set to 1 for a full training run
    # max_steps = 2,
    # save_steps = 400,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & BiasesSYSTEM_PROMPT
    output_dir = "outputs/grpo_400",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        consistency_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()

# try: 
#     print("begin to train_________________________________")
#     trainer.train()

# except:
#     # save lora
#     print("begin to save_____________________________")
#     model.save_pretrained("./save_lora/grpo_save_fixed")
#     tokenizer.save_pretrained("./save_lora/grpo_save_fixed")

print(type(model),"+++++++++++++++++++++++++++")
print(type(model),"+++++++++++++++++++++++++++")

model.save_pretrained("./save_lora/grpo_10")
tokenizer.save_pretrained("./save_lora/grpo_10")

print("model saved")
#test for lora

# text = tokenizer.apply_chat_template([
#     {"role" : "system", "content" : SYSTEM_PROMPT+instr2},
#     {"role" : "user", "content" : dataset[0]['prompt'][1]['content']},
# ], tokenize = False, add_generation_prompt = True)

# from vllm import SamplingParams
# sampling_params = SamplingParams(
#     temperature = 0.2,
#     top_p = 0.95,
#     max_tokens = 1024,
# )
# output_ = model.fast_generate(
#     text,
#     sampling_params = sampling_params,
#     lora_request = model.load_lora("./save_lora/cpt5"),
# )[0].outputs[0].text
# print(output_)

# 仅显卡0可见（其他卡对程序不可见）
# CUDA_VISIBLE_DEVICES=0 python grpo.py

