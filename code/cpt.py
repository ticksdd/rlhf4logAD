from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset

import torch
max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./save_lora/cpt5", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

alpaca_prompt = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
### Instruction:
{}
### Input:
{}
### Response:
{}"""

instr2="You are a log analysis expert tasked with determining whether any individual log in the given sequence is abnormal (e.g., error, interrupt, crashes, security violations). Evaluate each log independently for abnormalities, Ignoring sequence-level patterns. As long as one log is considered abnormal, the final judgment result is abnormal. If all logs are normal, the final judgment result is normal.  Don't be too sensitive, no obvious errors will be considered normal(e.g alignment exceptions, generating core, are considered as normal), and corrected errors will also be considered normal). Output Requirements: Firstly, anomaly status (Strictly limit the format to [anomaly status: #Abnormal# / #Normal#]).  Analysis of Log sequence: a. Given a brief summery of this log sequnce. b.  Identify triggered security protocol rules and anomaly patterns. c. Analyze and provide the anomaly severity score for this log sequence.(1-10). And the given log sequence is : "

# instr="You are an expert in the field of log analysis. And the BGL dataset is one of the most famous benchmark datasets in the field of log anomaly detection, collected by Lawrence Livermore National Laboratory (LLNL) during the operation of the IBM BlueGene/L supercomputer, recording the working status and abnormal events of system components. I will now provide you with a log template, you need analyse the log template. Please firstly reply [#normal# / #abnormal#], and then describe this log template and explain the reason for the normal/abnormal . Strictly limit reply to no more than 128 tokens. The given log template is:  "

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = instr2
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("./data/test3", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# instruction="You are an expert in the field of log analysis. And the BGL dataset is one of the most famous benchmark datasets in the field of log anomaly detection, collected by Lawrence Livermore National Laboratory (LLNL) during the operation of the IBM BlueGene/L supercomputer, recording the working status and abnormal events of system components. I will now provide you with a log template, you need analyse the log template. Please firstly reply [#normal# / #abnormal#], and then describe this log template and explain the reason for the normal/abnormal . Strictly limit reply to no more than 128 tokens.  "

# uncomment middle messages for 1-shot prompting
# def get_gsm8k_questions(split = "train") -> Dataset:
#     data = load_dataset('./data/test3', 'default')[split] # type: ignore
#     data = data.map(lambda x: { # type: ignore
#         'prompt': [
#             {'role': 'system', 'content':  SYSTEM_PROMPT + instr2},
#             {'role': 'user', 'content': x['input']+EOS_TOKEN}
#         ],
#         'answer':x['output']
#     }) # type: ignore
#     return data # type: ignore

# dataset = get_gsm8k_questions()



from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,

        # Use warmup_ratio and num_train_epochs for longer runs!
        # max_steps = 120,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        num_train_epochs = 10,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        # embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


trainer_stats = trainer.train()


model.save_pretrained("./save_lora/cpt6")  # Local saving
tokenizer.save_pretrained("./save_lora/cpt6")

# 仅显卡0可见（其他卡对程序不可见）
# CUDA_VISIBLE_DEVICES=0 python cpt.py





















