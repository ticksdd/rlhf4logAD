from unsloth import FastLanguageModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Optional set GPU device ID
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Optional set GPU device ID
# print(os.listdir('../data/gsm8k'))  # 确认目录中有数据文件

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "rlhf/model/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
instruction="You are an expert in the field of log analysis. And the BGL dataset is one of the most famous benchmark datasets in the field of log anomaly detection, collected by Lawrence Livermore National Laboratory (LLNL) during the operation of the IBM BlueGene/L supercomputer, recording the working status and abnormal events of system components. I will now provide you with a log template, you need analyse the log template. Please firstly reply [#normal# / #abnormal#], and then describe this log template and explain the reason for the normal/abnormal . Strictly limit reply to no more than 128 tokens.  "

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
# print(dataset[0])

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instr="You are an expert in the field of log analysis. And the BGL dataset is one of the most famous benchmark datasets in the field of log anomaly detection, collected by Lawrence Livermore National Laboratory (LLNL) during the operation of the IBM BlueGene/L supercomputer, recording the working status and abnormal events of system components. I will now provide you with a log template, you need analyse the log template. Please firstly reply [#normal# / #abnormal#], and then describe this log template and explain the reason for the normal/abnormal . Strictly limit reply to no more than 128 tokens. The given log template is:  "

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = instr
    inputs       = examples["text"]
    outputs      = examples["answer"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("./data/analyse_ad", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 3,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()
# trainer.train()

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

model.save_pretrained("./save_lora/analyse_ad_10epo")
tokenizer.save_pretrained("./save_lora/analyse_ad_10epo")

#test for lora
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# 仅显卡0可见（其他卡对程序不可见）
# CUDA_VISIBLE_DEVICES=1,2 python sft_.py

