from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only

max_seq_length = 8192
load_in_4bit = False
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


dataset = load_dataset("json", data_files="cc_messages.jsonl", split = "train")


tokenizer = get_chat_template(
    tokenizer,
#    chat_template = "qwen-2.5", 
    chat_template = "qwen-3", 

)

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False,
            enable_thinking=False 
        ) for messages in examples["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)


class SafeSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss

trainer = SafeSFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        save_strategy = "steps",
        save_steps = 500,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

print(tokenizer._ollama_modelfile)

print(trainer.train_dataset[5]["labels"])

trainer_stats = trainer.train()

model.save_pretrained_gguf("q3", tokenizer,) # Note: also saves full, merged hf version