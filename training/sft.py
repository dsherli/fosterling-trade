from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

#load base model + tokenizer from unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,         # or torch.float16 if supported
    load_in_4bit=True,
)

#apply LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# load  SFT data
dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

# Preprocess
def tokenize(example):
    full_text = example["prompt"].strip() + tokenizer.eos_token + example["completion"].strip()
    tokens = tokenizer(full_text, truncation=True, padding="max_length", max_length=2048)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Setup trainer
training_args = TrainingArguments(
    output_dir="adapters/sft_stock_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True if torch.cuda.is_available() else False, #this allows gpu usage if you have an nvidia gpu
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

#save adapter and tokenizer locally to add to model
model.save_pretrained("adapters/sft_stock_model")
tokenizer.save_pretrained("adapters/sft_stock_model")
