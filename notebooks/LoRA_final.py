import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
import torch

# =====================================
# 1. Load Fine-Tuning Data
# =====================================
file_path = "../output/fine_tuning/data/fine_tuning.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Combine conversation turns (You + Assistant)
combined_texts = []
for conv in data:
    for i in range(0, len(conv), 2):
        you = conv[i].get("content", "")
        assistant = conv[i + 1].get("content", "") if i + 1 < len(conv) else ""
        combined_texts.append(f"<s>{you}</s><sep>{assistant}</sep>")

# =====================================
# 2. Load Tokenizer & Model
# =====================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "sep_token": "<sep>"})

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =====================================
# 3. Apply LoRA
# =====================================
from transformer.lora import get_lora_model, print_trainable_parameters

lora_model = get_lora_model(
    model=model,
    lora_config={
        "rank": 4,
        "alpha": 8,
    },
    device=device
)
print_trainable_parameters(lora_model)

# =====================================
# 4. Tokenize Data (Automatic padding/truncation)
# =====================================
block_size = 128
tokenized_data = tokenizer(
    combined_texts,
    truncation=True,
    padding="max_length",
    max_length=block_size,
    return_tensors="pt"
)
tokenized_data["labels"] = tokenized_data["input_ids"]

# =====================================
# 5. Build Hugging Face Dataset
# =====================================
dataset = Dataset.from_dict({
    "input_ids": tokenized_data["input_ids"],
    "attention_mask": tokenized_data["attention_mask"],
    "labels": tokenized_data["labels"]
})

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# =====================================
# 6. Training Arguments
# =====================================
training_args = TrainingArguments(
    output_dir="./results_lora",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs_lora",
    report_to="none",
)

# =====================================
# 7. Trainer (LoRA model)
# =====================================
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# =====================================
# 8. Train LoRA model
# =====================================
trainer.train()

# =====================================
# 9. Visualize Loss
# =====================================
train_loss_steps = []
eval_loss_steps = []

for log in trainer.state.log_history:
    if "loss" in log:
        train_loss_steps.append((log["step"], log["loss"]))
    if "eval_loss" in log:
        eval_loss_steps.append((log["step"], log["eval_loss"]))

if train_loss_steps:
    steps, losses = zip(*train_loss_steps)
    plt.plot(steps, losses, label="Train Loss")
if eval_loss_steps:
    steps, losses = zip(*eval_loss_steps)
    plt.plot(steps, losses, label="Validation Loss")

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (LoRA)")
plt.legend()
plt.grid()
plt.show()

# =====================================
# 10. Generate from LoRA model
# =====================================
user_message = "Salam labas "
input_tokens = tokenizer.encode(user_message, return_tensors="pt").to(device)

lora_model.eval()
with torch.no_grad():
    output = lora_model.generate(input_tokens=input_tokens, max_new_tokens=100)

print(f"User: {user_message}")
print(f"Assistant (LoRA): {tokenizer.decode(output[0].tolist())}")
