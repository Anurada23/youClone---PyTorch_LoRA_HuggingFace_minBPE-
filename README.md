# youClone – Mimicking AI with Transformers

**youClone** is a passion-driven and an educational project built to deeply understand how an **end-to-end ML pipeline** works — from raw text data and tokenizer training all the way to building and fine-tuning a transformer model.  
The aim was not just to build a chatbot, but to **learn the internals of modern LLMs** — tokenization, embeddings, self-attention, decoder architecture, and fine-tuning strategies.

---

## 🚀 Project Overview

youClone is a personalized conversational AI that learns from a user's WhatsApp chat history and generates responses in their own style.

The project was developed in two main phases:

1. 🛠️ **From Scratch** – Built a custom tokenizer and a GPT-style transformer model from the ground up using PyTorch.  
   - Learned how tokenization, embeddings, attention, and generation work internally.
   - Implemented next-token prediction and trained the model on chat data.

2. 🤖 **Fine-Tuning Pretrained GPT-2** – Used Hugging Face Transformers to fine-tune GPT-2 on custom data.  
   - Performed **full fine-tuning** and **parameter-efficient LoRA fine-tuning**.
   - Compared training efficiency and results between both approaches.

---

## 📚 Features

- 🧠 Custom BPE Tokenizer built with **minBPE**
- 🏗️ GPT-style transformer model implemented from scratch with PyTorch  
- 🔁 Next-token prediction generation  
- 📚 Full fine-tuning of **GPT-2**  
- ⚡ LoRA-based parameter-efficient fine-tuning (PEFT)  
- 📈 Real-time training/validation loss visualization  
- ⚙️ FastAPI backend ready for API integration  
- 🧪 Complete **end-to-end ML pipeline**: raw text → tokenizer → model → training → generation

---

## 🧰 Tech Stack

| Component | Tools & Libraries |
|----------|--------------------|
| **Backend** | FastAPI, Python |
| **ML / DL** | PyTorch, Hugging Face Transformers, PEFT (LoRA), minBPE |
| **Data Processing** | Pandas, JSON |
| **Visualization** | Matplotlib |
| **Hardware Acceleration** | CUDA / GPU |

---



