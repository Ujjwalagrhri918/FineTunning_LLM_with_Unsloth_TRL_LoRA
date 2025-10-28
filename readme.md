# 🚀 Fine-Tuning Phi-3 Mini Model using Unsloth and TRL

This project demonstrates how to fine-tune the **Phi-3 Mini** language model efficiently using **Unsloth**, **TRL**, and **PEFT** with **4-bit quantization** for reduced memory usage and faster training.  
The process includes setup, dataset preparation, fine-tuning with LoRA, testing, and exporting the final model in GGUF format.

---

## 🧩 1. Environment Setup

We start by installing the required libraries:
- **Unsloth** — optimized library for fast model training and inference.  
- **TRL (Transformer Reinforcement Learning)** — provides SFT (Supervised Fine-Tuning) trainer utilities.  
- **PEFT** — for parameter-efficient fine-tuning (e.g., LoRA).  
- **Accelerate** — for efficient GPU usage.  
- **Bitsandbytes** — enables quantized (4-bit / 8-bit) model loading.

These libraries ensure that we can train large models on limited hardware efficiently.

---

## 🧠 2. Loading the Model

We load the **Phi-3 Mini 4K Instruct** model from the Unsloth model hub.  
The model is configured with:
- A **maximum sequence length** (e.g., 2048 tokens).  
- **Automatic precision** detection (FP16 or BF16 depending on GPU).  
- **4-bit quantization** enabled to reduce VRAM usage.

Unsloth’s `FastLanguageModel` automatically optimizes model loading for speed and memory efficiency.

---

## 📘 3. Dataset Preparation

A dataset is formatted so the model can learn from input–output pairs.  
Each example is converted into a structured prompt that includes:
- An **Input** section (user query).  
- An **Output** section (expected model response).

The formatted dataset is then loaded into a **Hugging Face Dataset** object for efficient processing and batching during training.

---

## ⚙️ 4. Adding LoRA Adapters

To fine-tune the model efficiently, **LoRA (Low-Rank Adaptation)** adapters are added.  
LoRA allows updating only a small subset of model parameters, making the fine-tuning process:
- **Faster** (less computation).  
- **Cheaper** (lower GPU memory).  
- **Safer** (base model remains intact).

Parameters like rank, alpha, dropout, and target layers define how LoRA adapters behave and where they are applied inside the transformer architecture.

---

## 🎯 5. Fine-Tuning with TRL’s SFTTrainer

The **TRL’s SFTTrainer** is used for fine-tuning. It wraps around Hugging Face’s `Trainer` API and simplifies supervised fine-tuning (SFT).

Training arguments include:
- Batch size, epochs, and learning rate.  
- Gradient accumulation for larger effective batch sizes.  
- Logging frequency, checkpoint saving, and mixed precision settings.

This step performs the actual learning process, where the model adapts to the dataset.

---

## ⚡ 6. Inference and Testing

Once fine-tuning is complete, the model is switched to **inference mode** for optimized text generation (up to 2× faster).  

The testing process involves:
- Providing a **sample prompt** (e.g., an HTML snippet for product extraction).  
- Tokenizing it using the chat template.  
- Generating a model response to verify output quality.

This ensures the fine-tuned model behaves as expected before saving.

---

## 💾 7. Exporting the Model to GGUF Format

The fine-tuned model is saved in **GGUF format**, a lightweight and efficient format used by:
- **Llama.cpp**  
- **Ollama**  
- **LM Studio**  

Quantization (e.g., `q4_k_m`) is applied during export to reduce model size while maintaining high response quality.  
This makes the model ideal for local deployment and edge inference.

---

## 📤 8. Downloading the Model from Colab

The exported `.gguf` file is detected and downloaded from the Colab environment to your local system.  
Once downloaded, the model can be easily integrated into lightweight inference frameworks.

---

## 🧾 Summary of Workflow

| Step | Description |
|------|--------------|
| **1. Install dependencies** | Set up environment for Unsloth, TRL, and PEFT. |
| **2. Load base model** | Load a quantized Phi-3 Mini model efficiently. |
| **3. Prepare dataset** | Format input–output pairs for training. |
| **4. Add LoRA adapters** | Enable efficient parameter fine-tuning. |
| **5. Fine-tune model** | Train using TRL’s SFTTrainer. |
| **6. Test inference** | Validate model responses with sample prompts. |
| **7. Export model** | Save model in GGUF format for deployment. |
| **8. Download model** | Retrieve trained model for local use. |

---

## 🧠 Key Takeaways

- **Unsloth** simplifies large model fine-tuning with powerful optimizations.  
- **LoRA** enables efficient training on smaller GPUs.  
- **TRL’s SFTTrainer** offers an easy, high-level fine-tuning workflow.  
- **GGUF format** allows lightweight, portable deployment.  

This complete workflow shows how to fine-tune and deploy large language models efficiently — combining **speed**, **scalability**, and **portability**.

---

## 🧰 Optional: Tools for Deployment

Once you’ve exported the `.gguf` model, you can use:
- **Ollama** → For local inference with prompt templates.  
- **Llama.cpp** → For CPU/GPU-based inference on any platform.  
- **LM Studio** → For GUI-based testing and chat interaction.

<img width="2339" height="913" alt="image" src="https://github.com/user-attachments/assets/584331a5-3184-4162-a398-10758b1e5054" />


---

📌 **Author:** *Ujjwal Agrahari*  
💡 **Tech Stack:** Python • Unsloth • TRL • Hugging Face • LoRA • Bitsandbytes  

---
