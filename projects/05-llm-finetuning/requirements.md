# Project 05: LLM Fine-Tuning with Hugging Face & QLoRA

**XP:** 350 | **Cost:** Free (Google Colab T4 GPU) | **Duration:** Weeks 9–10

## Objective
Fine-tune a 7B parameter open-source LLM (Mistral-7B or Llama 3.1 8B) using QLoRA (4-bit quantization + LoRA adapters) on a custom instruction dataset. Evaluate before/after quality with ROUGE and custom metrics, push the adapter to Hugging Face Hub, and serve locally via Ollama.

## Tools & Libraries
- **Unsloth** — 2x faster training, 70% less VRAM vs vanilla HF
- **PEFT** — LoRA adapter management
- **TRL** — SFTTrainer for instruction tuning
- **BitsAndBytes** — 4-bit quantization (QLoRA)
- **Hugging Face Hub** — model registry and sharing
- **Ollama** — local inference from GGUF
- **RAGAS** (optional) — quality evaluation

## Project Structure
```
05-llm-finetuning/
├── notebooks/
│   ├── 01_dataset_prep.ipynb
│   ├── 02_finetune_qlora.ipynb
│   └── 03_evaluate_and_export.ipynb
├── data/
│   ├── train.jsonl
│   └── eval.jsonl
├── configs/
│   └── training_config.yaml
├── docs/adr/
│   └── 001-base-model-selection.md
└── requirements.txt
```

## Hardware Options
| Option | Cost | VRAM | Speed | Notes |
|---|---|---|---|---|
| Google Colab T4 | Free | 16GB | ~200 tok/s | Recommended — use with Unsloth |
| Colab A100 | ~$10/hr | 40GB | ~800 tok/s | Faster, for 13B+ models |
| MacBook M2 Pro | Free | shared | ~20 tok/s | Use llama.cpp for inference only |
| RunPod RTX 4090 | ~$0.44/hr | 24GB | ~600 tok/s | Best value paid option |

## Phase 1: Dataset Preparation (Days 1–2)

### Choose a task domain
Pick one of:
- **SQL generation** — text-to-SQL on Spider dataset
- **Code generation** — Python docstring-to-function on CodeSearchNet
- **Customer support** — instruction-following on Alpaca format
- **Medical QA** — HealthCareMagic dataset (if you have access)

### Format as instruction tuning dataset (Alpaca format)
```python
# data format: each line is a JSON object
# {"instruction": "...", "input": "...", "output": "..."}

import json
from datasets import load_dataset

# Example: use a subset of the Alpaca dataset
ds = load_dataset("tatsu-lab/alpaca", split="train")

# Filter to 2000 high-quality examples
filtered = ds.filter(lambda x: len(x["output"]) > 50 and len(x["output"]) < 500)
subset = filtered.select(range(2000))

# 90/10 split
split = subset.train_test_split(test_size=0.1, seed=42)

def to_jsonl(dataset, path):
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps({
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"],
            }) + "\n")

to_jsonl(split["train"], "data/train.jsonl")
to_jsonl(split["test"], "data/eval.jsonl")
print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
```

### Create prompt template
```python
def format_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    return prompt + output if output else prompt
```

## Phase 2: QLoRA Fine-Tuning (Days 3–6)

In `notebooks/02_finetune_qlora.ipynb` (run on Colab with T4):

### Install dependencies
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### Load model with 4-bit quantization
```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None           # auto-detect (float16 for T4/V100)
load_in_4bit = True    # QLoRA

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

### Add LoRA adapters
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # LoRA rank (16 = good balance)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}% of total)")
```

### Prepare dataset
```python
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "test": "data/eval.jsonl"})

def formatting_func(example):
    return format_prompt(example["instruction"], example.get("input", ""), example["output"])

dataset = dataset.map(lambda x: {"text": formatting_func(x)})
```

### Train
```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        report_to="none",
    ),
)

trainer_stats = trainer.train()
print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.1f}")
```

## Phase 3: Evaluation (Days 7–8)

In `notebooks/03_evaluate_and_export.ipynb`:

### Before/After comparison
```python
from unsloth import FastLanguageModel

FastLanguageModel.for_inference(model)

test_prompts = [
    "Explain what a transformer model is in simple terms.",
    "Write a Python function to reverse a linked list.",
    "What are the differences between REST and GraphQL?",
]

def generate(prompt, max_new_tokens=256):
    inputs = tokenizer([format_prompt(prompt)], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

for p in test_prompts:
    print(f"\nPrompt: {p}")
    print(f"Response: {generate(p)}\n" + "="*50)
```

### ROUGE evaluation
```python
from rouge_score import rouge_scorer
import pandas as pd

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

eval_data = []
with open("data/eval.jsonl") as f:
    for line in f:
        eval_data.append(json.loads(line))

scores = []
for item in eval_data[:100]:
    pred = generate(item["instruction"], max_new_tokens=200)
    score = scorer.score(item["output"], pred)
    scores.append({
        "rouge1": score["rouge1"].fmeasure,
        "rouge2": score["rouge2"].fmeasure,
        "rougeL": score["rougeL"].fmeasure,
    })

df = pd.DataFrame(scores)
print("Mean ROUGE scores:")
print(df.mean().round(4))
```

## Phase 4: Export to GGUF and Ollama (Days 9–10)

### Push adapter to Hugging Face Hub
```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN")

model.save_pretrained("mistral-alpaca-lora")
tokenizer.save_pretrained("mistral-alpaca-lora")

# Push to Hub
model.push_to_hub("YOUR_USERNAME/mistral-7b-alpaca-qlora")
tokenizer.push_to_hub("YOUR_USERNAME/mistral-7b-alpaca-qlora")
```

### Export to GGUF for Ollama
```python
# Merge adapter weights and save full model (needed for GGUF export)
model.save_pretrained_merged("mistral-alpaca-merged", tokenizer, save_method="merged_16bit")

# Export to GGUF (quantized for CPU/Metal inference)
model.save_pretrained_gguf("mistral-alpaca-gguf", tokenizer, quantization_method="q4_k_m")
```

### Create Ollama Modelfile
```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./mistral-alpaca-gguf/unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
SYSTEM "You are a helpful AI assistant fine-tuned on instruction data."
EOF

# Register with Ollama
ollama create mistral-alpaca -f Modelfile

# Test
ollama run mistral-alpaca "Explain what RAG is in 3 sentences."
```

## Training Config Reference

`configs/training_config.yaml`:
```yaml
model:
  name: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 16
  alpha: 16
  dropout: 0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  batch_size: 2
  gradient_accumulation_steps: 4
  epochs: 3
  learning_rate: 2e-4
  warmup_steps: 5
  weight_decay: 0.01
  scheduler: "linear"

dataset:
  train_path: "data/train.jsonl"
  eval_path: "data/eval.jsonl"
  max_samples: 2000
```

## Acceptance Criteria
- [ ] Fine-tuning completes on Colab T4 without OOM errors
- [ ] Training loss decreases consistently over 3 epochs
- [ ] ROUGE-L score improves by at least 5% vs base model on eval set
- [ ] LoRA adapter pushed to Hugging Face Hub (public or private)
- [ ] GGUF model runs locally via Ollama
- [ ] Before/after comparison notebook shows quality improvement
- [ ] ADR-001 written: Mistral-7B vs Llama 3.1 8B vs Phi-3 mini for instruction tuning

## ADR Template
Create `docs/adr/001-base-model-selection.md`:
```markdown
# ADR-001: Base Model for Instruction Fine-Tuning

## Status: Accepted

## Context
Need to select a 7–8B parameter open-source LLM for instruction tuning via QLoRA.
Constraints: must fit in 16GB VRAM (T4 GPU) with 4-bit quantization.

## Decision
Mistral-7B-Instruct-v0.3 via Unsloth.

## Consequences
+ Strong instruction-following baseline, no safety restrictions on fine-tuning
+ Unsloth support: 2x training speed vs vanilla HuggingFace
+ Apache 2.0 license — commercial use allowed
- Smaller context window (8K) vs Llama 3.1 (128K)
- Less multilingual capability than Llama 3.1
```
