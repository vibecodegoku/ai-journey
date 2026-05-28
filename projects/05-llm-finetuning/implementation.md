# Project 05: LLM Fine-Tuning with QLoRA — Implementation Guide

## Prerequisites
- Google account (for Google Colab T4 GPU — free)
- Hugging Face account (https://huggingface.co — free)
- HF write token: Settings → Access Tokens → New token (write)
- Ollama installed locally (from Project 00)
- `conda activate aiarch`

> **Note:** All training notebooks run in Google Colab with a T4 GPU.
> Local setup is only needed for the Ollama Modelfile step.

---

## Project Structure
```
05-llm-finetuning/
├── notebooks/
│   ├── 01_dataset_prep.ipynb     # Alpaca format dataset creation
│   ├── 02_finetune.ipynb         # QLoRA training with Unsloth
│   └── 03_evaluate.ipynb         # ROUGE-L evaluation + comparison
├── Modelfile                      # Ollama model definition
├── docs/
│   └── adr/
│       └── ADR-001-base-model-selection.md
└── README.md
```

---

## Phase 1: Hugging Face Setup

### Step 1.1 — Create HF account and get token
1. Go to https://huggingface.co → Sign Up
2. Settings (top right avatar) → Access Tokens → New token
3. Name: `aiarch-finetune`, Role: **Write**
4. Copy the token (starts with `hf_...`)

### Step 1.2 — Request access to Llama 3.1 (optional)
If using Llama 3.1 instead of Mistral:
1. Visit https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Request access" and fill in the form (~1 day approval)
3. Mistral-7B has no approval requirement — use it if Llama isn't approved yet

### Step 1.3 — Create your model repo on HF Hub
```bash
conda activate aiarch
pip install huggingface_hub
huggingface-cli login
# Paste your HF token

huggingface-cli repo create aiarch-finetuned-adapter --type model
# This creates: https://huggingface.co/YOUR_USERNAME/aiarch-finetuned-adapter
```

---

## Phase 2: Dataset Preparation (Google Colab)

### Step 2.1 — Open Google Colab
Go to https://colab.research.google.com
- Runtime → Change runtime type → **T4 GPU**
- Connect

### Step 2.2 — Create `01_dataset_prep.ipynb`

**Cell 1 — Install libraries**
```python
!pip install -q datasets huggingface_hub transformers
```

**Cell 2 — Load and explore Alpaca dataset**
```python
from datasets import load_dataset

# Alpaca is a standard instruction-following dataset
raw_dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Total examples: {len(raw_dataset)}")
print(raw_dataset[0])
```

**Cell 3 — Filter and format to 2000 examples**
```python
import random
random.seed(42)

# Filter to good quality examples (non-empty output, reasonable length)
filtered = [
    ex for ex in raw_dataset
    if len(ex["output"]) > 50
    and len(ex["instruction"]) > 20
    and len(ex["output"]) < 800
]
print(f"Filtered to {len(filtered)} quality examples")

# Sample 2000
sample = random.sample(filtered, min(2000, len(filtered)))

# Format to Alpaca instruction template
def format_example(ex):
    if ex.get("input", "").strip():
        instruction_text = f"{ex['instruction']}\n\n{ex['input']}"
    else:
        instruction_text = ex["instruction"]
    return {
        "instruction": instruction_text,
        "output": ex["output"],
        "text": f"### Instruction:\n{instruction_text}\n\n### Response:\n{ex['output']}"
    }

formatted = [format_example(ex) for ex in sample]
print(f"\nSample formatted example:")
print(formatted[0]["text"][:300])
```

**Cell 4 — 90/10 train/eval split and save**
```python
from datasets import Dataset

dataset = Dataset.from_list(formatted)
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds  = splits["test"]

print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

# Save locally in Colab
train_ds.save_to_disk("/content/alpaca_train")
eval_ds.save_to_disk("/content/alpaca_eval")
print("Saved to /content/")
```

**Cell 5 — Upload to HF Hub**
```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

train_ds.push_to_hub("YOUR_USERNAME/alpaca-2k", split="train")
eval_ds.push_to_hub("YOUR_USERNAME/alpaca-2k", split="test")
print("Dataset pushed to HF Hub!")
```

---

## Phase 3: QLoRA Fine-Tuning (Google Colab)

### Step 3.1 — Create `02_finetune.ipynb`

**Cell 1 — Install Unsloth (fastest fine-tuning library)**
```python
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

**Cell 2 — Load model with 4-bit quantization**
```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # Auto-detect: float16 for T4
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print("Model loaded!")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

**Cell 3 — Add LoRA adapters (r=16)**
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

**Cell 4 — Load dataset and configure training**
```python
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments

train_ds = load_from_disk("/content/alpaca_train")
eval_ds  = load_from_disk("/content/alpaca_eval")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
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
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="/content/mistral-alpaca-output",
        report_to="none",
    ),
)
```

**Cell 5 — Train!**
```python
import time
start = time.time()

trainer_stats = trainer.train()

elapsed = time.time() - start
print(f"\nTraining complete in {elapsed/60:.1f} minutes")
print(f"Final train loss: {trainer_stats.training_loss:.4f}")
```
> Expected: ~200 tokens/sec on T4, ~30 min for 3 epochs on 1800 examples

**Cell 6 — Push LoRA adapter to HF Hub**
```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

HF_USERNAME = "YOUR_USERNAME"
REPO_NAME = "aiarch-finetuned-adapter"

model.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", token="hf_YOUR_TOKEN_HERE")
tokenizer.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", token="hf_YOUR_TOKEN_HERE")
print(f"Pushed to https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
```

**Cell 7 — Save GGUF for Ollama**
```python
# Export to 4-bit GGUF format
model.save_pretrained_gguf(
    "/content/mistral-alpaca-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF saved to /content/mistral-alpaca-gguf/")
!ls /content/mistral-alpaca-gguf/
```

**Cell 8 — Download GGUF to your machine**
```python
# Download from Colab to local machine
from google.colab import files
import os

gguf_dir = "/content/mistral-alpaca-gguf"
for f in os.listdir(gguf_dir):
    if f.endswith(".gguf"):
        files.download(os.path.join(gguf_dir, f))
        print(f"Downloading {f}...")
```

---

## Phase 4: ROUGE-L Evaluation (Google Colab)

### Step 4.1 — Create `03_evaluate.ipynb`

**Cell 1 — Install ROUGE**
```python
!pip install -q rouge-score
```

**Cell 2 — Load baseline model (no fine-tuning)**
```python
from unsloth import FastLanguageModel
from datasets import load_from_disk
from rouge_score import rouge_scorer

# Reload fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/mistral-alpaca-output/checkpoint-best",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # Enable faster inference

# Load eval set
eval_ds = load_from_disk("/content/alpaca_eval")
```

**Cell 3 — Generate predictions**
```python
def generate_response(instruction: str, max_new_tokens: int = 256) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# Evaluate on 50 examples (subset for speed)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
sample = eval_ds.select(range(50))

scores = []
for ex in sample:
    prediction = generate_response(ex["instruction"])
    score = scorer.score(ex["output"], prediction)
    scores.append(score["rougeL"].fmeasure)

avg_rougeL = sum(scores) / len(scores)
print(f"Fine-tuned model ROUGE-L: {avg_rougeL:.4f}")
```

**Cell 4 — Compare with base model baseline**
```python
# Load the original (non-fine-tuned) model for comparison
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(base_model)

baseline_scores = []
for ex in sample:
    prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
    inputs = base_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = base_model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    pred = base_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    score = scorer.score(ex["output"], pred)
    baseline_scores.append(score["rougeL"].fmeasure)

avg_baseline = sum(baseline_scores) / len(baseline_scores)
improvement = (avg_rougeL - avg_baseline) / avg_baseline * 100

print(f"Baseline ROUGE-L: {avg_baseline:.4f}")
print(f"Fine-tuned ROUGE-L: {avg_rougeL:.4f}")
print(f"Improvement: +{improvement:.1f}%")
assert improvement >= 5.0, f"ROUGE-L improvement {improvement:.1f}% < 5% target"
print("PASS: Meets 5% improvement threshold!")
```

---

## Phase 5: Ollama Modelfile (Local)

### Step 5.1 — Copy GGUF to Ollama models directory
```bash
# After downloading the .gguf file from Colab
mkdir -p ~/ollama-models/alpaca-mistral
cp ~/Downloads/unsloth.mistral-alpaca*.gguf ~/ollama-models/alpaca-mistral/model.gguf
```

### Step 5.2 — Create the Modelfile
```bash
cat > Modelfile << 'EOF'
FROM /Users/YOUR_USERNAME/ollama-models/alpaca-mistral/model.gguf

TEMPLATE """### Instruction:
{{ .Prompt }}

### Response:
{{ .Response }}"""

PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM "You are a helpful AI assistant fine-tuned on the Alpaca instruction dataset."
EOF
```

### Step 5.3 — Create the Ollama model
```bash
ollama create alpaca-mistral -f Modelfile
```

### Step 5.4 — Test the model
```bash
ollama run alpaca-mistral "Explain what a transformer model is in 3 bullet points."
```

---

## Phase 6: Architecture Decision Record

Create `docs/adr/ADR-001-base-model-selection.md`:

```markdown
# ADR-001: Base Model Selection for QLoRA Fine-Tuning

**Date:** 2026-05-27
**Status:** Accepted

## Context
We need to fine-tune a language model on instruction-following data (Alpaca format) using a
free T4 GPU in Google Colab. Constraints: ≤16 GB VRAM, training must complete in <4 hours,
final ROUGE-L improvement ≥5% over baseline.

## Decision
Use **Mistral-7B-Instruct-v0.3** via Unsloth's pre-quantized 4-bit checkpoint.

## Alternatives Considered
| Model | VRAM (4-bit) | Training Speed | Notes |
|-------|-------------|----------------|-------|
| **Mistral-7B-Instruct** | ~5 GB | ~200 tok/s | No approval needed, strong baseline |
| Llama 3.1 8B Instruct | ~6 GB | ~180 tok/s | Requires Meta approval (~1 day wait) |
| Phi-3 Mini 3.8B | ~3 GB | ~300 tok/s | Smaller, lower quality ceiling |
| Falcon-7B | ~5 GB | ~150 tok/s | Older, weaker than Mistral |

## Rationale
- Mistral-7B is immediately accessible (no approval gate)
- Unsloth provides a pre-quantized 4-bit checkpoint that loads in <2 minutes on Colab T4
- Strong instruction-following baseline makes the fine-tuning improvement measurable
- LoRA r=16 gives enough expressiveness without overfitting on 1800 examples

## Consequences
- GGUF export at q4_k_m quality gives ~4 GB model file
- Ollama can run the GGUF on M1/M2 MacBook with ~8 GB RAM usage
- Trade-off: 4-bit quantization introduces slight quality degradation vs full precision
```

---

## Verification Checklist
- [ ] Google Colab T4 GPU connected (check: `torch.cuda.is_available() == True`)
- [ ] `01_dataset_prep.ipynb` creates 1800 train + 200 eval examples
- [ ] Dataset pushed to HF Hub (`YOUR_USERNAME/alpaca-2k`)
- [ ] Training completes without OOM error (watch Colab RAM meter)
- [ ] Training loss decreases over 3 epochs (visible in trainer output)
- [ ] LoRA adapter pushed to HF Hub (`YOUR_USERNAME/aiarch-finetuned-adapter`)
- [ ] GGUF file downloaded to local machine
- [ ] ROUGE-L improvement ≥ 5% over baseline (Cell 4 of evaluate notebook)
- [ ] `ollama create alpaca-mistral` succeeds
- [ ] `ollama run alpaca-mistral "..."` returns coherent response
- [ ] ADR-001 written

---

## Troubleshooting

**Out of memory error during training**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8 to compensate
- Use `max_seq_length=1024` instead of 2048

**Colab session disconnects mid-training**
- Enable "Stay awake" in Colab settings (Settings → Site settings)
- Or run `while True: pass` in a separate browser tab on Colab

**ROUGE-L improvement < 5%**
- Ensure you're comparing against the SAME base model (not a different variant)
- Try increasing `num_train_epochs` to 5
- Check that the eval dataset has the same format as training data

**`ollama create` fails with model file error**
- Verify the absolute path in `FROM` in the Modelfile is correct
- Use `ls ~/ollama-models/alpaca-mistral/` to confirm the .gguf file exists

**Hugging Face `401 Unauthorized`**
- Regenerate your HF token and re-run `huggingface-cli login`

---

## Next Steps → Project 06: RAG + LangChain
```bash
conda activate aiarch
mkdir -p ~/Documents/ai-journey/projects/06-rag-langchain/{src,data,docs/adr}
cd ~/Documents/ai-journey/projects/06-rag-langchain
# Ensure Ollama is running: ollama serve &
ollama pull nomic-embed-text  # If not already pulled
```
