# Neurodegenerative Clinical QA — Fine-tuned LLM

A structured clinical information system for neurodegenerative diseases,
built by fine tuning Mistral-7B-Instruct using QLoRA. The model explains
symptoms, progression, and care guidance for Alzheimer's, Parkinson's,
ALS, Huntington's and etc. while strictly refusing to
diagnose, prescribe or express clinical certainty.

---

## What this project is

Most medical LLM projects fine tune on general medical text and output
raw paragraphs. This project does something different:

- Domain restricted to **neurodegenerative diseases only**
- All outputs follow a **fixed clinical structure**
  (Condition / Information type / Disclaimer)
- A **safety boundary** is enforced the model is trained and evaluated
  to never produce diagnostic statements or prescription advice
- The entire pipeline is **reproducible** frozen dataset, seeded
  randomness, versioned splits

---

## Results

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| ROUGE-L | 0.1144 | 0.1080 |
| BERTScore F1 | 0.8659 | 0.8520 |
| Safety Compliance | 99.0% | **100.0%** |
| Structure Adherence | 15.0% | **95.0%** |

The headline result is structure adherence jumping from 15% to 95%
the model learned to consistently produce the clinical output format
including condition labelling, information type headers, and medical
disclaimers. ROUGE-L and BERTScore staying near-identical confirms
factual quality was preserved through fine-tuning.

Training loss dropped from 2.29 to 1.02 over 188 steps (27 minutes,
RTX 3060 12GB)
---

## Diseases covered

Alzheimer's disease, Parkinson's disease, ALS, Huntington's disease,
Lewy body dementia, Frontotemporal dementia, Vascular dementia
and general neurodegenerative conditions

---

## Model

**Base:** `mistralai/Mistral-7B-Instruct-v0.2`  
**Fine-tuning:** QLoRA 4-bit quantization (bitsandbytes) + LoRA
adapters (PEFT)  
**LoRA config:** rank=8, alpha=16, target: q_proj + v_proj  
**Hardware:** NVIDIA RTX 3060 12GB VRAM  
**Training:** 1 epoch, 1500 pairs, max_length=256, batch=1,
gradient accumulation=8, paged_adamw_8bit, ~27 minutes

---

## Dataset
Three open-license sources combined and balanced:

| Source | Type | Balanced pairs |
|--------|------|----------------|
| lavita/MedQuAD | Structured clinical Q&A (NIH) | 800 |
| qiaojin/PubMedQA | Research literature Q&A | 800 |
| keivalya/MedQuad-MedicalQnADataset + medalpaca/wikidoc | Patient-facing explanations | 800 |

**Total:** 2400 pairs 2160 train / 240 val  
**Training subset used:** 1500 pairs (balanced across sources)  
**Frozen to disk:** `Results/frozen_dataset/train.json` + `val.json`

Each pair is formatted into one of six structured templates by question
type: symptom, progression, treatment, diagnosis, caregiver, or general.
Every output ends with a fixed medical disclaimer.

---

## Evaluation metrics

| Metric | What it measures |
|--------|-----------------|
| ROUGE-L | Surface-level token overlap vs reference |
| BERTScore F1 | Semantic similarity via contextual embeddings |
| Safety Compliance % | No diagnostic claims + disclaimer present |
| Structure Adherence % | Correct clinical output format followed |

Note: ROUGE-L and BERTScore compare model output against the short
prompt text rather than a gold-standard reference answer, since no
labeled reference answers exist in this pipeline. The more meaningful
metrics for this project are safety and structure, which measure what
the fine-tuning was actually designed to improve.

---

## Cell structure

| Cell | Purpose |
|------|---------|
| Cell 1 | Data loading, filtering, formatting, balancing, freezing |
| Cell 2 | Baseline model — zero-shot inference + loss check |
| Cell 3 | QLoRA fine-tuning + training loss curve |
| Cell 4 | Fine-tuned model — quick inference test |
| Cell 5 | Two-pass inference — generate + save outputs for both models |
| Cell 6 | Metrics, evaluation, failure analysis, plots |

---

## Challenges

### 1. ADNI data access

The original plan included ADNI clinical notes as a dataset source.
After reading the Data Use Agreement, ADNI prohibits redistribution of
derived artifacts including models trained on the data, making it
incompatible with a public GitHub and HuggingFace deployment. The
project pivoted to fully open datasets only.

### 2. Dataset pipeline trade-off — pair collapse vs label noise

The core tension was a classic biomedical NLP problem. Too strict a
filter caused pair counts to collapse below a trainable threshold. Too
flexible a filter introduced noisy labels and weak medical signals.
What stabilized the pipeline was freezing the dataset to disk,
locking all randomness to a single seeded object and upsampling
smaller sources to balance contributions across all three datasets.

### 3. Question type distribution

After building the type classifier, roughly 70% of pairs landed in the
general fallback category. Multiple rounds of regex loosening brought
this to 58%, which is the realistic floor the remaining general pairs
are genuinely definitional questions that do not fit any specific
template. The sweet spot was accepting this distribution rather than
forcing mislabels.

### 4. Finding the max_length sweet spot

At max_length=128 the model trained quickly but structured output
markers were getting truncated before the model could learn them,
resulting in 0% structure adherence despite clean factual outputs.
At max_length=256 with the full 2160 pairs the kernel crashed due to
peak VRAM exceeding 12GB. The working configuration was max_length=256
with 1500 training pairs plus paged_adamw_8bit and
gradient_checkpointing, keeping peak VRAM within the safe limit.
Structure adherence jumped from 0% to 95%.

### 5. Kernel crash pattern

Cell 3 crashed repeatedly when run in isolation but succeeded when run
from Cell 1. The cause is CUDA context fragmentation from earlier cells
leaving GPU memory in an inconsistent state. The fix is always
restarting the kernel before Cell 3 so VRAM starts clean.

### 6. Two-model evaluation on 12GB VRAM

Running inference for both baseline and fine-tuned models in a single
cell caused kernel crashes  two 4-bit Mistral-7B instances
simultaneously exceeded VRAM. The solution was a two-pass architecture
in Cell 5: load baseline, generate, delete and flush CUDA cache, then
load fine-tuned and repeat. Metrics and plots run in Cell 6 after a
kernel restart with no models in memory. BERTScore is forced to CPU to
avoid GPU contention.

### 7. Inference provider deprecation — runtime model substitution

The deployment target was `mistralai/Mistral-7B-Instruct-v0.2`,
matching the fine-tuned base model. Four approaches failed in sequence
before the real cause became clear. By mid-2025, the legacy HuggingFace
serverless inference provider had dropped support for models above
roughly 10GB, and Mistral-7B-Instruct-v0.2 was removed from all
HuggingFace inference provider deployments entirely. The model page
states "This model isn't deployed by any Inference Provider." Every
error i faced token misconfiguration, env var propagation, auth
failures, 404s had been masking this single upstream fact.

The fix was runtime model substitution: `Qwen/Qwen2.5-7B-Instruct`,
same 7B class, available across  Together and Novita
The fine-tuning work remains on Mistral in the
notebook. Training and serving environments do not always use identical
model versions, and understanding why is part of shipping real projects.

---

## Reproducibility

All randomness locked via `SEED = 42`. Dataset frozen to disk before
training. To reproduce:

```bash
pip install transformers peft bitsandbytes datasets evaluate \
            bert_score rouge_score matplotlib
```

Run cells in order: 1, 2, 3, 4, 5, 6.  
Restart kernel before Cell 3 and before Cell 6.  
No restricted data required. All datasets load from HuggingFace Hub.

---

## Safety boundary

**Trained to do:**

- Explain symptoms and conditions
- Discuss disease progression and staging
- Provide caregiver guidance
- Recommend when to seek professional help
- Include a medical disclaimer on every response

**Trained to refuse:**

- Stating "you have X"
- Prescribing or recommending specific medications
- Expressing diagnostic certainty

Safety compliance measured quantitatively: 100% on evaluation set.
