# Neurodegenerative Clinical QA — Fine-tuned LLM

A structured clinical information system for neurodegenerative diseases,
built by fine-tuning Mistral-7B-Instruct using QLoRA. The model explains
symptoms, progression, and care guidance for Alzheimer's, Parkinson's,
ALS, Huntington's, and related dementias — while strictly refusing to
diagnose, prescribe, or express clinical certainty.

---

## What this project is

Most medical LLM projects fine-tune on general medical text and output
raw paragraphs. This project does something different:

- Domain restricted to **neurodegenerative diseases only**
- All outputs follow a **fixed clinical structure**
  (Condition / Information type / Disclaimer)
- A **safety boundary** is enforced — the model is trained and evaluated
  to never produce diagnostic statements or prescription advice
- The entire pipeline is **reproducible** — frozen dataset, seeded
  randomness, versioned splits

---

## Results

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| ROUGE-L | 0.1144 | 0.1080 |
| BERTScore F1 | 0.8659 | 0.8520 |
| Safety Compliance | 99.0% | **100.0%** |
| Structure Adherence | 15.0% | **95.0%** |

The headline result is structure adherence jumping from 15% to 95% —
the model learned to consistently produce the clinical output format
including condition labelling, information type headers, and medical
disclaimers. ROUGE-L and BERTScore staying near-identical confirms
factual quality was preserved through fine-tuning.

Training loss dropped from 2.29 → 1.02 over 188 steps (27 minutes,
RTX 3060 12GB).

---

## Diseases covered

Alzheimer's disease, Parkinson's disease, ALS, Huntington's disease,
Lewy body dementia, Frontotemporal dementia, Vascular dementia,
and general neurodegenerative conditions.

---

## Model

**Base:** `mistralai/Mistral-7B-Instruct-v0.2`  
**Fine-tuning:** QLoRA — 4-bit quantization (bitsandbytes) + LoRA
adapters (PEFT)  
**Trainable parameters:** 3,407,872 / 7,245,139,968 (0.047%)  
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

**Total:** 2400 pairs — 2160 train / 240 val  
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
prompt text (not a gold-standard reference answer) since no labeled
reference answers exist in this pipeline. They measure how well the
model stays on-topic relative to the question. The more meaningful
metrics for this project are safety and structure, which measure
what the fine-tuning was actually designed to improve.

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

### 1. ADNI data access — the first blocked path

The original plan included ADNI (Alzheimer's Disease Neuroimaging
Initiative) clinical notes as a dataset source. After reading the
Data Use Agreement carefully, ADNI prohibits redistribution of derived
artifacts — including models trained on the data — making it
incompatible with a public GitHub + HuggingFace deployment. The project
pivoted to fully open datasets only. Everything is now reproducible
without any access approval.

### 2. The dataset pipeline trade-off — pair collapse vs label noise

Getting the dataset right took significant iteration. The core tension
was a classic biomedical NLP problem:

- **Too strict a filter** → almost everything rejected → pair count
  collapses → dataset too small to train anything useful
- **Too flexible a filter** → more data → labels become noisy →
  model learns weak or incorrect medical signals

What stabilized the pipeline was a three-part fix: freezing the dataset
to `train.json` / `val.json` and stopping regeneration from raw sources,
locking all randomness to a single seeded `rng` object, and upsampling
smaller sources to balance contributions across all three datasets.

### 3. Question type distribution

After building the type classifier, ~70% of pairs landed in the
`general` fallback category. Multiple rounds of regex loosening
brought this to ~58%, which is the realistic floor — the remaining
general pairs are genuinely definitional questions that don't fit
any specific template. The sweet spot was accepting this distribution
rather than forcing mislabels.

### 4. Finding the max_length sweet spot

The core training constraint was the 12GB VRAM ceiling of the RTX 3060.

At `max_length=128` the model trained quickly across the full dataset,
but structured output markers (`**Condition:**`, `**Note:**`) were
getting truncated before the model could learn them — resulting in
0% structure adherence despite clean factual outputs.

At `max_length=256` with the full 2160 pairs the kernel crashed
repeatedly due to peak VRAM exceeding 12GB during the forward pass.

The working configuration was `max_length=256` with 1500 training
pairs + `paged_adamw_8bit` optimizer + `gradient_checkpointing`.
This kept peak VRAM inside the safe limit and gave the model enough
sequence length to see complete structured outputs during training.
Result: structure adherence jumped from 0% to 95%.

### 5. Kernel crash pattern — always start from Cell 1

A recurring issue during development was Cell 3 crashing when run
in isolation, but succeeding when run from Cell 1. The cause is CUDA
context fragmentation — previous cells leave GPU memory in an
inconsistent state. The fix is always restarting the kernel before
running Cell 3, so the VRAM starts clean.

### 6. Two-model evaluation on 12GB VRAM

Running inference for both baseline and fine-tuned models in a single
cell for comparison caused kernel crashes — two 4-bit Mistral-7B
instances simultaneously exceeded VRAM. The solution was a two-pass
architecture in Cell 5: load baseline → generate → delete + flush
CUDA cache → load fine-tuned → generate → delete + flush. Metrics
and plots run in Cell 6 after a kernel restart with no models loaded,
with BERTScore forced to CPU to avoid any GPU contention.

---

## Reproducibility

All randomness locked via `SEED = 42`. Dataset frozen to disk before
training. To reproduce:

```bash
pip install transformers peft bitsandbytes datasets evaluate \
            bert_score rouge_score matplotlib
```

Run cells in order: 1 → 2 → 3 → 4 → 5 → 6.  
**Restart kernel before Cell 3 and before Cell 6.**  
No restricted data required. All datasets load from HuggingFace Hub.

---

## Safety boundary

✅ Explains symptoms and conditions  
✅ Discusses disease progression and staging  
✅ Provides caregiver guidance  
✅ Recommends when to seek professional help  
✅ Includes medical disclaimer on every response  

❌ Never states "you have X"  
❌ Never prescribes or recommends specific medications  
❌ Never expresses diagnostic certainty  

Safety compliance measured quantitatively: 100% on evaluation set.

---

## Project context

Built as part of a self-directed AI/ML specialization alongside a
Biotechnology undergraduate degree. Project 7 in a structured roadmap
combining ML engineering with biomedical domain knowledge.

Deployment via HuggingFace Spaces (Gradio) planned post-MLOps
block completion.