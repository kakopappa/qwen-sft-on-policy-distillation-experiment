# Qwen3 On-Policy Distillation — Sharp CV-P09FX AC Manual

A hybrid SFT + on-policy distillation experiment teaching a small reasoning model
(Qwen3-0.6B) to answer questions about an air conditioner manual — **without the manual
ever appearing in the student's inference prompt**.

Builds on a prior QLoRA SFT experiment:
<https://github.com/kakopappa/Fine-tuning-LFM2.5-350M-on-Sharp-CV-P09FX-AC-Manual>

**Hardware:** Google Colab L4 GPU (24 GB VRAM)  
**Student:** `Qwen/Qwen3-0.6B` (LoRA r=32 for SFT, r=16 for distillation)  
**Teacher:** `Qwen/Qwen3-4B-Thinking-2507` (frozen, manual in context)  
**Synthetic data:** `deepseek-ai/DeepSeek-V4-Flash` via DeepInfra API

---

## Pipeline overview

```
AC manual PDF
     │
     ▼
sharp_cv_p09fx_manual_en.md        ← clean English-only Markdown
     │
     ▼
synth_data_gen.ipynb               ← Step 1: generate ~200 QA pairs with <think> blocks
     │  DeepSeek-V4-Flash (DeepInfra)
     ▼
ac_manual_synth_tra.jsonl
ac_manual_synth_val.jsonl
     │
     ▼
sft_qwen3.ipynb                    ← Step 2: SFT on synthetic CoT data
     │  Qwen3-0.6B + LoRA r=32
     ▼
qwen3_06b_ac_sft_lora/             ← saved adapter (best checkpoint)
     │
     ▼
onpolicy_distill_qwen3.ipynb       ← Step 3: on-policy distillation
     │  student rollouts → frozen teacher scores → reverse KL backprop
     ▼
qwen3_06b_ac_onpolicy_lora/        ← saved adapter
```

---

## File reference

| File | Purpose |
|---|---|
| `sharp_cv_p09fx_manual_en.md` | Cleaned English-only Markdown manual (upload to Colab) |
| `synth_data_gen.ipynb` | Step 1: generate synthetic QA pairs with DeepSeek |
| `sft_qwen3.ipynb` | Step 2: SFT Qwen3-0.6B on synthetic CoT data |
| `onpolicy_distill_qwen3.ipynb` | Step 3: on-policy distillation with Qwen3-4B teacher |
| `ac_manual_synth_tra.jsonl` | Synthetic training set (~165 pairs, ChatML format) |
| `ac_manual_synth_val.jsonl` | Synthetic val set (~32 pairs, ChatML format) |
| `qwen3_06b_ac_sft_lora/` | Saved SFT LoRA adapter (best checkpoint, epoch 4) |
| `qwen3_06b_ac_onpolicy_lora/` | Saved distillation LoRA adapter |

---

## Step 1 — Synthetic data generation (`synth_data_gen.ipynb`)

Generates diverse QA pairs from the manual using a strong reasoning model via the
DeepInfra API. The output is the SFT training set — each example has a `<think>`
reasoning block followed by a factual answer.

### Why synthetic data?

On-policy distillation only uses questions (generates answers live from the student).
But SFT requires completed (question, answer) pairs. The synthetic dataset serves as the
SFT training set and also provides diverse questions for the distillation loop.

### Two-pass generation

**Pass 1 — Batch generation for volume:**
Five diversity modes × 3 passes × 15 pairs/batch ≈ 225 pairs per run.

Diversity modes: direct fact lookup, metric/imperial unit variants, edge cases and
boundary conditions, procedural step-by-step, troubleshooting.

**Pass 2 — Per-question reasoning enrichment:**
Pass 1 think blocks contain *meta-reasoning* ("Let me plan 15 Q&A pairs…"), not
question-specific reasoning. Pass 2 re-asks each question individually so the model's
`reasoning_content` is grounded in that specific question.

### Prompt caching

The full manual is placed in the **system message** (constant across all calls).
Only the short user message (diversity instruction) changes each call. DeepInfra's
automatic prefix caching applies after the first call, saving ~140k tokens per run.

### Key config

| Parameter | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash` |
| `reasoning_effort` | `"high"` |
| Pairs per batch | 15 |
| Passes | 3 |
| Val ratio | 15% |

### Output format

```
{"text": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n{answer}<|im_end|>"}
```

---

## Step 2 — SFT (`sft_qwen3.ipynb`)

Fine-tunes Qwen3-0.6B on the synthetic CoT dataset. The model learns both the reasoning
format (`<think>` blocks citing manual sections) and the correct facts.

### Key design choices

**Completion-only loss:** system + user prompt tokens are masked (`labels = -100`).
Gradient flows only through the assistant turn (think block + answer).
`DataCollatorForCompletionOnlyLM` is not available in newer TRL versions; a manual
`tokenize_and_mask()` function implements the same behaviour.

**LoRA r=32:** higher rank than the distillation run to give more capacity for storing
reasoning patterns from the synthetic chains.

**Epoch count:** val loss was still decreasing after 3 epochs so 2 more were run
(total 5). Mild overfitting appeared at epoch 5 (train 0.50, val 1.009 — slight uptick
from epoch 4's 0.999). `load_best_model_at_end=True` restored the epoch 4 checkpoint.

### Training results

| Epoch | Train Loss | Val Loss | Token Accuracy |
|---|---|---|---|
| 1 | 1.22 | 1.18 | 71% |
| 2 | 0.89 | 1.09 | 73% |
| 3 | 0.90 | 1.07 | 74% |
| 4 | 0.78 | **0.999** | 75% ← best |
| 5 | 0.50 | 1.009 | 75% |

Runtime: ~210 seconds total on L4.

After SFT, the model correctly states:
> *"The minimum window width for the window panel is **22 inches (559mm)**."*

---

## Step 3 — On-policy distillation (`onpolicy_distill_qwen3.ipynb`)

Loads the SFT adapter as the student starting point and runs context-distillation.
The student generates rollouts without the manual; the frozen teacher scores those
same tokens with the manual in context; reverse KL backpropagates into the student LoRA.

```
              question
                 │
    ┌────────────┴──────────────┐
    ▼                           ▼
STUDENT prompt             TEACHER prompt
(no manual)                (manual in context)
both end with <think>\n    both end with <think>\n
    │ sample                     │
    ▼                            │
<think>                          │
  reasoning...                   │
</think>           ─────────────►│  score SAME tokens (frozen)
                                 ▼
              reverse KL: dense signal over every reasoning token
                                 │
                    backprop into student LoRA
```

### Why thinking models?

Non-thinking models produce ~100-token rollouts with ~1 useful fact token surrounded by
filler. Per-token-averaged KL dilutes the fact token's gradient by ~1/100. Thinking
models generate 50–200 token reasoning chains where nearly every token is meaningful
— signal density goes from ~1% to ~80%.

### Loading the SFT adapter as student starting point

```python
from peft import PeftModel
student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, torch_dtype=DTYPE).to(DEVICE)
student = PeftModel.from_pretrained(student, "./qwen3_06b_ac_sft_lora", is_trainable=True)
```

`is_trainable=True` keeps the LoRA weights unfrozen so the distillation optimizer can
update them. Without it, `from_pretrained` loads in inference mode (frozen).

### KL results

| Checkpoint | Held-out reverse KL |
|---|---|
| SFT adapter (before distillation) | 1.61 |
| After 5 epochs of distillation | **0.35** |

---

## Lessons learned

### 0. Questions must cover all topics

If the teacher never scored tokens on these topics, so no KL pressure was applied to them hence hallucinating identically in both the SFT-only and the on-policy adapter .

### 1. SFT first, distillation second

Pure on-policy distillation from a cold start is slow — random rollouts give noisy,
low-signal KL gradients. SFT on synthetic CoT data first gives the student a warm start:
the reasoning format is already learned and approximate facts are already in the weights.
Distillation then only corrects factual drift rather than teaching the format from scratch.

### 2. Lower KL does not mean better answers

After distillation KL dropped from 1.61 → 0.35 (significant improvement), but
generation quality got *worse* — think blocks looped endlessly. The SFT-only checkpoint
was more useful in practice despite its higher KL score.

Metric and quality can diverge because reverse KL is mode-seeking: the student
concentrates probability mass on patterns the teacher favours token-by-token, but
greedy decoding at inference can then fall into a local mode that scores well
step-by-step but never globally terminates.

### 3. Why small models loop during distillation

Three compounding reasons:

- **Reverse KL is mode-seeking.** The student assigns high probability to search-like
  phrases the teacher favours ("I found this section…"). It has no explicit pressure to
  generate `</think>`, which is just one token among thousands with diluted gradient.
- **SFT supervised the endpoint; distillation supervises the process.** SFT trained on
  *complete* think blocks — the model saw start → reasoning → `</think>` → answer as a
  unit. Distillation only pressures token-by-token distribution matching.
- **0.6B lacks context capacity to break the loop.** A 4B+ model can track "I already
  cited this section" over a long context. A 0.6B model loses that thread and finds
  repeating the same pattern locally optimal.

### 4. Two fixes for the looping problem

Both are now in `onpolicy_distill_qwen3.ipynb`:

**Fix A — Truncate rollouts at `</think>`:**
`student_rollout()` trims generation at the first `</think>` token. The student is
never trained on tokens past termination, so it never receives gradient reinforcing
loops. KL is computed only over reasoning tokens, which is where the fact signal lives.

```python
def _trim_at_think_end(gen):
    for pos in range(gen.shape[1]):
        if gen[0, pos].item() == THINK_END_ID:
            return gen[:, :pos + 1]
    return gen
```

**Fix B — Format penalty:**
If a rollout hits `MAX_NEW_TOKENS` without reaching `</think>`, a constant
`FORMAT_PENALTY = 0.5` is added to the KL loss. This gives a direct gradient signal
rewarding termination without needing a reward model.

```python
terminated = gen[0, -1].item() == THINK_END_ID
penalty = 0.0 if terminated else FORMAT_PENALTY
return kl + penalty
```

### 5. The teacher probe is mandatory

Always verify the teacher answers correctly with the manual in context before training.
If the teacher answers wrong, you distill wrong reasoning — the student's ceiling is
the teacher's quality on your task.

### 6. Thinking models share vocab — assert it

Token-level KL is only valid if both models use the same vocabulary mapping. Qwen3
family models share a tokenizer, but the notebook asserts this at runtime:

```python
assert tokenizer(probe)["input_ids"] == t_tok(probe)["input_ids"], \
    "Student and teacher tokenizers disagree — token-level KL would be meaningless."
```

### 7. Prompt caching saves significant API cost

Placing the full manual (~10k tokens) in the system message and varying only the user
message means all API calls after the first are served from cache. ~140k tokens saved
per generation run.

### 8. Attention mask warning is harmless, but fix it

`[transformers] The attention mask is not set…` appears because Qwen3's pad token = eos
token. For single-sequence inference there is no padding so behaviour is unaffected.
Fix: pass `attention_mask=torch.ones_like(prompt)` to all `.generate()` calls.

---

## Setup

```bash
# Colab install cell — run once, then Runtime → Restart session
!pip install "transformers>=4.51" "peft>=0.13" "trl>=0.12" accelerate datasets
```

Set `DEEPINFRA_API_KEY` in `synth_data_gen.ipynb` before running Step 1. Steps 2 and 3
are fully local — no API key required.

Upload `sharp_cv_p09fx_manual_en.md`, `ac_manual_synth_tra.jsonl`, and
`ac_manual_synth_val.jsonl` to Colab before running Steps 2 and 3.

---

## Prior experiment (LFM2 / T4)

The original experiment used `LFM2.5-350M` student + `LFM2-1.2B` teacher on a T4 GPU
with non-thinking models and a keyword-filtered manual context. Key finding: per-token
KL is dominated by filler tokens (~1 fact token per 100 total), so facts don't transfer
reliably. This Qwen3 experiment was designed specifically to solve that problem using
thinking models.
