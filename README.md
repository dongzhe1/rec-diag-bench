# rec-diag-bench

**A diagnostic framework for recommender systems — systematically measure what breaks under cold-start and long-tail conditions.**

`rec-diag-bench` is an experiment toolkit that instruments the full recommendation pipeline to answer one question: *where exactly does a recommender fail, and why?* It runs models under two complementary regimes (positive-controlled pool to isolate reranker quality; retrieval-realistic pool to measure coverage ceilings), then breaks down every metric by failure-mode scenario — item cold-start, user cold-start, long-tail, and warm — so you can see which component is the bottleneck.

The framework supports 15+ models (from popularity baselines to LLM rerankers), computes ranking metrics with bootstrap confidence intervals, and provides diagnostic tools for score separation, tie analysis, retrieval coverage, oracle ceilings, fusion headroom, and end-to-end decomposition. It works out of the box on MovieLens and can be extended to any dataset in CSV format.

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Walkthrough: your first diagnostic run](#walkthrough-your-first-diagnostic-run)
  - [Step 1: Build the data split](#step-1-build-the-data-split-cpu-only-5-seconds)
  - [Step 2: Run the baseline models](#step-2-run-the-baseline-models-cpu--gpu)
  - [Step 3: Read the results](#step-3-read-the-results)
  - [Step 4: Test retrieval coverage](#step-4-test-retrieval-coverage--the-real-bottleneck)
  - [Step 5: End-to-end](#step-5-end-to-end-the-full-picture)
- [What this library does](#what-this-library-does)
- [Key diagnostics](#key-diagnostics)
  - [1. Reranker quality assessment](#1-reranker-quality-assessment-positive-controlled-pool)
  - [2. Retrieval coverage](#2-retrieval-coverage-retrieval-realistic-full-catalogue)
  - [3. End-to-end decomposition](#3-end-to-end-decomposition)
  - [4. Oracle ceiling & fusion headroom](#4-oracle-ceiling--fusion-headroom)
  - [5. Graph-evidence decomposition (GALA ablation)](#5-graph-evidence-decomposition-gala-ablation)
  - [6. Statistical significance](#6-statistical-significance)
  - [7. LLM scale study](#7-llm-scale-study)
  - [8. Prompt & scoring variant sensitivity](#8-prompt--scoring-variant-sensitivity)
  - [9. Learned Hybrid Fusion (LHF)](#9-learned-hybrid-fusion-lhf)
  - [10. Multi-seed aggregation](#10-multi-seed-aggregation)
  - [11. LHF → downstream ranker](#11-lhf--downstream-ranker-is-the-llm-even-needed-for-reranking)
  - [12. Recall@K curve comparison](#12-recallk-curve-comparison)
  - [13. Quota allocation grid search](#13-quota-allocation-grid-search)
- [Sensitivity ablations](#sensitivity-ablations-built-into-the-main-pipeline)
- [Data format (for custom datasets)](#data-format-for-custom-datasets)
- [Output files](#output-files)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Workflow tips](#workflow-tips)
- [Citation](#citation)
- [License](#license)

## Quick start

```bash
# Install
pip install -r requirements.txt

# Run the full pipeline — downloads data, builds the split, evaluates all models
python scripts/run_dataset.py \
  --dataset ml-100k \
  --data_dir data \
  --output_dir outputs \
  --config configs/local.yaml \
  --top_k 50 \
  --seed 42
```

This single command downloads MovieLens-100k, builds temporal train/valid/test splits with cold-start scenario labels, evaluates 10+ models, and writes all metrics to `outputs/ml-100k-top50-s42/`. See the [Walkthrough](#walkthrough-your-first-diagnostic-run) below for a step-by-step breakdown of what happens at each stage and how to interpret the outputs.

## Installation

### CPU-only (baselines only: popularity, ItemKNN, TF-IDF, Markov)

```bash
pip install numpy pandas scipy scikit-learn pyyaml tqdm matplotlib tabulate psutil numba
pip install -e .
```

### GPU (adds BPR, LightGCN, SASRec, SBERT, cross-encoder, LLM reranker)

```bash
# Core stack
pip install numpy pandas scipy scikit-learn pyyaml tqdm matplotlib tabulate psutil numba

# PyTorch (match your CUDA version; see pytorch.org)
pip install torch>=2.1

# HuggingFace ecosystem — all models downloaded automatically from HuggingFace Hub
pip install transformers accelerate datasets huggingface_hub safetensors tokenizers sentencepiece

# Embedding & reranking
pip install sentence-transformers

# Optional: 4-bit quantization for large LLMs
pip install bitsandbytes

# Install the library itself
pip install -e .
```

> **Note:** All models (SBERT, cross-encoder, LLM) are downloaded automatically from HuggingFace Hub on first use — no manual download needed. The config file uses public model IDs like `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`, `Qwen/Qwen3.5-0.8B`.

For a pinned, GPU-server-tested dependency list, see `requirements.txt`.

## Walkthrough: your first diagnostic run

Let's walk through a complete experiment on **MovieLens-100k** (smallest dataset, auto-downloaded, ~30 seconds on GPU). This will show you exactly what happens at each stage, what files are produced, and how to read them.

### Step 1: Build the data split (CPU-only, 5 seconds)

```bash
python scripts/run_dataset.py \
  --dataset ml-100k \
  --data_dir data \
  --output_dir outputs \
  --config configs/local.yaml \
  --top_k 50 \
  --seed 42 \
  --split_only --force_split
```

`--split_only` means "build the split and candidate pool, then exit before scoring any models." `--force_split` forces re-processing even if an old split exists (remove it for incremental work).

**What happens under the hood:**

1. **Download:** The script downloads `ml-100k.zip` from GroupLens to `data/raw/` and extracts it.
2. **Filter:** Interactions with rating < 4.0 are dropped; users with < 5 interactions and items with < 3 interactions are removed iteratively.
3. **Temporal split:** All remaining interactions are sorted by timestamp. The earliest 70% becomes `train`, the next 10% becomes `valid`, the final 20% becomes `test` (ratios controlled by `train_ratio` / `valid_ratio` in the config).
4. **Cold-start labeling:** For each test row, the script computes how many times that user and item appeared in training, then assigns mutually-exclusive labels:

   | Label | Definition | Meaning |
   |-------|-----------|---------|
   | `is_item_new` | `train_count == 0` | Item never seen in training — genuinely novel |
   | `is_item_cold` | `0 < train_count ≤ bottom 5%` | Item exists but extremely rare |
   | `is_long_tail` | `bottom 5% < train_count < bottom 20%` | Niche item, not mainstream |
   | `is_user_cold` | `train_count ≤ bottom 5%` | User has very little history |
   | `is_warm` | none of the above | User has history, item is popular |

5. **Candidate pool:** For each test user, the script builds a pool of 1 positive item + 49 sampled negatives (never seen by that user in training), writing `candidates.csv`.

**Check the output:**

```bash
ls data/processed/ml-100k/s42/
# train.csv  valid.csv  test.csv  candidates.csv  items_mapped.csv  users_mapped.csv  item_stats.csv
```

The script prints a summary like:

```
Split summary
  train interactions  : 56,789
  valid interactions  : 5,432
  test  interactions  : 12,345
  items in vocab      : 1,682
Derived thresholds (from train distribution)
  item_new  : train_count == 0              → 12.3% of items
  item_cold : train_count <= 3.0  (P5%)    → 5.1% of items
  long_tail : train_count <  12.0 (P20%)   → 15.2% of items
  user_cold : train_count <= 5.0  (P5%)    → 5.0% of users
Test  failure-mode counts
       is_item_new   234
      is_item_cold    89
      is_long_tail   1,523
      is_user_cold   617
            is_warm  9,882
```

This tells you: 234 test interactions involve brand-new items (12.3% of the catalogue), and 1,523 are long-tail items — these are the failure modes the benchmark will stress-test.

**Key files to inspect:**

```bash
# The test set — one row per evaluation target
head data/processed/ml-100k/s42/test.csv
# Columns: user_idx, item_idx, timestamp, train_user_count, train_item_count,
#          is_item_new, is_item_cold, is_long_tail, is_user_cold, is_warm

# The candidate pool — one row per (user, candidate) pair
head data/processed/ml-100k/s42/candidates.csv
# Columns: user_idx, item_idx, label (1=positive, 0=negative)

# Item metadata with text for content-based models
head data/processed/ml-100k/s42/items_mapped.csv
# Columns: item_id, item_idx, title, text
```

### Step 2: Run the baseline models (CPU + GPU)

Now run the evaluation on the split you built in Step 1. The command is exactly the [Quick Start](#quick-start) invocation — it finds the existing split (no `--force_split` needed) and scores every enabled model on the candidate pool. By default (`local.yaml`), this includes:

| Model | Type | Device |
|-------|------|--------|
| Popularity | Global popularity rank | CPU |
| ItemKNN | Item-item collaborative filtering | CPU |
| TF-IDF | Content-based (item titles) | CPU |
| Markov | First-order Markov chain | CPU |
| BPR | Matrix factorization | GPU |
| LightGCN | Graph convolution | GPU |
| SASRec | Self-attention sequential | GPU |
| SBERT | Dense retrieval (BGE-small) | GPU |
| Graph-Aware | LightGCN + TF-IDF + Popularity fusion | GPU |
| Cross-Encoder | BGE-reranker-base | GPU |
| LLM | Qwen3.5-0.8B (logprob scoring) | GPU |
| GALA | LLM + graph evidence | GPU |
| Two-Tower | User/item text encoders with InfoNCE loss | GPU |

> **Two-Tower** is a standalone module (`src/coldtail/recommenders/two_tower.py`) not wired into the default pipeline — it trains a lightweight projection head on a frozen text encoder for full-catalogue ANN retrieval. See the module docstring for usage.

To skip expensive models, toggle them off in `configs/local.yaml` under `baselines:` (set `run_sasrec: false`, etc.) or under `rerank:` (set `sample_users: 0`).

**Check the output:**

```bash
ls outputs/ml-100k-top50-s42/
# popularity_metrics.csv  lightgcn_metrics.csv  llm_metrics.csv  gala_metrics.csv  ...
# all_model_metrics.csv   all_model_subgroup_metrics.csv
# timing_summary.csv      scenario_counts.csv    report.md
```

### Step 3: Read the results

**Overall metrics** (`all_model_metrics.csv`):

```bash
head outputs/ml-100k-top50-s42/all_model_metrics.csv
```

| model | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | mrr@10 | ... |
|-------|----------|-----------|-----------|--------|---------|--------|-----|
| popularity | 0.034 | 0.051 | 0.089 | 0.023 | 0.029 | 0.021 | ... |
| lightgcn | 0.142 | 0.201 | 0.298 | 0.098 | 0.119 | 0.087 | ... |
| llm | 0.156 | 0.218 | 0.316 | 0.107 | 0.131 | 0.095 | ... |

The LLM does better than LightGCN here — but remember, this is on a **positive-controlled pool** where the gold item is guaranteed present. The LLM is a better reranker, but...

**Subgroup breakdown** (`all_model_subgroup_metrics.csv`) reveals the failure modes:

| model | scenario | recall@10 | ndcg@10 | num_users |
|-------|----------|-----------|---------|-----------|
| lightgcn | is_item_new | 0.023 | 0.014 | 234 |
| llm | is_item_new | **0.071** | 0.047 | 234 |
| lightgcn | is_warm | 0.245 | 0.148 | 9882 |
| llm | is_warm | 0.256 | 0.154 | 9882 |

The LLM's advantage is largest on `is_item_new` (3× LightGCN!) — but recall@10 is still only 7.1%. Can retrieval even surface these items?

**Score separation** (`{model}_metrics.csv`) includes Cohen's d, Spearman r, and overlap coefficient — diagnostic numbers that tell you whether the model can distinguish relevant from irrelevant candidates at all. A Cohen's d near 0 means the score distributions of positives and negatives completely overlap (the model is effectively random).

**Tie diagnostics** (`{model}_metrics.csv`, `tie_rate`) show what fraction of users have their gold item tied in score with at least one negative. High tie rates mean the headline metric is unreliable due to random tie-breaking.

**The human-readable report:**

```bash
cat outputs/ml-100k-top50-s42/report.md
```

### Step 4: Test retrieval coverage — the real bottleneck

```bash
python scripts/run_retrieval.py \
  --dataset ml-100k \
  --seed 42 \
  --retrieval_n 50 \
  --config configs/local.yaml
```

This lets each retriever scan the **full catalogue** — no positive guarantee. The gold item must be found by the retriever itself.

**Check the output:**

```bash
cat outputs/ml-100k-retrieval-s42-N50/all_model_metrics.csv
```

Here, `recall@50` means "what fraction of test users have their gold item in the retriever's top-50?" This is your **retrieval ceiling** — the LLM reranker can never exceed it.

Typical finding: coverage@50 = 5–15% on cold-dominated domains. That means even a perfect reranker is capped at 5–15% recall. The bottleneck is retrieval, not reranking.

Compare per-scenario:

```bash
cat outputs/ml-100k-retrieval-s42-N50/all_model_subgroup_metrics.csv
```

You'll see coverage on `is_item_new` is often near 0% — brand-new items are structurally invisible to interaction-based retrievers.

### Step 5: End-to-end (the full picture)

```bash
# First, write realistic pools from each retriever
python scripts/run_retrieval.py \
  --dataset ml-100k --seed 42 --retrieval_n 50 \
  --config configs/local.yaml --write_pools \
  --retrievers lightgcn,sbert

# Then rerank each pool with the LLM
python scripts/run_end2end.py \
  --dataset ml-100k --seed 42 --top_k 50 \
  --config configs/local.yaml \
  --retrievers lightgcn,sbert
```

This produces `outputs/ml-100k-e2e-s42/end2end_summary.csv` with one row per retriever, decomposing end-to-end recall into:

```
e2e_recall@10 = coverage@50 × conditional_rerank_success
```

When coverage is low, end-to-end recall is low — the reranker can't help if the item is not in the pool.

### Summary: what you learned

| Step | Script | Question answered |
|------|--------|-------------------|
| 1 | `run_dataset.py --split_only` | How are cold/tail/warm defined? What's the scenario distribution? |
| 2 | `run_dataset.py` | How good is each model as a reranker (gold guaranteed)? |
| 3 | (read CSVs) | Does the LLM outperform CF baselines on cold items? Are scores separable? |
| 4 | `run_retrieval.py` | What fraction of test items can retrieval *actually find*? |
| 5 | `run_end2end.py` | What's the realistic end-to-end recall when retrieval feeds the LLM? |

---

## What this library does

This library instruments the standard recommendation pipeline to answer a specific question: **when an LLM-based recommender fails, is it because the LLM can't distinguish relevant items (reranker failure), or because retrieval never surfaces the right item (retrieval failure)?**

It runs under two complementary regimes:

| Regime | How the candidate pool is built | What it measures |
|--------|--------------------------------|------------------|
| **Positive-controlled** | 1 gold item + K−1 sampled negatives per user (gold guaranteed present) | Reranker quality in isolation |
| **Retrieval-realistic** | Each retriever scans the full catalogue; no gold guarantee | Retrieval coverage (ceiling) |

## Key diagnostics

### 1. Reranker quality assessment (positive-controlled pool)

```bash
python scripts/run_dataset.py --dataset ml-100k --top_k 50 --seed 42 --config configs/local.yaml
```

Runs up to 15 models (popularity, ItemKNN, TF-IDF, Markov, BPR, LightGCN, SASRec, SBERT, cross-encoder, LLM, plus GALA variants) on a pool where the gold item is guaranteed present. Produces:

- **Per-model recall@k / NDCG@k / MRR@k** with bootstrap confidence intervals
- **Subgroup breakdowns** by `is_item_new`, `is_item_cold`, `is_long_tail`, `is_user_cold`, `is_warm`
- **Score separation diagnostics**: Cohen's d, Spearman r, overlap coefficient — answers "can the model even tell relevant from irrelevant?"
- **Tie diagnostics**: best-case vs worst-case ranking under random tie-breaking
- **Exposure concentration**: Gini coefficient, unique items exposed.

### 2. Retrieval coverage (retrieval-realistic, full catalogue)

```bash
python scripts/run_retrieval.py --dataset ml-100k --seed 42 --retrieval_n 50 --config configs/local.yaml
```

Lets 13+ retrievers scan the full catalogue with NO gold guarantee. Measures:

- **Coverage@N**: fraction of test users whose gold item appears in the retriever's top-N
- **Pool coverage**: fraction who have their gold item *anywhere* in the pool
- **Median true rank**: how deep the gold item sits in each retriever's ranking
- **Per-scenario coverage**: breakdown by cold/tail/warm

This directly answers "is the failure retrieval or rerank?" — if coverage@200 is 5%, your LLM reranker is capped at 5% recall, no matter how good it is.

### 3. End-to-end decomposition

```bash
python scripts/run_end2end.py --dataset ml-100k --seed 42 --top_k 50 \
  --config configs/local.yaml --retrievers lightgcn,sbert,fusion,cara
```

Combines retrieval-realistic pools with LLM reranking to measure:

```
end-to-end recall@10 = retrieval coverage@N × conditional rerank success rate
```

The output (`end2end_summary.csv`) shows exactly where the bottleneck sits for each retriever.

### 4. Oracle ceiling & fusion headroom

```bash
python scripts/analyze_oracle_ceiling.py --dataset ml-100k --seed 42
```

Measures the oracle union of all retrievers (upper bound on what retrieval can achieve) and compares it against heuristic fusions (RRF, interleave, CARA) — quantifying how much headroom remains before retrieval saturates.

### 5. Graph-evidence decomposition (GALA ablation)

```bash
# GALA variants are automatically run as part of run_dataset.py when llm reranking is enabled.
# For post-hoc analysis:
python scripts/analyze_rq3.py --dataset ml-100k --seed 42
```

Ablates the LLM's graph evidence components across four variants to determine whether LLM improvements come from graph reasoning or from a tail-item prior learned during pre-training:

| Variant | What is ablated |
|---------|----------------|
| `gala` | Full GALA: co-occurrence graph + user history + item metadata → LLM prompt |
| `gala_no_evidence` | Removes all graph evidence; LLM sees only item metadata + user history |
| `gala_no_cooccur` | Removes co-occurrence graph; keeps user history + item metadata |
| `gala_no_tail` | Removes tail-prior (pre-training frequency signal); keeps graph evidence |

Compare `gala` vs `gala_no_evidence` to measure the total graph contribution. Compare `gala_no_cooccur` vs `gala_no_tail` to disentangle co-occurrence structure from the tail-item frequency prior.

### 6. Statistical significance

```bash
# Paired bootstrap + McNemar for end-to-end comparisons
python scripts/paired_bootstrap.py --dataset ml-100k --seed 42

# Bootstrap confidence intervals for ranking metrics
python scripts/end2end_ci.py --dataset ml-100k --seed 42
```

### 7. LLM scale study

```bash
python scripts/run_second_llms.py --dataset ml-100k --seed 42 --top_k 50 \
  --config configs/local.yaml
```

Tests whether scaling the LLM (8B → 32B → 70B) changes the conclusion. Models are downloaded automatically from HuggingFace Hub.

### 8. Prompt & scoring variant sensitivity

```bash
python scripts/run_rerank_variants.py --dataset ml-100k --seed 42 --top_k 50 \
  --config configs/local.yaml
```

Compares logprob-based scoring (P("Yes")) against generate-based (0–100 integer parsing), yes-no, listwise, and pairwise prompts — testing whether the failure mode is prompt- or scoring-dependent.

### 9. Learned Hybrid Fusion (LHF)

```bash
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42
```

Trains a lightweight fusion model (HistGradientBoosting or logistic regression) on the validation set to combine multiple retrievers' scores. Measures how much of the oracle headroom is practically realizable.

**LHF ablation flags:**

```bash
# Drop all user/item metadata — keep only retriever features
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42 --no_metadata

# Drop only cold-start signals (item_new, item_pop)
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42 --no_cold_metadata

# Keep only rank features (no raw scores)
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42 --rank_only

# Keep only score features (no rank signals)
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42 --score_only

# Exclude text retrievers from the union
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42 --no_text_retrievers

# Exclude CF retrievers from the union
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42 --no_cf_retrievers
```

These ablations answer: what drives the fusion gain — metadata, cold-start signals, rank structure, or raw scores? And which retriever family contributes more?

### 10. Multi-seed aggregation

```bash
python scripts/aggregate_multiseed.py --dataset ml-100k
```

Aggregates results across seeds (1, 42, 2026) with confidence intervals for robust reporting.

### 11. LHF → downstream ranker (is the LLM even needed for reranking?)

```bash
# Stage 1: train LHF and rank the pool
python scripts/run_learned_fusion.py --dataset ml-100k --seed 42

# Stage 2: train a non-LLM second-stage ranker (LightGBM) on the LHF-ranked pool
python scripts/run_lhf_downstream.py --dataset ml-100k --seed 42 --data_dir data

# Stage 3: paired significance test — LHF-only vs LHF→LightGBM vs LHF→LLM
python scripts/run_downstream_paired.py --dataset ml-100k --seed 42 \
  --data_dir data --output_dir outputs --k 10
```

Ablation: if a simple LightGBM ranker trained on the LHF pool performs comparably to the prompt-level LLM reranker, then the LLM's advantage is not about reasoning — it's about the pool quality. This tests whether the bottleneck is the **reranker** (LLM vs non-LLM) or the **retrieval pool** itself.

### 12. Recall@K curve comparison

```bash
python scripts/run_recallk_curve.py --dataset ml-100k --seed 42 --data_dir data
```

Compares recall across K=10/50/100/200/500 for every retriever plus fusions (RRF, CARA, LHF, oracle union), with per-scenario breakdowns. Shows how coverage grows with budget and where each method saturates. Post-hoc, CPU-only — reads existing pool CSVs.

### 13. Quota allocation grid search

```bash
python scripts/run_quota_allocation.py --dataset ml-100k --seed 42 --data_dir data
```

Given multiple retrievers and a fixed total budget N, searches for the optimal allocation of candidates per retriever. Compares equal-quota, CF-heavy, text-heavy, CARA, and validation-tuned vectors. Post-hoc, CPU-only.

### 14. Exploratory probes (GO / NO-GO)

Three lightweight probe scripts test hypotheses about retrieval-side LLM use before committing to a full method:

```bash
# Probe 1: LLM-generated intents as retrieval queries for cold items
python scripts/probe_intent_retrieval.py --dataset ml-100k --n_users 200 \
  --encoder BAAI/bge-large-en-v1.5 --llm Qwen/Qwen3-8B

# Probe 2: Freshness / trend-aware candidate generation
python scripts/probe_freshness_retrieval.py --dataset ml-100k --n_users 200

# Probe 3: Confidence-gated selective LLM reranking
python scripts/probe_selective_rerank.py --dataset ml-100k --n_users 200
```

Each probe tests a single hypothesis with a small user sample — designed for a quick GO/NO-GO decision before scaling up. GPU required (LLM generation + embedding).

### Post-hoc utility scripts

These read existing outputs — no GPU, no re-computation:

```bash
# Per-domain coverage diagnostic table
python scripts/diagnose_coverage.py --dataset ml-100k --seed 42

# Build CARA pool from pre-computed retriever pools
python scripts/build_cara_from_pools.py --dataset ml-100k --seed 42

# Merge metrics across multiple runs / datasets
python scripts/aggregate_metrics.py --dataset ml-100k
```

---

## Sensitivity ablations (built into the main pipeline)

These don't need separate scripts — they use flags on `run_dataset.py`:

### Negative sampling: uniform vs popularity-weighted

```bash
python scripts/run_dataset.py --dataset ml-100k --top_k 50 --seed 42 \
  --config configs/local.yaml --negative_sampling popularity
```

Tests whether using popularity-weighted "hard" negatives (instead of uniform) changes the relative model rankings. Writes to a separate output dir `…-negpopularity/`.

### History window: how much user context does the LLM need?

```bash
for h in 5 10 20; do
  python scripts/run_dataset.py --dataset ml-100k --top_k 50 --seed 42 \
    --config configs/local.yaml --max_history $h
done
```

Truncates each test user's history to the last `h` interactions before building the LLM prompt. Tests whether the LLM's cold-start advantage comes from long history context.

### Top-K budget: how does reranker quality degrade with more distractors?

```bash
for K in 10 50 100 200; do
  python scripts/run_dataset.py --dataset ml-100k --top_k $K --seed 42 \
    --config configs/local.yaml
done
```

The candidate pool at budget K contains 1 positive + K−1 negatives. Larger K = harder task. Measures whether the LLM maintains its advantage as the pool grows.

---

## Data format (for custom datasets)

MovieLens datasets (`ml-100k`, `ml-1m`, `ml-20m`, `ml-25m`, `ml-32m`) are **auto-downloaded** — no preparation needed. For other datasets, prepare two CSV files in `data/raw/<dataset>/`:

**`interactions.csv`**

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int/str | User identifier |
| `item_id` | int/str | Item identifier |
| `rating` | float | Interaction strength (≥ threshold = positive) |
| `timestamp` | int/float | Unix timestamp or ordinal for temporal splitting |

**`items.csv`**

| Column | Type | Description |
|--------|------|-------------|
| `item_id` | int/str | Item identifier (matches interactions) |
| `title` | str | Item title/name |
| `text` | str | Item description for content-based models (TF-IDF, SBERT) |

### Supported datasets

| Prefix | Example | Source |
|--------|---------|--------|
| `ml-*` | `ml-100k`, `ml-1m`, `ml-20m`, `ml-25m`, `ml-32m` | [MovieLens](https://grouplens.org/datasets/movielens/) — auto-downloaded |
| `amazon-*` | `amazon-videogames` | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — manual CSV |
| `yelp-*` | `yelp-Philadelphia-Restaurants` | [Yelp Academic Dataset](https://www.yelp.com/dataset) — manual CSV |
| `mind-*` | `mind-small` | [MIND News](https://msnews.github.io/) — uses impression-based hard negatives |

---

## Output files

Each run produces in `outputs/<dataset>-top<K>-s<seed>/`:

| File | Content |
|------|---------|
| `{model}_scores.csv` | Per-user scored candidates (model = `popularity`, `lightgcn`, `llm`, `gala`, …) |
| `{model}_metrics.csv` | Aggregate metrics + score separation + tie diagnostics |
| `{model}_subgroup_metrics.csv` | Metrics broken down by cold/tail/warm scenario |
| `{model}_metrics_ci.csv` | Bootstrap confidence intervals (95% CI) |
| `all_model_metrics.csv` | Combined metrics across all models |
| `all_model_subgroup_metrics.csv` | Combined subgroup metrics |
| `timing_summary.csv` | Latency, throughput, GPU memory per model |
| `candidate_size_summary.csv` | Candidate pool statistics |
| `scenario_counts.csv` | User counts per failure-mode scenario |
| `report.md` | Human-readable summary |

Retrieval runs produce `outputs/<dataset>-retrieval-s<seed>-N<n>/` with analogous files where `recall@k` = pool coverage.

---

## CLI reference

### `run_dataset.py` — main pipeline

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Dataset name (`ml-100k`, `ml-20m`, `amazon-videogames`, …) |
| `--data_dir` | `data` | Root directory for raw/processed data |
| `--output_dir` | `outputs` | Output root |
| `--config` | `configs/local.yaml` | YAML configuration file |
| `--top_k` | *required* | Per-user candidate budget (50/100/200/500) |
| `--seed` | `42` | Random seed |
| `--max_history` | *None* | Truncate user history to last N interactions |
| `--force_split` | *off* | Force raw prep + re-split even if artifacts exist |
| `--split_only` | *off* | Build split + candidate pool, exit before model evaluation |
| `--negative_sampling` | `uniform` | Pool negative scheme: `uniform` or `popularity` |
| `--with_retrieval` | *off* | Also run retrieval-realistic coverage |

### `run_retrieval.py` — retrieval coverage

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Dataset name (split must already exist) |
| `--seed` | `42` | Seed of the split to reuse |
| `--retrieval_n` | `200` | Per-user retrieved pool size (no gold guarantee) |
| `--retrievers` | *all 13* | Comma-separated subset of retrievers |
| `--write_pools` | *off* | Save each retriever's pool for downstream reranking |

### `run_end2end.py` — end-to-end retrieve→rerank

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Dataset name |
| `--seed` | `42` | Seed |
| `--top_k` | `200` | Pool size |
| `--retrievers` | `lightgcn,sbert,fusion,fusion_il,cara` | Retrievers to evaluate |
| `--n_users` | *None* | Evaluate on N sampled test users |

---

## Configuration

Edit `configs/local.yaml` to control:

- **Split parameters:** interaction cutoffs, cold-start quantile thresholds, test set size
- **Training:** embedding dimension, epochs, batch size, learning rate, early stopping patience
- **Models to run:** toggle individual baselines on/off under `baselines:`
- **SBERT:** model name (any SentenceTransformer-compatible model from HuggingFace Hub)
- **Reranking:** LLM model name (any CausalLM from HuggingFace Hub), cross-encoder model, batch sizes, scoring mode (`logprob` or `generate`)
- **GALA:** graph evidence weights (alpha/beta/gamma/lambda)

Dataset-specific overrides are under `dataset_specific:` (e.g., different rating thresholds for Amazon vs Yelp).

All model paths use public HuggingFace Hub IDs — models are downloaded automatically on first use. No manual download or local path configuration needed.

---

## Project structure

```
rec-diag-bench/
├── scripts/                    # Executable experiment scripts
│   ├── run_dataset.py          # Main pipeline (positive-controlled, 15 models)
│   ├── run_retrieval.py        # Retrieval-realistic coverage (13+ retrievers)
│   ├── run_end2end.py          # End-to-end retrieve → LLM rerank
│   ├── run_second_llms.py      # LLM scale study (8B/32B/70B)
│   ├── run_rerank_variants.py  # Prompt/scoring variant sensitivity
│   ├── run_learned_fusion.py   # Learned Hybrid Fusion (LHF)
│   ├── run_recallk_curve.py    # Recall@k curve comparison
│   ├── run_quota_allocation.py # Multi-retriever quota allocation
│   ├── analyze_rq3.py          # Graph-evidence decomposition
│   ├── analyze_oracle_ceiling.py  # Oracle union vs. fusion headroom
│   ├── diagnose_coverage.py    # Per-domain coverage diagnostic table
│   ├── build_cara_from_pools.py   # CARA pool builder
│   ├── end2end_ci.py           # End-to-end confidence intervals
│   ├── paired_bootstrap.py     # Paired bootstrap + McNemar
│   ├── aggregate_metrics.py    # Cross-run metric aggregation
│   ├── aggregate_multiseed.py  # Multi-seed aggregation with CIs
│   ├── probe_intent_retrieval.py     # LLM-intent retrieval probe
│   ├── probe_freshness_retrieval.py  # Freshness-aware retrieval probe
│   ├── probe_selective_rerank.py     # Selective LLM reranking probe
│   └── _bootstrap.py           # Path setup (imported by all scripts)
├── src/coldtail/               # Core library
│   ├── config.py               # YAML config loading
│   ├── metrics.py              # Recall, NDCG, MRR, bootstrap CIs, score separation, ties
│   ├── utils.py                # Seed, device, normalization helpers
│   ├── graph_adapt.py          # Score fusion & normalization
│   ├── data/
│   │   └── split.py            # Temporal splits, candidate pool construction, scenario labels
│   ├── recommenders/           # All recommendation models
│   │   ├── base.py             # Dataset shape, user-item matrix, sampling
│   │   ├── popularity.py       # Global popularity baseline
│   │   ├── itemknn.py          # Item-item collaborative filtering
│   │   ├── content_tfidf.py    # TF-IDF content-based retrieval
│   │   ├── markov.py           # First-order Markov chain
│   │   ├── bpr.py              # Bayesian Personalized Ranking (MF)
│   │   ├── lightgcn.py         # LightGCN (graph convolution)
│   │   ├── sasrec.py           # Self-attentive sequential recommender
│   │   ├── sbert_retrieval.py  # Sentence-BERT dense retrieval
│   │   └── two_tower.py        # Two-tower (user/item encoder) architecture
│   └── experiments/            # Experiment orchestration
│       ├── common.py           # GPU/CPU monitoring
│       ├── run_baselines.py    # Baseline orchestrator
│       ├── run_graph_aware.py  # LightGCN+TF-IDF+popularity fusion
│       ├── run_cross_encoder.py   # Cross-encoder reranker
│       ├── run_llm_rerank.py      # LLM reranker + GALA variants
│       ├── gala_evidence.py       # Graph evidence builder for GALA
│       ├── gala_ablation.py       # GALA ablation wrapper
│       ├── retrieval_coverage.py  # Full-catalogue retrieval coverage engine
│       ├── failure_analysis.py    # Score separation & tie diagnostics
│       ├── timing.py              # Per-model latency collector
│       └── report.py              # Human-readable report generator
├── configs/
│   └── local.yaml              # Configuration (all models from HuggingFace Hub)
├── pyproject.toml
└── requirements.txt
```

---

## Workflow tips

### Pre-build splits on CPU before GPU sweeps

```bash
# Build the seed-specific split once (CPU-only, no model evaluation)
python scripts/run_dataset.py --dataset ml-100k --data_dir data --output_dir outputs \
  --config configs/local.yaml --top_k 200 --seed 42 --split_only --force_split

# Reuse the same split at different budgets
for K in 50 100 200; do
  python scripts/run_dataset.py --dataset ml-100k --data_dir data --output_dir outputs \
    --config configs/local.yaml --top_k $K --seed 42
done
```

### Candidate budget protocol

Build the candidate pool at your largest budget (e.g., `top_k=200`) once. Smaller budgets (`top_k=100`, `50`) are nested subsets — this preserves the split while changing the number of distractors.

### Add retrieval to any main-pipeline run

```bash
python scripts/run_dataset.py ... --with_retrieval --retrieval_n 200
```

This writes retrieval results to `outputs/<dataset>-top200-s42/retrieval/` alongside the positive-controlled outputs.

### Quick smoke test (minimal GPU, ~2 minutes)

```bash
# Disable heavy models in configs/local.yaml first:
#   baselines: {run_sasrec: false, run_sbert: false}
#   rerank: {sample_users: 50, llm_model_name: "Qwen/Qwen3.5-0.8B"}

python scripts/run_dataset.py \
  --dataset ml-100k --top_k 20 --seed 1 \
  --config configs/local.yaml
```

---

## Citation

If you use this library in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20993306.svg)](https://doi.org/10.5281/zenodo.20993306)

## License

MIT
