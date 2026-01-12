# Reward Design Lessons for Teaching a Small Language Model to Use Emoji  
*A Comparative Study of RLHF (PPO) and DPO*

This repository contains code, data scripts, and evaluation utilities for our PRML course project on **reward design for stylistic control**: encouraging a small instruction-tuned language model to use emojis **appropriately** in dialogue.

**Authors:** Zile Wang, Zhide Xie (Fudan University)

## Overview

We study how different reward specifications affect alignment outcomes on a lightweight behavior control task: **emoji usage in responses**.

**Base policy:** `Qwen2.5-0.5B-Instruct`  
**Teacher for synthetic preferences:** DeepSeek API (used to generate preference pairs from DailyDialog prompts)

We compare two training pipelines:

- **RLHF (Reward Model + PPO)** using HuggingFace **TRL PPOTrainer**
- **DPO (Direct Preference Optimization)** using **TRL DPOTrainer**

## Preference Datasets

We construct two synthetic preference datasets:

- **Dataset A: Emoji vs. No-Emoji (presence-only)**
  - Winner contains emoji; loser contains no emoji.
- **Dataset B: Normal Emoji vs. Overuse (overuse-aware)**
  - Mix of:
    1) presence pairs (same as A)
    2) *hard negatives* where overuse/spam emojis are dispreferred  
  - Ratio (presence : overuse-negative) ≈ **5 : 1**

## Key Findings (Qualitative)

- **PPO + Dataset A** → strong **reward hacking**: emoji spam, irrelevant emojis, occasional garbling.
- **DPO + Dataset A** → mostly reasonable emoji usage, but still occasional overuse.
- **Adding overuse negatives (Dataset B)**:
  - PPO becomes **very conservative** (often near-zero emoji).
  - DPO yields **controlled, context-appropriate** emoji usage with no obvious spam.

## Evaluation

We evaluate on a fixed held-out set of **100 prompts** with consistent decoding settings, using:

- **Emoji rate**: fraction of responses with ≥1 emoji  
- **Emoji density**: # emoji characters (or tokens) per 100 generated tokens

Figures and sample generations are included under `figures/` (histograms) and the qualitative table image.
