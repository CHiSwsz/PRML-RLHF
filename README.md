# Reward Design for Emoji Control in Small Language Models

This repo contains the code and evaluation scripts for our PRML course project:  
**"Reward Design Lessons for Teaching a Small Language Model to Use Emoji"**  
Authors: Zile Wang, Zhide Xie (Fudan University)

## 📌 Overview

We study how different reward designs affect the stylistic alignment of a small instruction-tuned language model, specifically encouraging or discouraging the use of emojis in dialogue responses.

We compare:
- **RLHF (with PPO)** using a learned reward model
- **DPO** (Direct Preference Optimization) as a reward-model-free baseline

Two reward signal variants are tested:
1. **Binary emoji presence** (emoji vs. no-emoji)
2. **Overuse-aware** preferences (normal usage vs. emoji spam)

## 🧪 Key Findings
- PPO with underspecified reward signals leads to severe reward hacking (emoji spam).
- DPO is more stable but still benefits from overuse-negative examples.
- Explicit negatives in preference data reshape the reward landscape and improve robustness.

## 🗂️ Repo Structure

