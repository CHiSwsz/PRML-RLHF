import os
import re
import math
import random
import argparse
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig

# ✅ 你日志里已经在用 experimental 入口；按官方 deprecation 提示也建议这么导入
from trl.experimental.ppo import PPOConfig, PPOTrainer  # :contentReference[oaicite:1]{index=1}

EMOJI_RE = re.compile(
    "["                     # rough emoji range (good enough)
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_emoji(text: str) -> int:
    return len(EMOJI_RE.findall(text))

def emoji_score(text: str) -> float:
    c = count_emoji(text)
    return float(min(c, 4) - max(c - 4, 0) * 0.5)

# --- ✅ 做一个“规则奖励模型”，让新 PPOTrainer 能吃进去 ---
class EmojiRewardConfig(PretrainedConfig):
    model_type = "emoji_reward"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = 1

class EmojiRewardModel(PreTrainedModel):
    config_class = EmojiRewardConfig

    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(EmojiRewardConfig())
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # input_ids: [B, L]  (这里一般是 prompt+response 拼起来的序列)
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        scores = [emoji_score(t) for t in texts]
        logits = torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)  # [B,1]
        return SequenceClassifierOutput(logits=logits)

def main():
    parser = argparse.ArgumentParser()

    # data / output
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="ppo_emoji_out")

    # models
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

    # reward
    parser.add_argument("--reward_type", type=str, choices=["emoji_rule"], default="emoji_rule")

    # lengths / generation
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)

    # PPO knobs
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--total_episodes", type=int, default=2000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_ppo_epochs", type=int, default=1)
    parser.add_argument("--num_mini_batches", type=int, default=1)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--missing_eos_penalty", type=float, default=0.0)
    parser.add_argument("--whiten_rewards", action="store_true")

    # misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------
    # Dataset: 新 PPOTrainer 更喜欢“只留 input_ids”
    # --------------------
    ds = load_dataset("json", data_files=args.train_file, split="train")
    if "prompt" not in ds.column_names:
        raise ValueError(f"train_file must contain field 'prompt'. got columns={ds.column_names}")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, padding_side="left", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_row(ex):
        ex["input_ids"] = tok.encode(ex["prompt"], truncation=True, max_length=args.max_prompt_length)
        return ex

    ds = ds.map(tokenize_row, batched=False, desc="Tokenizing prompts")
    ds = ds.remove_columns([c for c in ds.column_names if c != "input_ids"])
    ds = ds.shuffle(seed=args.seed)

    # --------------------
    # Models (按官方示例：policy/ref 是 CausalLM；reward/value 是 SeqCls) :contentReference[oaicite:2]{index=2}
    # --------------------
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, torch_dtype=dtype, trust_remote_code=True)
    ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, torch_dtype=dtype, trust_remote_code=True)

    # ✅ 规则 reward model
    reward_model = EmojiRewardModel(tok)

    # ✅ value model：用同底座做一个 num_labels=1 的序列回归头（给 PPO 学 value）
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # --------------------
    # PPO config + trainer
    # --------------------
    ppo_args = PPOConfig(
        learning_rate=args.learning_rate,
        total_episodes=args.total_episodes,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=args.num_mini_batches,
        kl_coef=args.kl_coef,
        whiten_rewards=args.whiten_rewards,
        missing_eos_penalty=args.missing_eos_penalty,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,  # ✅ 让 trainer 内部 generate 用这个
        temperature=args.temperature,
        top_p=args.top_p,
    )

    trainer = PPOTrainer(
        args=ppo_args,                # ✅ 这里必须是 args=，不是 config=
        processing_class=tok,         # ✅ 新版用 processing_class
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,    # ✅ 必须提供
        value_model=value_model,      # ✅ 必须提供
        train_dataset=ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved to:", args.output_dir)

if __name__ == "__main__":
    main()
