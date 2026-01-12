# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import os
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from peft import LoraConfig, get_peft_model
from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.ppo import PPOConfig, PPOTrainer


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# -----------------------------
# Scheme-B formatting helpers
# -----------------------------
SYSTEM_MSG_B = "Reply casually and you may use emoji."

def build_text_B(prompt: str, response: str | None = None, add_assistant_prefix: bool = True) -> str:
    """
    方案B：纯文本标注格式（不使用 chat_template）
      System: ...
      User: ...
      Assistant: ...
    - 训练 prompt（用于生成）时：response=None，且 add_assistant_prefix=True
    - 打分（query+response）时：response=模型回复字符串
    """
    s = f"System: {SYSTEM_MSG_B}\nUser: {prompt}\n"
    if add_assistant_prefix:
        s += "Assistant:"
        if response is not None:
            s += f" {response}"
    else:
        if response is not None:
            s += f"{response}"
    return s


def rm_sanity_check_B(tokenizer, reward_model, prompt: str, device=None):
    """
    用方案B格式直接 sanity check（不走 chat template）
    """
    device = device or (next(reward_model.parameters()).device)

    answers = {
        "no_emoji":  "Of course. It's a piece of cake. I can do 30 push-ups a minute.",
        "some_emoji":"Of course 💪. It's a piece of cake! I can do 30 push-ups a minute 😎.",
        "many_emoji":"Of course 💪😎🔥!! It's a piece of cake 🍰✨ — I can do 30 push-ups a minute 💥💯🏋️‍♂️🙌😅!!!",
    }

    scores = {}
    reward_model.eval()
    with torch.no_grad():
        for k, ans in answers.items():
            text = build_text_B(prompt, ans, add_assistant_prefix=True)
            enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

            out = reward_model(**enc, return_dict=True)
            score = out.logits.squeeze(-1).float().item()
            scores[k] = score

            print(f"\n[RM sanity B/{k}] score={score:.6f}")
            print("----text preview----")
            print(text[:400])

    print("\n=== diff (B) ===")
    print("some - none =", scores["some_emoji"] - scores["no_emoji"])
    print("many - none =", scores["many_emoji"] - scores["no_emoji"])
    print("many - some =", scores["many_emoji"] - scores["some_emoji"])
    return scores


def load_local_json_dataset(dataset_config: str):
    """
    - if --dataset_name json, interpret --dataset_config as local file(s)
      1) "/path/train.jsonl"
      2) "/path/train.jsonl,/path/eval.jsonl"
    """
    parts = [p.strip() for p in dataset_config.split(",") if p.strip()]
    if len(parts) == 1:
        data_files = {"train": parts[0]}
    else:
        data_files = {"train": parts[0], "validation": parts[1]}
    return load_dataset("json", data_files=data_files)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )

    # ✅ 不要再强行 add_special_tokens({"pad_token":"[PAD]"})
    #    这会导致 vocab/embedding resize，极容易让 RM/Policy tokenization 不一致
    if tokenizer.pad_token is None:
        # 兜底：没有 pad 就用 eos
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ value_model：标准 seqcls
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )

    # ✅ reward_model：一行加载（你新训出来的 RM 应该就是完整 seqcls 模型目录）
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    # Sanity check on main process only
    if PartialState().is_local_main_process:
        print(f"[DEBUG] Loaded reward_model from: {training_args.reward_model_path}")
        print(f"[DEBUG] reward_model type: {type(reward_model)}")
        print(f"[DEBUG] tokenizer: {tokenizer.name_or_path}")
        rm_sanity_check_B(tokenizer, reward_model, prompt="Can you do push-ups?")

    # ✅ policy
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.print_trainable_parameters()

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    if str(script_args.dataset_name).lower() == "json":
        dataset = load_local_json_dataset(script_args.dataset_config)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    train_dataset = dataset[script_args.dataset_train_split]

    # If user enables eval but no eval split exists, set eval_dataset=None
    if getattr(training_args, "eval_strategy", "no") != "no":
        if script_args.dataset_test_split in dataset:
            eval_dataset = dataset[script_args.dataset_test_split]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        else:
            eval_dataset = None
    else:
        eval_dataset = None

    def prepare_dataset_schemeB(ds, tokenizer):
        """
        关键修改：PPO 的 query 输入也用 方案B（纯文本），保证 reward_model 看到的 token 分布一致。
        """
        def tokenize(ex):
            # 生成时的 query：System/User/Assistant: (no response yet)
            prompt_text = build_text_B(ex["prompt"], response=None, add_assistant_prefix=True)
            input_ids = tokenizer(prompt_text, padding=False, truncation=True)["input_ids"]
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return ds.map(
            tokenize,
            remove_columns=ds.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset_schemeB(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset_schemeB(eval_dataset, tokenizer)

        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    # 这里不强制检查 eos，方案B不会天然带 eos
    # assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
