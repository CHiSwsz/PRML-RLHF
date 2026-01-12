# train_reward_model_pairwise_lora_export_merged.py
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    set_seed,
)
from peft import LoraConfig, get_peft_model


# -----------------------------
# 1) Data collator for pairwise samples
# -----------------------------
@dataclass
class PairwiseBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


class PairwiseDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen = [{"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]} for f in features]
        rejected = [{"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]} for f in features]

        chosen_batch = self.tokenizer.pad(chosen, padding=True, return_tensors="pt")
        rejected_batch = self.tokenizer.pad(rejected, padding=True, return_tensors="pt")

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


# -----------------------------
# 2) Custom Trainer with pairwise loss (SeqCls logits)
# -----------------------------
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # model is AutoModelForSequenceClassification(num_labels=1)
        out_c = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            return_dict=True,
        )
        out_r = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            return_dict=True,
        )

        # logits shape: [B, 1]  (num_labels=1)
        r_chosen = out_c.logits.squeeze(-1)
        r_rejected = out_r.logits.squeeze(-1)

        diff = r_chosen - r_rejected
        loss = -F.logsigmoid(diff).mean()

        if return_outputs:
            return loss, {"r_chosen": r_chosen.detach(), "r_rejected": r_rejected.detach()}
        return loss


# -----------------------------
# 3) Text format (keep stable!)
# -----------------------------
def build_text(prompt: str, response: str) -> str:
    # Keep EXACTLY the same format at train+inference to avoid distribution shift
    return f"User: {prompt}\nAssistant: {response}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--eval_file", type=str, default=None, help="Optional eval.jsonl")

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./rm_lora_out")

    # Export dirs
    parser.add_argument("--export_merged_dir", type=str, default=None,
                        help="If set, export a merged pure HF SeqCls model here. "
                             "Default: output_dir + '_merged'")

    parser.add_argument("--max_length", type=int, default=160)

    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # precision flags
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    export_merged_dir = args.export_merged_dir or (args.output_dir.rstrip("/")+ "_merged")
    os.makedirs(export_merged_dir, exist_ok=True)

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Base model: SeqCls(num_labels=1)
    # -----------------------------
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=torch_dtype,
    )
    rm.config.pad_token_id = tokenizer.pad_token_id  # avoid some edge warnings

    # -----------------------------
    # LoRA (IMPORTANT: save "score" head!)
    # -----------------------------
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",  # ✅ correct for sequence classification
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["score"],  # ✅关键：把 reward head 一起存进 adapter
    )
    rm = get_peft_model(rm, lora_cfg)

    # -----------------------------
    # Dataset
    # -----------------------------
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["eval"] = args.eval_file
    ds = load_dataset("json", data_files=data_files)

    def tokenize_pair(ex):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        text_c = build_text(prompt, chosen)
        text_r = build_text(prompt, rejected)

        tok_c = tokenizer(text_c, truncation=True, max_length=args.max_length, padding=False)
        tok_r = tokenizer(text_r, truncation=True, max_length=args.max_length, padding=False)

        return {
            "chosen_input_ids": tok_c["input_ids"],
            "chosen_attention_mask": tok_c["attention_mask"],
            "rejected_input_ids": tok_r["input_ids"],
            "rejected_attention_mask": tok_r["attention_mask"],
        }

    remove_cols = ds["train"].column_names
    ds_tok = ds.map(tokenize_pair, remove_columns=remove_cols)

    train_ds = ds_tok["train"]
    eval_ds = ds_tok["eval"] if "eval" in ds_tok else None

    collator = PairwiseDataCollator(tokenizer)

    # -----------------------------
    # Training args
    # -----------------------------
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["none"],
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    trainer = RewardTrainer(
        model=rm,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # -----------------------------
    # 1) Save adapter (optional, for inspection / future)
    # -----------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # -----------------------------
    # 2) Export merged pure HF model (THIS is what PPO will load)
    # -----------------------------
    # NOTE: need to access the underlying model on main process
    merged = trainer.model.merge_and_unload()  # -> AutoModelForSequenceClassification
    merged.eval()

    merged.save_pretrained(export_merged_dir)
    tokenizer.save_pretrained(export_merged_dir)

    print(f"\n[OK] Exported merged reward model to: {export_merged_dir}")
    print("PPO can load with ONE line:")
    print(f'  reward_model = AutoModelForSequenceClassification.from_pretrained("{export_merged_dir}", trust_remote_code=True)\n')

    # -----------------------------
    # Optional sanity check on a single example (same format)
    # -----------------------------
    if args.eval_file is not None:
        ex0 = load_dataset("json", data_files={"eval": args.eval_file})["eval"][0]
        merged_device = "cuda" if torch.cuda.is_available() else "cpu"
        merged = merged.to(merged_device)

        with torch.no_grad():
            tc = tokenizer(build_text(ex0["prompt"], ex0["chosen"]), return_tensors="pt",
                           truncation=True, max_length=args.max_length).to(merged_device)
            tr = tokenizer(build_text(ex0["prompt"], ex0["rejected"]), return_tensors="pt",
                           truncation=True, max_length=args.max_length).to(merged_device)
            rc = merged(**tc, return_dict=True).logits.squeeze(-1).item()
            rr = merged(**tr, return_dict=True).logits.squeeze(-1).item()
        print(f"[SANITY] r(chosen)={rc:.4f}, r(rejected)={rr:.4f}, diff={rc-rr:.4f}")


if __name__ == "__main__":
    main()
