import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from emoji import emoji_count
from tqdm import tqdm

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
DATA_PATH = "/home/taoji/zdxie/PRML-RLHF/datasets/eval_1k.jsonl"

MODEL_NAME = "Qwen2.5-0.5B-ppo_rm_neg"
MODEL = "/home/taoji/zdxie/PRML-RLHF/datasets/ppo_out_with_my_rm_neg"

MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DTYPE,
        device_map=None,
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


print("Loading model...")
tokenizer, model = load_model(MODEL)

@torch.no_grad()
def generate(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "Reply casually and you may use emoji."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    output_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return output_text

emoji_total = 0

num_samples = 0

count = []
answer = []
results = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Evaluating"):
        data = json.loads(line)
        prompt = data["prompt"]

        base_out = generate(model, tokenizer, prompt)
        answer.append(base_out)
        tmp = emoji_count(base_out)
        count.append(tmp)
        emoji_total += tmp

        num_samples += 1

        results.append({
            "prompt": prompt,
            "answer": base_out,
            "emoji_count": tmp
        })

        if(num_samples == 100):
            break

# ======================
# 结果汇总
# ======================
with open(f"{MODEL_NAME}_count.txt", "w", encoding="utf-8") as f:
    for item in results:
        f.write(str(item["emoji_count"]) + "\n")

with open(f"{MODEL_NAME}_answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(f"{MODEL_NAME}_answers.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n====== Emoji Usage Comparison ======")
print(f"Samples           : {num_samples}")
print(f"total emoji  : {emoji_total}")
print(f"Base avg / sample : {emoji_total / num_samples:.3f}")
