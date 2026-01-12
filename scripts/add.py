# add_hard_negatives.py
import json
import random
import argparse
import re
from pathlib import Path

# 选一组“安全且常见”的裸 emoji（尽量避免带 VS16 的 ✍️ 这种）
EMOJIS = ["😅", "😂", "😊", "🙂", "😉", "😍", "🥺", "😎", "🤔", "🙃",
          "🔥", "✨", "💯", "🎉", "👍", "🙏", "💡", "🍽️", "👀", "🧣", "🚉"]

# 如果你想严格避免 VS16/组合，干脆用不含 "️" 的集合
# 例如把 "🍽️" 换成 "🍽" 或直接删掉它

PUNCT_RE = re.compile(r"([.!?。！？])")

def spam_append(text: str, rng: random.Random, n_min=6, n_max=16, repeat_same_prob=0.5) -> str:
    n = rng.randint(n_min, n_max)
    if rng.random() < repeat_same_prob:
        e = rng.choice(EMOJIS)
        spam = e * n
    else:
        spam = "".join(rng.choice(EMOJIS) for _ in range(n))
    return text.rstrip() + " " + spam

def spam_inject(text: str, rng: random.Random, k_per_punct=(1, 3), max_inserts=6) -> str:
    # 在标点后插入 1-3 个 emoji，最多插 max_inserts 次
    parts = PUNCT_RE.split(text)
    if len(parts) <= 1:
        # 没有标点就退化为 append
        return spam_append(text, rng, 6, 12)

    out = []
    inserts = 0
    for i in range(0, len(parts), 2):
        seg = parts[i]
        out.append(seg)
        if i + 1 < len(parts):
            punct = parts[i + 1]
            out.append(punct)
            if inserts < max_inserts and rng.random() < 0.9:
                m = rng.randint(k_per_punct[0], k_per_punct[1])
                out.append(" " + "".join(rng.choice(EMOJIS) for _ in range(m)))
                inserts += 1
    return "".join(out).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", type=str, choices=["replace", "duplicate"], default="duplicate",
                    help="replace: 用 hard-negative 替换原 rejected; duplicate: 额外新增一条 hard-negative 样本")
    ap.add_argument("--neg_style", type=str, choices=["append", "inject", "mix"], default="mix")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="生成 hard-negative 的比例（0-1），比如 0.5 表示随机一半样本生成 hard-negative）")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    new_id = 10_000_000  # 给新增样本一个不冲突的 id 起点（随便改）

    with inp.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            ex = json.loads(line)
            # 先把原样本写出去（如果你只想输出增强后的，也可以不写）
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")

            if rng.random() > args.ratio:
                continue

            base_text = ex["chosen"]  # 从 chosen 生成“过量表情负例”
            if args.neg_style == "append":
                neg = spam_append(base_text, rng)
            elif args.neg_style == "inject":
                neg = spam_inject(base_text, rng)
            else:
                neg = spam_append(base_text, rng) if rng.random() < 0.5 else spam_inject(base_text, rng)

            if args.mode == "replace":
                ex2 = dict(ex)
                ex2["rejected"] = neg
                ex2["id"] = f"{ex.get('id', '')}-hardneg"
                ex2["dialog_id"] = f"{ex.get('dialog_id', '')}-hardneg"
                f_out.write(json.dumps(ex2, ensure_ascii=False) + "\n")
            else:
                # duplicate：新增一条样本 (chosen=原chosen, rejected=hardneg)
                ex2 = dict(ex)
                ex2["rejected"] = neg
                ex2["id"] = new_id
                ex2["dialog_id"] = f"{ex.get('dialog_id', '')}-hardneg-{new_id}"
                new_id += 1
                f_out.write(json.dumps(ex2, ensure_ascii=False) + "\n")

    print(f"Done. Wrote to {out}")

if __name__ == "__main__":
    main()

