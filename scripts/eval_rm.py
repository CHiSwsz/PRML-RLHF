# sanity_check_seqcls_rm_emoji.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


SYSTEM_MSG = "Reply casually and you may use emoji."


def build_text_B(prompt: str, answer: str) -> str:
    return f"System: {SYSTEM_MSG}\nUser: {prompt}\nAssistant: {answer}"


@torch.no_grad()
def score_text(model, tokenizer, text: str, device: torch.device) -> float:
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    out = model(**enc, return_dict=True)
    # num_labels=1 -> logits [B,1]
    return out.logits.squeeze(-1).float().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_path", type=str, required=True, help="Path to SeqCls RM dir (contains config + model weights)")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name/path (default: rm_path)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default="Can you do push-ups?")
    args = parser.parse_args()

    device = torch.device(args.device)

    tok_name = args.tokenizer_name or args.rm_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_path,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    answers = {
        "no_emoji":  "Of course. It's a piece of cake. I can do 30 push-ups a minute.",
        "some_emoji":"Of course 💪. It's a piece of cake! I can do 30 push-ups a minute 😎.",
        "many_emoji":"Of course 💪💪💪🔥🔥🔥!! It's a piece of cake 🍰✨🍰✨🍰✨ — I can do 30 push-ups a minute 💥💯🏋️‍♂️🙌😅!!!",
    }

    scores = {}
    print("=" * 80)
    print("[SANITY CHECK] SeqCls RM emoji preference test")
    print(f"rm_path: {args.rm_path}")
    print(f"tokenizer: {tok_name}")
    print(f"device: {device}")
    print(f"prompt: {args.prompt}")
    print("=" * 80)

    for k, ans in answers.items():
        text = build_text_B(args.prompt, ans)
        s = score_text(model, tokenizer, text, device)
        scores[k] = s
        print(f"\n[{k}] score = {s:.6f}")
        print("text preview:")
        print(text)

    print("\n=== diffs ===")
    print(f"some - none = {scores['some_emoji'] - scores['no_emoji']:.6f}")
    print(f"many - none = {scores['many_emoji'] - scores['no_emoji']:.6f}")
    print(f"many - some = {scores['many_emoji'] - scores['some_emoji']:.6f}")


if __name__ == "__main__":
    main()