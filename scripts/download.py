#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, random, time, zipfile
from pathlib import Path

import ijson
from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="dailydialog_first_ua.jsonl")
    p.add_argument("--n", type=int, default=5000, help="number of pairs to export")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hub_cache", type=str, default=os.environ.get("HF_HUB_CACHE", ""))
    p.add_argument("--max_user_chars", type=int, default=400)
    p.add_argument("--max_asst_chars", type=int, default=500)
    p.add_argument("--log_every", type=int, default=5000)
    return p.parse_args()


def locate_data_zip(repo_dir: str) -> Path:
    p = Path(repo_dir) / "data.zip"
    if p.exists():
        return p
    hits = list(Path(repo_dir).rglob("data.zip"))
    if hits:
        hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return hits[0]
    raise FileNotFoundError(f"data.zip not found under snapshot dir: {repo_dir}")


def trunc(s: str, n: int) -> str:
    s = (s or "").strip()
    if n > 0 and len(s) > n:
        s = s[:n].rstrip()
    return s


def main():
    args = parse_args()
    random.seed(args.seed)

    # force hf-mirror (optional but recommended)
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    repo_dir = snapshot_download(
        repo_id="ConvLab/dailydialog",
        repo_type="dataset",
        cache_dir=args.hub_cache if args.hub_cache else None,
        allow_patterns=["*"],
    )
    print("[INFO] snapshot_download dir:", repo_dir)

    zip_path = locate_data_zip(repo_dir)
    print("[INFO] using data.zip:", zip_path)

    inner = "data/dialogues.json"

    out = []
    scanned = 0
    accepted = 0
    t0 = time.time()

    with zipfile.ZipFile(zip_path, "r") as zf:
        if inner not in zf.namelist():
            raise FileNotFoundError(f"{inner} not found in zip. Entries: {zf.namelist()[:20]}")

        print("[INFO] streaming parse dialogues.json ...")
        with zf.open(inner, "r") as fp:
            # dialogues.json 是一个 JSON array，ijson.items(fp, "item") 逐条产出对象
            for ex in ijson.items(fp, "item"):
                scanned += 1
                turns = ex.get("turns")
                if not isinstance(turns, list) or len(turns) < 2:
                    continue

                t0_turn = turns[0] if isinstance(turns[0], dict) else None
                t1_turn = turns[1] if isinstance(turns[1], dict) else None
                if not t0_turn or not t1_turn:
                    continue

                spk0 = str(t0_turn.get("speaker", "")).strip().lower()
                spk1 = str(t1_turn.get("speaker", "")).strip().lower()
                if spk0 not in ("user", "human"):
                    continue
                if spk1 not in ("assistant", "system"):  # 有些数据可能用 assistant；system 基本不会但留兜底
                    if spk1 != "assistant":
                        continue

                user = trunc(str(t0_turn.get("utterance", "")), args.max_user_chars)
                asst = trunc(str(t1_turn.get("utterance", "")), args.max_asst_chars)
                if not user or not asst:
                    continue

                out.append({
                    "id": accepted,
                    "dialog_id": str(ex.get("dialogue_id", f"dlg-{scanned}")),
                    "user": user,
                    "assistant": asst,
                })
                accepted += 1

                if scanned % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"[INFO] scanned={scanned} accepted={accepted} elapsed={dt:.1f}s")

                if accepted >= args.n:
                    break

    with open(args.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(out)} pairs -> {args.out} (scanned={scanned})")


if __name__ == "__main__":
    main()
