#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import os
import random
import time
from typing import Dict, Any, Optional, List

import aiohttp


DEFAULT_BASE_URL = "https://api.deepseek.com" 
DEFAULT_MODEL = "deepseek-chat"   


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Your task is to rewrite the assistant reply to include emojis naturally."
)

def build_user_prompt(user: str, assistant: str) -> str:
    return (
        "Rewrite the assistant reply to include emojis naturally.\n\n"
        "Constraints:\n"
        "- Keep the meaning the same (do not add new facts).\n"
        "- Add relevant emojis naturally (2–6 emojis).\n"
        "- Keep length similar.\n"
        "- Do NOT change the user message.\n"
        "- Output ONLY the rewritten assistant reply text.\n\n"
        f"User message:\n<<<{user}>>>\n\n"
        f"Original assistant reply:\n<<<{assistant}>>>\n\n"
        "Rewritten assistant reply with emojis:"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=str, required=True, help="Input jsonl with {id, user, assistant, ...}")
    p.add_argument("--out_rewrite", type=str, default="emoji_rewrite.jsonl")
    p.add_argument("--out_dpo", type=str, default="dpo_pairs.jsonl")
    p.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--max_retries", type=int, default=6)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(int(obj["id"]))
            except Exception:
                continue
    return done


async def deepseek_chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    user: str,
    assistant: str,
    temperature: float,
    timeout_s: int,
) -> str:
    """
    Calls DeepSeek OpenAI-compatible chat completions endpoint.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(user, assistant)},
        ],
    }

    async with session.post(url, headers=headers, json=payload, timeout=timeout_s) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
        js = json.loads(text)
        content = js["choices"][0]["message"]["content"]
        return content.strip()


async def worker(
    name: str,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    args,
    api_key: str,
    fout_rewrite,
    fout_dpo,
    lock: asyncio.Lock,
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        ex = item
        ex_id = int(ex["id"])
        user = str(ex["user"])
        assistant = str(ex["assistant"])

        # retry with backoff + jitter
        last_err: Optional[str] = None
        for attempt in range(args.max_retries):
            try:
                emoji_reply = await deepseek_chat_completion(
                    session=session,
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    user=user,
                    assistant=assistant,
                    temperature=args.temperature,
                    timeout_s=args.timeout,
                )

                # write outputs atomically (under lock)
                out_rewrite = {
                    "id": ex_id,
                    "dialog_id": ex.get("dialog_id", ""),
                    "user": user,
                    "assistant": assistant,
                    "assistant_emoji": emoji_reply,
                }

                out_dpo = {
                    "prompt": user,
                    "chosen": emoji_reply,
                    "rejected": assistant,
                    "id": ex_id,
                    "dialog_id": ex.get("dialog_id", ""),
                }

                async with lock:
                    fout_rewrite.write(json.dumps(out_rewrite, ensure_ascii=False) + "\n")
                    fout_rewrite.flush()
                    fout_dpo.write(json.dumps(out_dpo, ensure_ascii=False) + "\n")
                    fout_dpo.flush()

                break  # success
            except Exception as e:
                last_err = repr(e)
                # exponential backoff with jitter
                wait = min(2 ** attempt, 30) + random.random()
                await asyncio.sleep(wait)

        if last_err is not None:
            # If all retries failed, record failure line (optional)
            async with lock:
                fout_rewrite.write(json.dumps({"id": ex_id, "error": last_err}, ensure_ascii=False) + "\n")
                fout_rewrite.flush()

        queue.task_done()


async def main_async(args):
    random.seed(args.seed)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment.")

    data = load_jsonl(args.infile)

    done_ids = load_done_ids(args.out_rewrite)  # resume by rewrite file
    todo = [ex for ex in data if int(ex["id"]) not in done_ids]
    print(f"[INFO] total={len(data)} done={len(done_ids)} todo={len(todo)}")

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=args.timeout + 10)

    q: asyncio.Queue = asyncio.Queue()
    for ex in todo:
        q.put_nowait(ex)

    lock = asyncio.Lock()

    with open(args.out_rewrite, "a", encoding="utf-8") as fout_rewrite, open(args.out_dpo, "a", encoding="utf-8") as fout_dpo:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            workers = [
                asyncio.create_task(worker(f"w{i}", q, session, args, api_key, fout_rewrite, fout_dpo, lock))
                for i in range(args.concurrency)
            ]

            # add stop signals
            for _ in workers:
                q.put_nowait(None)

            t0 = time.time()
            await q.join()
            for w in workers:
                await w
            dt = time.time() - t0
            print(f"[DONE] finished in {dt:.1f}s")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
