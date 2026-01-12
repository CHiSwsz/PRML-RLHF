# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
export WANDB_API_KEY="a1f7c4c6a4c03538ee368586c56fcf49a49c99ee"
export WANDB_ENTITY="antnlp"
export WANDB_PROJECT="PRML-RLHF"
export WANDB_RUN_NAME="dailydialog_10k__dpo"

export HF_HUB_DISABLE_IPV6=1
export CURL_IPRESOLVE=4
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0


export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ETAG_TIMEOUT=300
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_HTTP_TIMEOUT=300

export HF_HOME=${HF_HOME:-/home/taoji/zdxie/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/taoji/zdxie/hf_cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/home/taoji/zdxie/hf_cache/datasets}
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$HF_DATASETS_CACHE"

wandb login

export CUDA_VISIBLE_DEVICES=3

python scripts/DPO.py \
    --dataset_name /home/taoji/zdxie/PRML-RLHF/datasets/dailydialog_10k__dpo.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8  \
    --max_steps 2000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2.5-0.5B-dailydialog_10k__dpo \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 16 \
    --report_to wandb