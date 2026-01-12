python train.py \
  --train_file dailydialog_10k__dpo.jsonl \
  --output_dir rmcl_qwen05b_lora \
  --max_length 160 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --bf16