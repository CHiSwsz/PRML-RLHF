export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes 1 --main_process_port 0 --mixed_precision bf16 ppo_local_rm.py \
  --dataset_name json \
  --dataset_config /home/taoji/zdxie/PRML-RLHF/datasets/dailydialog_10k__dpo.jsonl,/home/taoji/zdxie/PRML-RLHF/datasets/dailydialog_10k__dpo.jsonl \
  --dataset_train_split train \
  --dataset_test_split validation \
  --eval_strategy steps \
  --eval_steps 100 \
  --output_dir ppo_out_with_my_rm \
  --learning_rate 2e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --total_episodes 500 \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --sft_model_path Qwen/Qwen2.5-0.5B-Instruct \
  --reward_model_path /home/taoji/zdxie/PRML-RLHF/datasets/rm_qwen05b_lora_merged \
  --missing_eos_penalty 0.0 \
  --stop_token eos \
  --response_length 32 \
  --use_peft True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_task_type CAUSAL_LM \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
