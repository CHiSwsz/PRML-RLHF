import matplotlib.pyplot as plt
import numpy as np
MODEL_NAME = "Qwen2.5-0.5B-dailydialog_10k__dpo_hardneg_2k_steps"
# 读取计数数据
with open(f"{MODEL_NAME}_count.txt", "r", encoding="utf-8") as f:
    emoji_counts = [int(line.strip()) for line in f if line.strip()]

# 统计各个分类的数量
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '≥10']
category_counts = [0] * len(categories)

for count in emoji_counts:
    if count <= 9:
        category_counts[count] += 1
    else:
        category_counts[-1] += 1  # ≥10

# 创建直方图
plt.figure(figsize=(12, 6))
bars = plt.bar(categories, category_counts, color='steelblue', edgecolor='black', alpha=0.8)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                int(height), ha='center', va='bottom', fontsize=10)

# 设置图表属性
plt.title(f'{MODEL_NAME} - Emoji using distribution (num_samples: {len(emoji_counts)})', fontsize=14, pad=20)
plt.xlabel('Emoji number in each response', fontsize=12)
plt.ylabel('number of response', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# 添加统计信息
emoji_total = sum(emoji_counts)
avg_emoji = emoji_total / len(emoji_counts) if emoji_counts else 0

stats_text = ''
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'Qwen2.5-0.5B-dpo_neg_emoji_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细统计
print(f"Emoji数量分布统计:")
for i, cat in enumerate(categories):
    count = category_counts[i]
    percentage = count / len(emoji_counts) * 100 if emoji_counts else 0
    print(f"  {cat}个Emoji: {count}个样本 ({percentage:.1f}%)")