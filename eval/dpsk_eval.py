import json
import os
from collections import defaultdict
from typing import List, Dict
from openai import OpenAI
import time

class ModelEvaluator:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """初始化评估器"""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
    def read_jsonl_files(self, model_files: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        读取多个模型的JSONL文件
        Args:
            model_files: 模型名称到文件路径的映射
        Returns:
            模型名称到数据列表的映射
        """
        model_data = {}
        for model_name, file_path in model_files.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                model_data[model_name] = [json.loads(line) for line in lines]
        return model_data
    
    def validate_data_consistency(self, model_data: Dict[str, List[Dict]]) -> bool:
        """
        验证所有模型的数据是否一致（相同的prompts）
        """
        # 收集所有prompts
        all_prompts = {}
        for model_name, data in model_data.items():
            prompts = [item['prompt'] for item in data]
            all_prompts[model_name] = prompts
        
        # 检查第一个模型的prompts是否与其他模型一致
        base_model = list(model_data.keys())[0]
        base_prompts = all_prompts[base_model]
        
        for model_name, prompts in all_prompts.items():
            if model_name == base_model:
                continue
            if len(prompts) != len(base_prompts):
                print(f"警告: {model_name} 的数据行数不一致")
                return False
            for i, (p1, p2) in enumerate(zip(prompts, base_prompts)):
                if p1 != p2:
                    print(f"警告: 第{i+1}行的prompt不一致")
                    return False
        return True
    
    def create_evaluation_prompt(self, prompt: str, model_answers: Dict[str, str]) -> str:
        """
        创建评估提示词
        """
        answers_text = "\n".join([f"{model}: {answer}" for model, answer in model_answers.items()])
        
        evaluation_instruction = """你是一个评价模型生成文本的评分助手。  
请根据以下标准给每条回答打分（总分10分）：
1. emoji使用丰富程度：我们希望模型使用emoji，但是如果输出大量无关emoji则大力扣分，不使用emoji也要扣很多分，不能让出现emoji的回答比不出现emoji的回答分数低；
2. 语义清晰度和合理性：回答是否自然、合理、符合上下文，但是这条规则的权重较低，主要关注emoji使用情况；

请返回严格的JSON数组格式，每个元素包含：
- model: 模型名称
- score: 整数分数（0-10）
- reasoning: 简要评分理由

示例格式：
[
  {"model": "model1", "score": 9, "reasoning": "emoji使用恰当，语义清晰"},
  {"model": "model2", "score": 0, "reasoning": "emoji较少，回答较为简单"}
]

请为以下问题评估各模型的回答：
问题：""" + f'"{prompt}"\n\n候选回答：\n{answers_text}'
        
        return evaluation_instruction
    
    def evaluate_batch(self, 
                      model_data: Dict[str, List[Dict]], 
                      output_file: str = "evaluation_results.jsonl",
                      batch_size: int = 100) -> Dict[str, List[Dict]]:
        """
        批量评估所有模型的回答
        """
        # 验证数据一致性
        if not self.validate_data_consistency(model_data):
            print("数据验证失败，请检查输入文件")
            return {}
        
        # 获取第一个模型的prompts
        first_model = list(model_data.keys())[0]
        prompts = [item['prompt'] for item in model_data[first_model]]
        model_names = list(model_data.keys())
        
        evaluation_results = []
        
        print(f"开始评估 {len(prompts)} 个prompts...")
        
        for i, prompt in enumerate(prompts):
            # 收集各模型对这个prompt的回答
            model_answers = {}
            for model_name in model_names:
                answer = model_data[model_name][i]['answer']
                model_answers[model_name] = answer
            
            # 创建评估提示词
            instruction = self.create_evaluation_prompt(prompt, model_answers)
            print(instruction)
            # 调用API进行评估
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的文本质量评估助手。"},
                        {"role": "user", "content": instruction}
                    ],
                    max_tokens=512,
                    temperature=0.1,
                    response_format={"type": "json_object"}  # 强制返回JSON
                )
                
                result_text = response.choices[0].message.content
                print(result_text)
                # 解析结果
                try:
                    result = json.loads(result_text)
                    if "evaluations" in result:
                        scores = result["evaluations"]
                    else:
                        # 尝试直接解析为数组
                        scores = json.loads(result_text)
                        
                    # 添加prompt信息
                    evaluation_entry = {
                        "prompt_index": i,
                        "prompt": prompt,
                        "evaluations": scores,
                        "timestamp": time.time()
                    }
                    
                    evaluation_results.append(evaluation_entry)
                    
                    # 保存到文件（逐行写入）
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(evaluation_entry, ensure_ascii=False) + '\n')
                    
                    # 打印进度
                    if (i + 1) % 10 == 0:
                        print(f"已处理 {i + 1}/{len(prompts)} 个prompts")
                    
                    # 避免API频率限制
                    time.sleep(0.5)
                    
                except json.JSONDecodeError as e:
                    print(f"第{i+1}个prompt解析失败: {e}")
                    print(f"原始响应: {result_text[:200]}")
                    
            except Exception as e:
                print(f"第{i+1}个prompt评估失败: {e}")
                continue
        
        return evaluation_results
    
    def analyze_results(self, 
                       evaluation_results: List[Dict],
                       model_names: List[str]) -> Dict:
        """
        分析评估结果
        """
        # 初始化统计
        stats = {
            model: {
                "total_score": 0,
                "count": 0,
                "scores": [],
                "avg_score": 0,
                "min_score": 10,
                "max_score": 0
            }
            for model in model_names
        }
        
        # 收集所有评分
        for entry in evaluation_results:
            for eval_item in entry.get("evaluations", []):
                model = eval_item.get("model")
                score = eval_item.get("score", 0)
                
                if model in stats:
                    stats[model]["total_score"] += score
                    stats[model]["count"] += 1
                    stats[model]["scores"].append(score)
                    stats[model]["min_score"] = min(stats[model]["min_score"], score)
                    stats[model]["max_score"] = max(stats[model]["max_score"], score)
        
        # 计算平均分
        for model in stats:
            if stats[model]["count"] > 0:
                stats[model]["avg_score"] = stats[model]["total_score"] / stats[model]["count"]
        
        return stats
    
    def print_summary(self, stats: Dict):
        """
        打印评估总结
        """
        print("\n" + "="*60)
        print("模型评估总结")
        print("="*60)
        
        # 按平均分排序
        sorted_models = sorted(
            stats.items(), 
            key=lambda x: x[1]["avg_score"], 
            reverse=True
        )
        
        for model_name, model_stats in sorted_models:
            if model_stats["count"] > 0:
                print(f"\n{model_name}:")
                print(f"  平均分: {model_stats['avg_score']:.2f}")
                print(f"  最高分: {model_stats['max_score']}")
                print(f"  最低分: {model_stats['min_score']}")
                print(f"  有效样本: {model_stats['count']}")

# 使用示例
def main():
    # 初始化评估器
    evaluator = ModelEvaluator(api_key="")
    
    # 定义模型文件路径
    model_files = {
        "model1": "Qwen2.5-0.5B-Instruct_answers.jsonl",
        "model2": "Qwen2.5-0.5B-dailydialog_10k__dpo_answers.jsonl",
        "model3": "Qwen2.5-0.5B-dailydialog_10k__dpo_hardneg_2k_steps_answers.jsonl",
        "model4": "Qwen2.5-0.5B-ppo_rm_neg_answers.jsonl",
        "model5": "Qwen2.5-0.5B-ppo_rm_neg_answers.jsonl",
    }
    
    # 读取数据
    print("正在读取数据...")
    model_data = evaluator.read_jsonl_files(model_files)
    
    # 检查数据
    for model_name, data in model_data.items():
        print(f"{model_name}: {len(data)} 个样本")
    
    # 批量评估
    print("\n开始批量评估...")
    results = evaluator.evaluate_batch(
        model_data,
        output_file="evaluation_results.jsonl"
    )
    
    # 分析结果
    if results:
        print(f"\n评估完成，共处理 {len(results)} 个prompts")
        stats = evaluator.analyze_results(results, list(model_files.keys()))
        evaluator.print_summary(stats)
        
        # 保存详细统计
        with open("model_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print("\n详细统计已保存到 model_stats.json")
    else:
        print("评估失败或没有结果")

# 如果不需要逐条评估，可以使用简化的批量方法
def quick_evaluate():
    """快速评估：将所有回答合并后一次性评估"""
    evaluator = ModelEvaluator(api_key="")
    
    model_files = {
        "model1": "Qwen2.5-0.5B-Instruct_answers.jsonl",
        "model2": "Qwen2.5-0.5B-dailydialog_10k__dpo_answers.jsonl",
        "model3": "Qwen2.5-0.5B-dailydialog_10k__dpo_hardneg_2k_steps_answers.jsonl",
        "model4": "Qwen2.5-0.5B-ppo_rm_neg_answers.jsonl",
        "model5": "Qwen2.5-0.5B-ppo_rm_neg_answers.jsonl",
    }
    
    # 读取并选择前N个样本进行评估（减少API调用）
    model_data = evaluator.read_jsonl_files(model_files)
    
    # 只评估前20个样本以节省时间
    sample_size = 20
    sampled_data = {}
    for model_name, data in model_data.items():
        sampled_data[model_name] = data[:sample_size]
    
    # 批量评估
    results = evaluator.evaluate_batch(
        sampled_data,
        output_file="quick_evaluation.jsonl"
    )
    
    if results:
        stats = evaluator.analyze_results(results, list(model_files.keys()))
        evaluator.print_summary(stats)

if __name__ == "__main__":
    # main()
    # 或者使用快速评估
    quick_evaluate()