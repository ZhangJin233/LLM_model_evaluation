"""
语义相似度评估器对比演示

本脚本演示和比较了三种语义相似度评估方法：
- BERTScore: 基于BERT嵌入的评估方法
- MoverScore: 基于Earth Mover's Distance的评估方法
- COMET: 基于神经网络的翻译质量评估方法
"""

import os
import sys
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.rag_evaluator import RAGEvaluator

# 示例文本
EXAMPLES = [
    # 示例1：高相似度
    {
        "reference": "标准电池可提供长达 35 分钟的飞行时间。采用最新的 SkyLink 3.0 技术，最大图传距离为 15 公里。",
        "response": "AeroBot V2 无人机的续航时间是 35 分钟，最大图传距离为 15 公里。",
        "expected": "高"
    },
    # 示例2：中等相似度
    {
        "reference": "AeroBot V2可以在5级风下稳定飞行，但产品不具备防水功能，因此在海边使用时需要注意避免接触海水和过于湿润的环境。",
        "response": "不合适。AeroBot V2 只能在5级风下稳定飞行。",
        "expected": "中"
    },
    # 示例3：低相似度
    {
        "reference": "文档中没有提供AeroBot V2的价格信息。",
        "response": "不知道",
        "expected": "低"
    },
    # 示例4：英文示例
    {
        "reference": "The standard battery provides up to 35 minutes of flight time. It uses the latest SkyLink 3.0 technology with a maximum transmission distance of 15 kilometers.",
        "response": "The AeroBot V2 drone can fly for 35 minutes and has a transmission range of 14 km.",
        "expected": "高",
        "lang": "en"
    }
]

def run_evaluation(use_bertscore=True, use_moverscore=True, use_comet=False):
    """
    使用不同评估器组合运行评估
    """
    print("\n" + "=" * 50)
    print(f"评估配置: BERTScore={use_bertscore}, MoverScore={use_moverscore}, COMET={use_comet}")
    print("=" * 50)
    
    # 初始化评估器
    start_time = time.time()
    evaluator = RAGEvaluator(
        use_bertscore=use_bertscore,
        use_moverscore=use_moverscore,
        use_comet=use_comet,
        lang="zh",
        comet_model="Unbabel/wmt22-comet-da"
    )
    init_time = time.time() - start_time
    print(f"初始化用时: {init_time:.2f}秒\n")
    
    # 评估每个示例
    for i, example in enumerate(EXAMPLES):
        print(f"\n示例 {i+1}: (预期质量: {example['expected']})")
        
        # 使用适当的语言
        lang = example.get("lang", "zh")
        
        # 开始计时
        start_time = time.time()
        
        # 运行评估
        evaluator.print_evaluation(example["response"], example["reference"])
        
        # 计算用时
        eval_time = time.time() - start_time
        print(f"评估用时: {eval_time:.2f}秒")

if __name__ == "__main__":
    # 允许从命令行启用/禁用评估器
    import argparse
    parser = argparse.ArgumentParser(description='语义相似度评估器对比演示')
    parser.add_argument('--bert', action='store_true', help='使用BERTScore')
    parser.add_argument('--mover', action='store_true', help='使用MoverScore')
    parser.add_argument('--comet', action='store_true', help='使用COMET')
    parser.add_argument('--all', action='store_true', help='使用所有评估器')
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助信息
    if not (args.bert or args.mover or args.comet or args.all):
        print("请指定至少一个评估器。例如:")
        print("  python semantic_similarity_demo.py --bert    # 仅使用BERTScore")
        print("  python semantic_similarity_demo.py --bert --mover  # 使用BERTScore和MoverScore")
        print("  python semantic_similarity_demo.py --all     # 使用所有评估器")
        parser.print_help()
        sys.exit(1)
    
    if args.all:
        # 使用所有评估器
        print("\n===== 使用所有评估器 =====")
        run_evaluation(use_bertscore=True, use_moverscore=True, use_comet=True)
        
    else:
        # 使用指定的评估器
        use_bertscore = args.bert
        use_moverscore = args.mover
        use_comet = args.comet
        
        run_evaluation(
            use_bertscore=use_bertscore,
            use_moverscore=use_moverscore,
            use_comet=use_comet
        )
