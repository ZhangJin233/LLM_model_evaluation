from bert_score import score as bert_score_fn
from bert_score import BERTScorer
import torch

def calculate_bert_score(candidate, reference, lang="en", model_type=None, rescale_with_baseline=True):
    """
    计算一个候选文本与参考文本之间的BERTScore
    
    Args:
        candidate (str): 候选文本
        reference (str): 参考文本
        lang (str): 语言代码
        model_type (str): 使用的预训练模型类型
        rescale_with_baseline (bool): 是否使用基线分数进行缩放
        
    Returns:
        tuple: (precision, recall, f1) 元组
    """
    # 为中文选择合适的默认模型
    if model_type is None and lang == "zh":
        model_type = "bert-base-chinese"
    
    P, R, F1 = bert_score_fn(
        [candidate], 
        [reference], 
        lang=lang, 
        model_type=model_type,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False
    )
    return P.item(), R.item(), F1.item()

def print_bert_score_results(candidate, reference, lang="en", model_type=None, rescale_with_baseline=True):
    """
    计算并打印BERTScore结果
    
    Args:
        candidate (str): 候选文本
        reference (str): 参考文本
        lang (str): 语言代码
        model_type (str): 使用的预训练模型类型
        rescale_with_baseline (bool): 是否使用基线分数进行缩放
    
    Returns:
        float: F1分数
    """
    p, r, f1 = calculate_bert_score(candidate, reference, lang, model_type, rescale_with_baseline)
    print("=== BERTScore ===")
    print(f"候选文本: {candidate}")
    print(f"参考文本: {reference}")
    if rescale_with_baseline:
        print(f"精确率(缩放后): {p:.4f}, 召回率(缩放后): {r:.4f}, F1(缩放后): {f1:.4f}")
    else:
        print(f"精确率: {p:.4f}, 召回率: {r:.4f}, F1: {f1:.4f}")
    return f1

class BERTScoreEvaluator:
    """
    BERTScore评估器，适用于多次评估以节省模型加载时间
    """
    def __init__(self, lang="zh", model_type=None, rescale_with_baseline=True):
        """
        初始化BERTScore评估器
        
        Args:
            lang (str): 语言代码
            model_type (str): 使用的预训练模型类型
            rescale_with_baseline (bool): 是否使用基线分数进行缩放
        """
        # 为中文选择合适的默认模型
        if model_type is None and lang == "zh":
            model_type = "bert-base-chinese"
            
        self.scorer = BERTScorer(
            lang=lang, 
            model_type=model_type,
            rescale_with_baseline=rescale_with_baseline
        )
        self.lang = lang
        self.model_type = model_type
        self.rescale_with_baseline = rescale_with_baseline
    
    def score(self, candidates, references):
        """
        计算多个候选文本与参考文本之间的BERTScore
        
        Args:
            candidates (list): 候选文本列表
            references (list): 参考文本列表
            
        Returns:
            tuple: (precision, recall, f1) 元组，每个元素都是一个张量
        """
        if isinstance(candidates, str):
            candidates = [candidates]
        if isinstance(references, str):
            references = [references]
        
        return self.scorer.score(candidates, references)
    
    def evaluate_rag_response(self, response, reference):
        """
        评估RAG系统的响应与参考答案的匹配程度
        
        Args:
            response (str): RAG系统的响应
            reference (str): 参考答案
            
        Returns:
            dict: 包含评估结果的字典
        """
        P, R, F1 = self.score([response], [reference])
        
        result = {
            "precision": P.item(),
            "recall": R.item(),
            "f1": F1.item()
        }
        
        # 添加质量评估
        if result["f1"] > 0.8:
            result["quality"] = "高"
            result["description"] = "✅ 回答质量很高，与参考答案语义相似度高"
        elif result["f1"] > 0.6:
            result["quality"] = "中"
            result["description"] = "🟡 回答质量不错，但与参考答案有一定差异"
        else:
            result["quality"] = "低"
            result["description"] = "❌ 回答质量较差，与参考答案差异较大"
            
        return result

if __name__ == "__main__":
    # 简单的示例
    candidate = "AeroBot V2 无人机的续航时间是 35 分钟，最大图传距离为 15 公里。"
    reference = "标准电池可提供长达 35 分钟的飞行时间。采用最新的 SkyLink 3.0 技术，最大图传距离为 15 公里。"
    
    # 使用单次评估函数
    print("=== 单次评估 ===")
    print_bert_score_results(candidate, reference, "zh", rescale_with_baseline=True)
    
    # 使用评估器进行多次评估（更高效）
    print("\n=== 使用评估器 ===")
    evaluator = BERTScoreEvaluator(lang="zh")
    result = evaluator.evaluate_rag_response(candidate, reference)
    print(f"精确率: {result['precision']:.4f}, 召回率: {result['recall']:.4f}, F1: {result['f1']:.4f}")
    print(result["description"])
    
    # 另一个示例
    candidate2 = "这款无人机可以在5级风下稳定飞行，但是不防水，不建议在海边使用。"
    reference2 = "AeroBot V2可以在5级风下稳定飞行，但产品不具备防水功能，请避免在雨天飞行。"
    
    result2 = evaluator.evaluate_rag_response(candidate2, reference2)
    print("\n候选文本:", candidate2)
    print("参考文本:", reference2)
    print(f"精确率: {result2['precision']:.4f}, 召回率: {result2['recall']:.4f}, F1: {result2['f1']:.4f}")
    print(result2["description"])
