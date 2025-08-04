from typing import List, Dict, Any, Union
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class MoverScoreEvaluator:
    """
    MoverScore评估器，用于评估文本生成质量
    专为RAG系统评估设计
    """
    def __init__(self, model_name='bert-base-uncased', device=None):
        """
        初始化MoverScore评估器

        Args:
            model_name (str): 要使用的预训练模型
            device (str): 运行设备，'cuda'或'cpu'，如果为None则自动选择
        """
        self.model_name = model_name
        
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"MoverScore评估器初始化完成，使用{model_name}，运行在{self.device}上")
    
    def get_embeddings(self, texts, batch_size=8):
        """
        获取文本的嵌入表示
        
        Args:
            texts (List[str]): 文本列表
            batch_size (int): 批处理大小
            
        Returns:
            np.ndarray: 文本嵌入表示
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 编码文本
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取模型输出
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # 使用最后一层的[CLS]标记表示作为句子嵌入
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        # 合并所有批次的嵌入
        if embeddings:
            embeddings = np.vstack(embeddings)
        else:
            embeddings = np.array([])
            
        return embeddings
    
    def calculate_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        计算参考文本和假设文本之间的MoverScore
        
        Args:
            reference (str): 参考文本
            hypothesis (str): 假设文本
            
        Returns:
            Dict[str, float]: 包含评分结果的字典
        """
        # 获取嵌入
        ref_embedding = self.get_embeddings([reference])
        hyp_embedding = self.get_embeddings([hypothesis])
        
        # 转为张量
        ref_tensor = torch.tensor(ref_embedding)
        hyp_tensor = torch.tensor(hyp_embedding)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(ref_tensor, hyp_tensor).item()
        
        # 返回结果
        result = {
            "moverscore": similarity,
            "quality": self._get_quality_label(similarity)
        }
        
        return result
    
    def evaluate_rag_response(self, response: str, reference: str) -> Dict[str, Any]:
        """
        评估RAG响应与参考答案的匹配程度
        
        Args:
            response (str): RAG系统的响应
            reference (str): 参考答案
            
        Returns:
            Dict[str, Any]: 包含评估结果的字典
        """
        score_result = self.calculate_score(reference, response)
        
        # 添加质量描述
        score = score_result["moverscore"]
        quality = score_result["quality"]
        
        if quality == "高":
            description = "✅ 回答质量很高，与参考答案语义相似度高"
        elif quality == "中":
            description = "🟡 回答质量不错，但与参考答案有一定差异"
        else:
            description = "❌ 回答质量较差，与参考答案差异较大"
        
        result = {
            "moverscore": score,
            "quality": quality,
            "description": description
        }
        
        return result
    
    def _get_quality_label(self, score: float) -> str:
        """
        根据分数确定质量标签
        
        Args:
            score (float): MoverScore分数
            
        Returns:
            str: 质量标签，"高"、"中"或"低"
        """
        if score > 0.85:
            return "高"
        elif score > 0.75:
            return "中"
        else:
            return "低"

# 示例用法
if __name__ == "__main__":
    # 初始化评估器
    evaluator = MoverScoreEvaluator()
    
    # 示例1：高相似度
    reference1 = "标准电池可提供长达 35 分钟的飞行时间。采用最新的 SkyLink 3.0 技术，最大图传距离为 15 公里。"
    response1 = "AeroBot V2 无人机的续航时间是 35 分钟，最大图传距离为 15 公里。"
    
    # 示例2：中等相似度
    reference2 = "AeroBot V2可以在5级风下稳定飞行，但产品不具备防水功能，因此在海边使用时需要注意避免接触海水和过于湿润的环境。"
    response2 = "不合适。AeroBot V2 只能在5级风下稳定飞行。"
    
    # 示例3：低相似度
    reference3 = "文档中没有提供AeroBot V2的价格信息。"
    response3 = "不知道"
    
    # 评估并打印结果
    print("\n示例1评估结果：")
    result1 = evaluator.evaluate_rag_response(response1, reference1)
    print(f"MoverScore: {result1['moverscore']:.4f}")
    print(f"质量评级: {result1['quality']}")
    print(result1['description'])
    
    print("\n示例2评估结果：")
    result2 = evaluator.evaluate_rag_response(response2, reference2)
    print(f"MoverScore: {result2['moverscore']:.4f}")
    print(f"质量评级: {result2['quality']}")
    print(result2['description'])
    
    print("\n示例3评估结果：")
    result3 = evaluator.evaluate_rag_response(response3, reference3)
    print(f"MoverScore: {result3['moverscore']:.4f}")
    print(f"质量评级: {result3['quality']}")
    print(result3['description'])
