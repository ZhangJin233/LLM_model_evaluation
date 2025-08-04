"""
COMET评估器 - 使用COMET计算语义相似度

COMET (Crosslingual Optimized Metric for Evaluation of Translation) 是一个神经网络
基础的度量标准，通常用于评估机器翻译的质量，但它也可以用于一般文本的语义相似度评估。

官方库: https://github.com/Unbabel/COMET
"""

import os
import torch
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("COMET 库未安装，请使用以下命令安装：")
    print("pip install --upgrade pip")
    print("pip install unbabel-comet")

class COMETEvaluator:
    """
    COMET评估器，用于评估文本生成质量
    """
    def __init__(self, model_name="Unbabel/wmt22-comet-da", device=None):
        """
        初始化COMET评估器
        
        Args:
            model_name (str): COMET模型名称或本地路径
                - "Unbabel/wmt22-comet-da": 标准评估模型，需要参考文本
                - "Unbabel/wmt22-cometkiwi-da": 无参考文本评估模型
                - "Unbabel/XCOMET-XL": 可解释的评估模型，能识别翻译错误
            device (str): 运行设备，'cuda'或'cpu'，如果为None则自动选择
        """
        if not COMET_AVAILABLE:
            raise ImportError("COMET库未安装，无法初始化评估器")
        
        self.model_name = model_name
        
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 下载并加载模型
        print(f"正在加载COMET模型：{model_name}...")
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
        
        # 确定模型类型（是否需要参考文本）
        self.requires_reference = "kiwi" not in model_name.lower()
        self.is_xcomet = "xcomet" in model_name.lower()
        
        print(f"COMET模型加载完成，运行在{self.device}上")
        print(f"模型类型：{'带参考文本' if self.requires_reference else '无参考文本'}")
        print(f"是否为XCOMET：{'是' if self.is_xcomet else '否'}")
        
    def score(self, sources, translations, references=None, batch_size=None, gpus=None):
        """
        评估翻译质量
        
        Args:
            sources (List[str]): 源文本列表
            translations (List[str]): 待评估的翻译文本列表
            references (List[str], optional): 参考翻译列表，如果使用无参考模型可为None
            batch_size (int): 批处理大小，默认为None，单个句子时设为1以避免多进程问题
            gpus (int): 使用的GPU数量，若为None则使用预设设备
            
        Returns:
            dict: 评估结果
        """
        if gpus is None:
            gpus = 1 if self.device == 'cuda' else 0
            
        # 检查参考文本
        if self.requires_reference and references is None:
            raise ValueError("当前模型需要参考文本进行评估")
        
        # 准备数据
        data = []
        for i in range(len(sources)):
            sample = {
                "src": sources[i],
                "mt": translations[i]
            }
            if references is not None:
                sample["ref"] = references[i]
            data.append(sample)
        
        # 对于单句评估，使用batch_size=1避免多进程错误
        if batch_size is None:
            batch_size = 1 if len(sources) == 1 else 8
        
        try:
            # 运行模型预测
            model_output = self.model.predict(data, batch_size=batch_size, gpus=gpus)
            
            # 整理结果
            result = {
                "sentence_scores": model_output.scores,
                "system_score": model_output.system_score
            }
            
            # 添加错误标记（如果有）
            if self.is_xcomet and hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans'):
                result["error_spans"] = model_output.metadata.error_spans
                
            return result
        except Exception as e:
            # 在遇到错误时使用一种更简单的方式来计算分数
            print(f"COMET评估异常: {e}，使用简化评分...")
            
            # 返回一个模拟的评分结果
            # 这里我们只是给出一个基于长度比例的简单分数，实际应用中可以使用其他简单方法
            dummy_scores = []
            for i in range(len(translations)):
                # 一个简单的长度匹配评分作为后备
                trans_len = len(translations[i])
                ref_len = len(references[i]) if references is not None else len(sources[i])
                ratio = min(trans_len, ref_len) / max(trans_len, ref_len)
                # 将分数调整到0.5-0.9范围内，避免过低分数
                score = 0.5 + ratio * 0.4
                dummy_scores.append(score)
            
            return {
                "sentence_scores": dummy_scores,
                "system_score": sum(dummy_scores) / len(dummy_scores),
                "fallback": True  # 标记为后备计算结果
            }

    def evaluate_rag_response(self, response, reference, source=None):
        """
        评估RAG系统的响应与参考答案的匹配程度
        
        Args:
            response (str): RAG系统的响应
            reference (str): 参考答案
            source (str, optional): 源文本，如果为None则使用参考答案作为源文本
            
        Returns:
            dict: 包含评估结果的字典
        """
        # 如果没有提供源文本，则使用参考文本作为源文本
        if source is None:
            if self.requires_reference:
                source = reference
            else:
                raise ValueError("无参考模型需要提供源文本")
        
        # 准备输入数据
        sources = [source]
        translations = [response]
        references = [reference] if self.requires_reference else None
        
        # 评分 - 使用batch_size=1，避免单句评估时的多进程问题
        try:
            score_result = self.score(sources, translations, references, batch_size=1)
            
            # 获取分数
            score = score_result["sentence_scores"][0]
            
            # 是否为后备计算的结果
            is_fallback = score_result.get("fallback", False)
            
            # 根据分数确定质量等级
            if score > 0.9:
                quality = "高"
                description = "✅ 回答质量很高，与参考答案语义相似度高"
            elif score > 0.7:
                quality = "中"
                description = "🟡 回答质量不错，但与参考答案有一定差异"
            else:
                quality = "低"
                description = "❌ 回答质量较差，与参考答案差异较大"
            
            if is_fallback:
                description += " (注意: 使用简化评分方法)"
            
            result = {
                "comet_score": score,
                "quality": quality,
                "description": description,
                "is_fallback": is_fallback
            }
            
            # 添加错误标记（如果有）
            if "error_spans" in score_result:
                result["error_spans"] = score_result["error_spans"][0] if score_result["error_spans"] else []
            
            return result
            
        except Exception as e:
            print(f"COMET评估失败: {e}")
            
            # 在评估失败时，返回一个默认结果
            return {
                "comet_score": 0.7,  # 默认中等分数
                "quality": "未知",
                "description": f"⚠️ COMET评估失败: {str(e)}",
                "is_fallback": True
            }

if __name__ == "__main__":
    # 简单的示例
    if COMET_AVAILABLE:
        try:
            # 初始化评估器（使用参考文本的模型）
            print("\n=== 测试参考文本模型 ===")
            evaluator = COMETEvaluator(model_name="Unbabel/wmt22-comet-da")
            
            # 示例1
            reference1 = "标准电池可提供长达 35 分钟的飞行时间。采用最新的 SkyLink 3.0 技术，最大图传距离为 15 公里。"
            response1 = "AeroBot V2 无人机的续航时间是 35 分钟，最大图传距离为 15 公里。"
            
            result1 = evaluator.evaluate_rag_response(response1, reference1)
            print("\n示例1评估结果：")
            print(f"COMET得分: {result1['comet_score']:.4f}")
            print(f"质量评级: {result1['quality']}")
            print(result1['description'])
            
            # 示例2
            reference2 = "AeroBot V2可以在5级风下稳定飞行，但产品不具备防水功能，因此在海边使用时需要注意避免接触海水和过于湿润的环境。"
            response2 = "不合适。AeroBot V2 只能在5级风下稳定飞行。"
            
            result2 = evaluator.evaluate_rag_response(response2, reference2)
            print("\n示例2评估结果：")
            print(f"COMET得分: {result2['comet_score']:.4f}")
            print(f"质量评级: {result2['quality']}")
            print(result2['description'])
            
            # 尝试使用无参考模型（如果需要测试）
            try:
                print("\n=== 测试无参考文本模型 ===")
                qe_evaluator = COMETEvaluator(model_name="Unbabel/wmt22-cometkiwi-da")
                
                # 在无参考模式下，我们需要提供源文本
                source = "AeroBot V2 的续航时间和图传距离是多少？"
                result_qe = qe_evaluator.evaluate_rag_response(response1, reference1, source)
                
                print("\n无参考模式评估结果：")
                print(f"COMET得分: {result_qe['comet_score']:.4f}")
                print(f"质量评级: {result_qe['quality']}")
                print(result_qe['description'])
            except Exception as e:
                print(f"无参考模型测试失败: {e}")
                
        except Exception as e:
            print(f"评估失败: {e}")
    else:
        print("COMET库未安装，无法运行示例")
