import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 如果可用，导入BERTScore评估器
try:
    from metrics.bert_score_evaluator import BERTScoreEvaluator
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("BERTScore不可用，请安装bert-score: pip install bert-score")

# 导入MoverScore评估器
try:
    from metrics.mover_score_evaluator import MoverScoreEvaluator
    MOVERSCORE_AVAILABLE = True
except ImportError:
    MOVERSCORE_AVAILABLE = False
    print("MoverScore不可用")

# 导入COMET评估器
try:
    from metrics.comet_evaluator import COMETEvaluator
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("COMET不可用，请安装unbabel-comet: pip install unbabel-comet")

class RAGEvaluator:
    """
    RAG系统评估器，结合多种语义相似度指标
    """
    def __init__(self, use_bertscore=True, use_moverscore=True, use_comet=False, lang="zh", 
                 bert_model="bert-base-chinese", mover_model="bert-base-uncased",
                 comet_model="Unbabel/wmt22-comet-da"):
        """
        初始化RAG评估器
        
        Args:
            use_bertscore (bool): 是否使用BERTScore
            use_moverscore (bool): 是否使用MoverScore
            use_comet (bool): 是否使用COMET
            lang (str): 语言代码
            bert_model (str): BERTScore使用的模型
            mover_model (str): MoverScore使用的模型
            comet_model (str): COMET使用的模型
                - "Unbabel/wmt22-comet-da": 标准评估模型，需要参考文本
                - "Unbabel/wmt22-cometkiwi-da": 无参考文本评估模型
                - "Unbabel/XCOMET-XL": 可解释的评估模型，能识别翻译错误
        """
        self.metrics = {}
        self.lang = lang
        
        # 初始化BERTScore
        if use_bertscore and BERTSCORE_AVAILABLE:
            self.bert_scorer = BERTScoreEvaluator(
                lang=lang, 
                model_type=bert_model, 
                rescale_with_baseline=False
            )
            self.metrics["bertscore"] = True
        else:
            self.metrics["bertscore"] = False
        
        # 初始化MoverScore
        if use_moverscore and MOVERSCORE_AVAILABLE:
            self.mover_scorer = MoverScoreEvaluator(model_name=mover_model)
            self.metrics["moverscore"] = True
        else:
            self.metrics["moverscore"] = False
        
        # 初始化COMET
        if use_comet and COMET_AVAILABLE:
            self.comet_scorer = COMETEvaluator(model_name=comet_model)
            self.metrics["comet"] = True
        else:
            self.metrics["comet"] = False
    
    def evaluate(self, response, reference, source=None):
        """
        评估RAG响应与参考答案的匹配程度
        
        Args:
            response (str): RAG系统的响应
            reference (str): 参考答案
            source (str, optional): 源文本，用于无参考COMET评估
            
        Returns:
            dict: 评估结果
        """
        results = {}
        
        # 使用BERTScore评估
        if self.metrics["bertscore"]:
            bertscore_result = self.bert_scorer.evaluate_rag_response(response, reference)
            results["bertscore"] = bertscore_result
        
        # 使用MoverScore评估
        if self.metrics["moverscore"]:
            moverscore_result = self.mover_scorer.evaluate_rag_response(response, reference)
            results["moverscore"] = moverscore_result
        
        # 使用COMET评估
        if self.metrics["comet"]:
            comet_result = self.comet_scorer.evaluate_rag_response(response, reference, source)
            results["comet"] = comet_result
        
        return results
    
    def print_evaluation(self, response, reference, source=None):
        """
        评估并打印结果
        
        Args:
            response (str): RAG系统的响应
            reference (str): 参考答案
            source (str, optional): 源文本，用于无参考COMET评估
        """
        results = self.evaluate(response, reference, source)
        
        print("\n=== RAG评估结果 ===")
        print(f"回答: {response}")
        print(f"参考: {reference}")
        
        if "bertscore" in results:
            bertscore = results["bertscore"]
            print("\n🔍 BERTScore 评估:")
            print(f"精确率(P): {bertscore['precision']:.4f}, 召回率(R): {bertscore['recall']:.4f}, F1分数: {bertscore['f1']:.4f}")
            print(bertscore["description"])
        
        if "moverscore" in results:
            moverscore = results["moverscore"]
            print("\n🔄 MoverScore 评估:")
            print(f"分数: {moverscore['moverscore']:.4f}")
            print(moverscore["description"])
        
        if "comet" in results:
            comet = results["comet"]
            print("\n🌟 COMET 评估:")
            print(f"分数: {comet['comet_score']:.4f}")
            print(comet["description"])
            
            # 如果有错误标记，打印它们
            if "error_spans" in comet and comet["error_spans"]:
                print("\n错误标记:")
                for error in comet["error_spans"]:
                    severity = error["severity"]
                    text = error["text"]
                    confidence = error.get("confidence", 0)
                    print(f"- {severity} 错误 (置信度: {confidence:.2f}): '{text}'")
                    

if __name__ == "__main__":
    # 测试RAG评估器
    print("\n=== 测试基本评估 (BERTScore + MoverScore) ===")
    evaluator = RAGEvaluator(use_bertscore=True, use_moverscore=True)
    
    # 示例1：高相似度
    reference1 = "标准电池可提供长达 35 分钟的飞行时间。采用最新的 SkyLink 3.0 技术，最大图传距离为 15 公里。"
    response1 = "AeroBot V2 无人机的续航时间是 35 分钟，最大图传距离为 15 公里。"
    evaluator.print_evaluation(response1, reference1)
    
    # 示例2：中等相似度
    reference2 = "AeroBot V2可以在5级风下稳定飞行，但产品不具备防水功能，因此在海边使用时需要注意避免接触海水和过于湿润的环境。"
    response2 = "不合适。AeroBot V2 只能在5级风下稳定飞行。"
    evaluator.print_evaluation(response2, reference2)
    
    # 示例3：低相似度
    reference3 = "文档中没有提供AeroBot V2的价格信息。"
    response3 = "不知道"
    evaluator.print_evaluation(response3, reference3)
    
    # 测试包含COMET的评估
    try:
        if COMET_AVAILABLE:
            print("\n=== 测试完整评估 (BERTScore + MoverScore + COMET) ===")
            comet_evaluator = RAGEvaluator(
                use_bertscore=True, 
                use_moverscore=True, 
                use_comet=True,
                comet_model="Unbabel/wmt22-comet-da"
            )
            
            print("\n测试示例1 (高相似度):")
            comet_evaluator.print_evaluation(response1, reference1)
            
            # 尝试使用XCOMET模型进行可解释评估
            try:
                print("\n=== 测试XCOMET (可解释COMET) ===")
                xcomet_evaluator = RAGEvaluator(
                    use_bertscore=False, 
                    use_moverscore=False, 
                    use_comet=True,
                    comet_model="Unbabel/XCOMET-XL"
                )
                
                # 英文示例，便于XCOMET识别错误
                en_reference = "The standard battery provides up to 35 minutes of flight time. It uses the latest SkyLink 3.0 technology with a maximum transmission distance of 15 kilometers."
                en_response = "The AeroBot V2 drone can fly for 35 minutes and has a transmission range of 14 km."
                
                print("\n测试英文示例 (故意引入错误):")
                xcomet_evaluator.print_evaluation(en_response, en_reference)
            except Exception as e:
                print(f"XCOMET测试失败: {e}")
    except Exception as e:
        print(f"COMET测试失败: {e}")
