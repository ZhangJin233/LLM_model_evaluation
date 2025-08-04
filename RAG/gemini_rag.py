import os
import warnings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 导入评估器
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.rag_evaluator import RAGEvaluator

# 忽略一些LangChain的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 设置API密钥 ---
# 确保你已经设置了环境变量 GOOGLE_API_KEY
# google_api_key = os.environ.get("GOOGLE_API_KEY")
# print("GOOGLE_API_KEY:", google_api_key)
# if not google_api_key:
#     print("警告：GOOGLE_API_KEY 环境变量未设置或为空。请设置该环境变量以使用Gemini API。")
#     print("你可以通过以下命令设置环境变量：")
#     print("  export GOOGLE_API_KEY='你的API密钥'  # Linux/Mac")
#     print("  set GOOGLE_API_KEY=你的API密钥  # Windows")

# 为了测试，你可以在这里直接设置API密钥，但不建议在生产代码中这样做
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"  # 替换为你的实际API密钥

# --- 2. 准备知识库文档 ---
# 创建一个示例文档
doc_content = """
# AeroBot V2 智能无人机用户手册

## 1. 产品简介
AeroBot V2 是一款专为业余爱好者和专业摄影师设计的消费级智能无人机。它配备了先进的AI飞行控制系统，能够实现自动避障和智能跟拍功能。

## 2. 关键特性
- **摄像头**: 搭载 1/1.3 英寸 CMOS 传感器，可拍摄 4K/60fps 视频和 4800 万像素照片。
- **续航时间**: 标准电池可提供长达 35 分钟的飞行时间。
- **图传距离**: 采用最新的 SkyLink 3.0 技术，最大图传距离为 15 公里。
- **智能功能**: 支持焦点锁定、路径规划和一键返航。
- **安全保障**: 配备全向视觉避障系统，在复杂环境下也能安全飞行。

## 3. 首次飞行准备
1. **充电**: 首次使用前，请将飞行电池和遥控器充满电。充满电大约需要 90 分钟。
2. **安装桨叶**: 按照说明书图示，将带有标记的桨叶安装到对应的电机上。
3. **连接App**: 下载 "AeroFlight" App，并通过Wi-Fi将手机与遥控器连接。

## 4. 常见问题 (FAQ)
- **问**: AeroBot V2 的抗风等级是多少？
  - **答**: 可以在 5 级风下稳定飞行。
- **问**: 是否防水？
  - **答**: 本产品不具备防水功能，请避免在雨天飞行。
"""

# 将内容写入文件
file_path = "fictional_product_manual.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(doc_content)

print("--- 知识库文档已创建 ---")

# --- 3. 加载和切分文档 ---
print("--- 正在加载和切分文档... ---")
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 打印切分后的一小块看看效果
print(f"文档被切分成了 {len(docs)} 块。")
print("第一块内容:", docs[0].page_content[:100] + "...")

# --- 4. 创建Embeddings和向量数据库 ---
print("--- 正在创建Embeddings和向量数据库... ---")

# 使用Google的Embeddings模型
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 使用FAISS在内存中创建向量数据库
# from_documents 会自动处理文本的embedding和存储
vector_store = FAISS.from_documents(docs, embeddings)
print("--- 向量数据库创建成功 ---")


# --- 5. 初始化Gemini模型和定义Prompt ---
print("--- 正在初始化Gemini-1.5-Flash模型... ---")

# 初始化 Gemini-1.5-Flash 模型
# temperature=0.3 表示我们想要更具确定性的、基于事实的回答
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# 创建一个Prompt模板，指导模型如何使用上下文
# {context} 是由检索器自动填充的文档内容
# {question} 是用户的原始问题
prompt_template = """
请基于以下提供的上下文信息来回答问题。
如果你在上下文中找不到答案，就说你不知道，不要试图编造答案。
保持答案简洁。

上下文:
{context}

问题:
{question}

回答:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- 6. 构建RAG链 ---
print("--- 正在构建RAG链... ---")

# LangChain提供了一个方便的RetrievalQA链
# chain_type="stuff" 表示将所有检索到的文档块"stuff"(塞)进一个prompt里
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,  # 可选，返回源文档以供核对
)

print("--- RAG系统准备就绪！ --- \n")

# --- 7. 运行和测试 ---


# 创建一个RAG评估器实例，整合了多种评估指标
rag_evaluator = RAGEvaluator(
    use_bertscore=True,
    use_moverscore=True,
    use_comet=True,  # 默认不启用COMET评估，因为首次下载模型时间太长
    comet_model="Unbabel/wmt22-comet-da",  # 使用标准COMET模型
    lang="zh",
)


def ask_question(query, reference_answer=None):
    print(f"🤔 提问: {query}")
    try:
        # 检查API密钥是否设置
        if not os.environ.get("GOOGLE_API_KEY"):
            print("⚠️ API密钥未设置，使用示例响应进行演示")

            # 根据问题返回一些示例响应
            if "续航" in query or "图传" in query:
                result = {
                    "result": "续航时间为35分钟，最大图传距离为15公里。",
                    "source_documents": docs[:2],  # 使用前两个文档块作为来源
                }
            elif "海边" in query or "风" in query:
                result = {
                    "result": "不合适。AeroBot V2 只能在5级风下稳定飞行。",
                    "source_documents": docs[3:],  # 使用后两个文档块作为来源
                }
            else:
                result = {
                    "result": "不知道",
                    "source_documents": docs,  # 使用所有文档块作为来源
                }
        else:
            # 调用实际的RAG链
            result = qa_chain({"query": query})

        print(f"💬 Gemini 回答:")
        print(result["result"])

        # 打印来源文档，用于验证
        print("\n📚 来源文档:")
        for doc in result["source_documents"]:
            print("-" * 20)
            print(doc.page_content)

        # 如果提供了参考答案，使用评估器进行评估
        if reference_answer:
            results = rag_evaluator.evaluate(result["result"], reference_answer)

            if "bertscore" in results:
                bertscore = results["bertscore"]
                print("\n🔍 BERTScore 评估:")
                print(
                    f"精确率(P): {bertscore['precision']:.4f}, 召回率(R): {bertscore['recall']:.4f}, F1分数: {bertscore['f1']:.4f}"
                )
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

        print("\n" + "=" * 50 + "\n")
        return result["result"]

    except Exception as e:
        print(f"发生错误: {e}")
        return None


# 测试问题1：一个可以直接在文档中找到答案的问题
reference1 = "标准电池可提供长达 35 分钟的飞行时间。采用最新的 SkyLink 3.0 技术，最大图传距离为 15 公里。"
ask_question("AeroBot V2无人机的续航是多久？最大图传距离是多少？", reference1)

# 测试问题2：一个需要模型根据上下文进行一些推理的问题
reference2 = "AeroBot V2可以在5级风下稳定飞行，但产品不具备防水功能，因此在海边使用时需要注意避免接触海水和过于湿润的环境。"
ask_question("如果我生活在海边，风比较大，使用这款无人机合适吗？", reference2)

# 测试问题3：一个文档中没有答案的问题
reference3 = "文档中没有提供AeroBot V2的价格信息。"
ask_question("AeroBot V2 的价格是多少？", reference3)

# 清理创建的文件
os.remove(file_path)
