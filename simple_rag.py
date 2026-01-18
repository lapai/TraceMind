import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from tracemind.client.callback import TraceMindCallbackHandler

# 1. 加载环境变量
load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")
base_url = os.getenv("SILICONFLOW_BASE_URL")

# --- 关键校验：确保 Key 不是占位符 ---
if not api_key or "your_api_key" in api_key:
    print("\n❌ 错误: 检测到无效的 API Key！")
    exit(1)
else:
    print(f"✅ 已成功加载 API Key: {api_key[:8]}******")

# 2. 初始化模型
# 使用标准的 openai_api_key 参数名，这在 LangChain 中最兼容
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.2", 
    openai_api_key=api_key,
    openai_api_base=base_url,
    temperature=0.7
)

# Embedding 模型配置
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3", 
    openai_api_key=api_key,
    openai_api_base=base_url
)

# 3. 准备示例数据（实际应用中你可以使用 TextLoader 加载文件）
text_data = """
检索增强生成（Retrieval-Augmented Generation, RAG）是一种结合了检索和生成的自然语言处理技术。
它的工作流程主要包括：文档加载、文本切分、向量化、存储、检索和生成。
LangChain 是一个非常流行的构建 RAG 应用的框架。
硅基流动（SiliconFlow）提供高性能的 AI 模型推理服务，兼容 OpenAI 接口。
"""

# 4. 文档切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.create_documents([text_data])

# 5. 向量化并存储到内存中的 FAISS 数据库
print("正在构建向量数据库...")
vectorstore = FAISS.from_documents(
    documents=chunks, 
    embedding=embeddings
)

# 6. 构建检索链 (LCEL 语法)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

template = """你是一个专业的助手。请根据以下提供的上下文信息回答问题。
如果上下文中没有相关信息，请诚实回答你不知道，不要胡编乱造。

上下文:
{context}

问题:
{question}

回答:"""

prompt = ChatPromptTemplate.from_template(template)

# 定义 RAG 链
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 运行查询 (流式输出)
if __name__ == "__main__":
    query = "你知道我喜欢什么吗"
    print(f"\n用户提问: {query}")
    print("AI 回答: ", end="")
    
    # 初始化 TraceMind 监控
    tracemind_callback = TraceMindCallbackHandler(
        server_url="http://localhost:8000", 
        project_name="RAG-Tutorial"
    )
    # 添加 tags 示例：用于多维度分类监控
    config = {
        "callbacks": [tracemind_callback], 
        "run_name": "老派123",
        "tags": ['deepseek-ai/DeepSeek-V3.2']
    }
    result = rag_chain.invoke(query, config=config)
    print(result)
    print("\n")
    
    # 等待 1 秒确保异步追踪数据全部发送完成
    import time
    time.sleep(1)