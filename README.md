# TraceMind: 轻量级 RAG 监控与评估平台

TraceMind 是一个专为 RAG (检索增强生成) 应用设计的轻量级监控工具。它能够自动追踪 LangChain 工作流中的检索过程、模型调用和最终输出，并利用大模型进行自动化的三维评估（忠实度、相关性、精确度）。

---

## 🚀 核心功能

- **全链路追踪**：自动捕获 RAG 流程中的每个步骤（Chain, Retriever, LLM, Tool）。
- **自动化评估**：基于大模型自动计算 `Faithfulness` (忠实度), `Answer Relevancy` (回答相关性), `Context Precision` (上下文精确度)。
- **交互式看板**：
    - **树状视图**：清晰展示嵌套的调用关系。
    - **详情审计**：实时查看 Prompt、检索到的原文以及模型原始响应，
    - **统计导出**：一键导出 Excel 风格的统计报表，支持多维度筛选。
- **配置简单**：只需几行代码即可集成到现有的 LangChain 项目中。
- **数据管理**：支持自动清理过期数据和一键清空数据库。

---

## 🛠️ 快速开始

### 1. 启动服务端

#### **Windows 平台 (本地直接运行)**
我们提供了全自动脚本，会自动创建虚拟环境并安装依赖：
1.  进入 `tracemind` 目录。
2.  双击运行 `start_windows.bat`。
3.  脚本运行完成后，访问：`http://localhost:8000/dashboard`。
    *   *提示：如果 8000 端口被占用，系统会自动切换到 8001 或更高，请留意控制台输出的链接。*

#### **Linux / NAS / 服务器 (Docker 推荐)**
确保你已安装 Docker 和 Docker Compose：
```bash
cd tracemind
docker-compose up -d
```
启动后访问：`http://服务器IP:8000/dashboard`。

---

### 2. 集成到你的 RAG 应用

在你的 Python 代码中引入 `TraceMindCallbackHandler`：

```python
from tracemind.client.callback import TraceMindCallbackHandler

# 初始化追踪器
# server_url: 服务端地址 (例如 http://localhost:8000)
# project_name: 用于在面板中区分不同的项目
tracemind_callback = TraceMindCallbackHandler(
    server_url="http://localhost:8000", 
    project_name="My-RAG-App"
)

# 在调用 LangChain 时添加 callbacks
config = {"callbacks": [tracemind_callback]}
result = rag_chain.invoke("你的问题", config=config)
```

---

## ⚙️ 配置说明

### 评估模型配置
在 Dashboard 页面点击“统计”窗口 -> “评估高级配置”：
- **测试连接**：填好配置后，点击“测试模型连接”按钮，实时验证配置是否正确。
- **API Key**: 评估所用大模型的 API 密钥（兼容 OpenAI 格式）。
- **Base URL**: 接口地址。
- **Model**: 评估模型名称（建议使用 GPT-4o 或 DeepSeek-V3）。

### 端口与清理设置
- **端口自定义**：支持在界面修改服务运行端口，修改后重启生效。
- **自动避让**：如果设置的端口冲突，服务端会自动寻找下一个可用端口并保存。
- **数据自动清理**：可设置保留天数（默认 7 天），系统将自动删除过期 Trace 数据。

---

## 💡 NAS 部署说明 (高级)

如果你想将镜像打包部署到 NAS，请按照以下步骤：

1.  **本地打包镜像**：
    ```bash
    cd tracemind
    docker build -t tracemind:v1 ./server
    docker save -o tracemind.tar tracemind:v1
    ```
2.  **NAS 部署**：
    - 将 `tracemind.tar` 上传到 NAS 并通过 Docker 管理器导入映像。
    - **挂载卷**：将 NAS 的文件夹映射到容器的 `/app/data` (用于持久化数据库)。
    - **端口映射**：将容器的 `8000` 端口映射到你希望的 NAS 访问端口。

---

## 💡 路线图 (Roadmap)

1. [x] **一键测试连接**：实时反馈模型配置是否正确。
2. [x] **端口自定义**：支持界面配置服务端口。
3. [x] **自动避让端口**：解决端口占用导致的启动失败问题。