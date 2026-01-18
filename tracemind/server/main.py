from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import sqlite3
import time
import re
from sqlalchemy import create_engine, Column, String, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import random
import requests

# Evaluation metrics
def extract_json_from_text(text: str) -> Optional[Dict]:
    """从文本中提取 JSON 内容"""
    print(f"[DEBUG] 输入文本: {text}")
    try:
        # 尝试直接解析
        return json.loads(text)
    except:
        # 尝试寻找 ```json ... ``` 或 { ... }
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
    return None

def call_llm_evaluator(model: str, question: str, answer: str, contexts: List[str], api_key: str = None, base_url: str = None, reasoning: str = None):
    """调用大模型进行 RAG 评估"""
    context_str = "\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    
    reasoning_section = f"\n[AI 原始推理过程]:\n{reasoning}\n" if reasoning else ""
    
    prompt = f"""你是一个 RAG 系统评估专家。请根据以下内容，对系统的回答进行深度审计。

[问题]: {question}
[检索到的上下文]: 
{context_str}
{reasoning_section}
[系统的回答]: {answer}

请按照以下步骤进行评估：
1. 分析回答是否忠实于上下文。请找出回答中的事实点，并逐一核对上下文。
2. 分析回答的相关性。回答是否直接解决了用户的问题？
3. 分析上下文的精确度。检索到的内容是否都是回答问题所必需的？

最后，请给出以下三个指标的分数（0.0 到 1.0 之间），并以 JSON 格式输出。
指标定义：
- faithfulness: 回答是否完全基于上下文，没有幻觉。
- answer_relevancy: 回答是否直接且有效地解决了问题。
- context_precision: 检索到的内容中，与回答问题相关的占比。

请先输出你的详细分析过程，最后再输出 JSON 评分，例如：
---
分析报告：...
---
{{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7}}
"""
    
    # 默认使用环境变量或传入的 key
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    final_base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    if not final_api_key:
        print("DEBUG: Error: No API Key provided for evaluation")
        return None

    try:
        headers = {
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json"
        }
        # 移除 response_format，因为不是所有模型都支持。依靠 prompt 约束和正则提取。
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        
        url = f"{final_base_url.rstrip('/')}/chat/completions"
        print(f"DEBUG: Calling Evaluator URL: {url}, Model: {model}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"DEBUG: LLM API error (Status {response.status_code}): {response.text}")
            return None
            
        res_json = response.json()
        if "choices" not in res_json or not res_json["choices"]:
            print(f"DEBUG: LLM API unexpected response structure: {res_json}")
            return None
            
        content = res_json["choices"][0]["message"]["content"]
        scores = extract_json_from_text(content)
        
        return {
            "scores": scores,
            "raw_content": content
        }
    except Exception as e:
        print(f"DEBUG: LLM API Exception: {e}")
        return None

async def run_evaluation(trace_id: str, db_session_factory):
    """后台任务：运行 RAG 评估"""
    print(f"\n--- DEBUG EVALUATION START: {trace_id} ---")
    db = db_session_factory()
    try:
        # 获取当前 trace 的所有节点
        all_nodes = db.query(TraceDB).filter(TraceDB.trace_id == trace_id).all()
        if not all_nodes:
            print(f"DEBUG: Trace {trace_id} not found")
            return

        # 寻找根节点 (通常是整个 chain)
        root_node = next((n for n in all_nodes if not n.parent_run_id), all_nodes[0])
        
        # 提取问答对
        question = ""
        answer = ""
        contexts = []
        original_reasoning = ""

        # 1. 尝试从输入输出中提取
        if root_node.inputs_json:
            try:
                inp = json.loads(root_node.inputs_json)
                if isinstance(inp, dict):
                    question = inp.get("question") or inp.get("input") or inp.get("query") or str(inp)
                else:
                    question = str(inp)
            except:
                question = root_node.inputs_json

        if root_node.outputs_json:
            try:
                out = json.loads(root_node.outputs_json)
                if isinstance(out, dict):
                    answer = out.get("answer") or out.get("output") or out.get("result") or str(out)
                else:
                    answer = str(out)
            except:
                answer = root_node.outputs_json

        # 2. 寻找检索节点提取 Contexts
        retriever_nodes = [n for n in all_nodes if n.run_type == "retriever"]
        for rn in retriever_nodes:
            try:
                out = json.loads(rn.outputs_json)
                docs = []
                if isinstance(out, list):
                    docs = out
                elif isinstance(out, dict):
                    # 兼容不同格式
                    for k, v in out.items():
                        if isinstance(v, list):
                            docs = v
                            break
                    
                for doc in docs:
                    if isinstance(doc, dict):
                        content = doc.get("page_content") or doc.get("text") or doc.get("content")
                        if not content and "metadata" in doc:
                            content = str(doc)
                        if content: contexts.append(content)
                    else:
                        contexts.append(str(doc))
            except Exception as e:
                print(f"DEBUG: Error parsing retriever outputs: {e}")

        # 如果 question 还是空的，尝试从全量节点中寻找可能的输入
        if not question or question == "{}":
            print("DEBUG: Question is empty, searching in other nodes...")
            for n in all_nodes:
                if n.inputs_json and len(n.inputs_json) > 5:
                    try:
                        inp = json.loads(n.inputs_json)
                        if isinstance(inp, str) and len(inp) > 5:
                            question = inp
                            break
                    except:
                        pass

        print(f"DEBUG: Data extracted - Q: {question[:100]}...")
        print(f"DEBUG: A: {answer[:100]}...")
        print(f"DEBUG: Contexts: {len(contexts)} items")

        if not question or question == "{}":
            print("DEBUG: Evaluation aborted - Question is still empty")
            return

        # 获取设置
        settings_db = db.query(SettingsDB).all()
        settings = {s.key: s.value for s in settings_db}
        eval_model = settings.get("eval_model") or "gpt-3.5-turbo"
        api_key = settings.get("eval_api_key")
        api_base = settings.get("eval_api_base")
        
        if not api_key:
            print("DEBUG: Evaluation skipped - No API Key in settings")
            return
            
        # 调用评估
        print(f"DEBUG: Calling evaluator with model: {eval_model}")
        eval_result = call_llm_evaluator(
            model=eval_model,
            question=question,
            answer=answer,
            contexts=contexts,
            api_key=api_key,
            base_url=api_base,
            reasoning=original_reasoning
        )
        
        if eval_result and eval_result.get("scores"):
            scores = eval_result["scores"]
            scores["eval_model"] = eval_model
            scores["reasoning"] = eval_result.get("raw_content") # 保存原始推理过程
            print(f"DEBUG: Evaluation successful: {scores}")
        else:
            print(f"DEBUG: Evaluation failed - no scores returned")
            scores = {
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": 0,
                "eval_error": True,
                "reasoning": eval_result.get("raw_content") if eval_result else "评估失败，未获得返回内容"
            }
        
        # 保存评分
        root_node.scores_json = json.dumps(scores)
        db.commit()
        print(f"DEBUG: Trace {trace_id} evaluation saved. Final Scores: {scores}")
        
    except Exception as e:
        print(f"DEBUG: Evaluation Exception for {trace_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        print(f"--- DEBUG EVALUATION END: {trace_id} ---\n")

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 数据库配置 - 优先从环境变量获取，方便 Docker 部署
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # 如果是 sqlite:///./data/tracemind.db 这种格式
    if DATABASE_URL.startswith("sqlite:///"):
        db_file_path = DATABASE_URL.replace("sqlite:///", "")
        # 如果是相对路径，相对于 BASE_DIR
        if not os.path.isabs(db_file_path):
            db_file_path = os.path.join(BASE_DIR, db_file_path)
        
        # 确保数据库所在目录存在
        os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
        engine = create_engine(f"sqlite:///{db_file_path}", connect_args={"check_same_thread": False})
    else:
        # 其他类型的数据库（如 PostgreSQL, MySQL）
        engine = create_engine(DATABASE_URL)
else:
    # 默认本地路径
    DB_PATH = os.path.join(BASE_DIR, "tracemind.db")
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

# 确保静态目录存在
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="TraceMind Server")

# 根路由重定向到 Dashboard
@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")

# 静态文件服务 - 将 static 目录映射到 /dashboard
app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# 数据库会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TraceDB(Base):
    __tablename__ = "traces"
    
    id = Column(String, primary_key=True, index=True) # 这里用 run_id
    project_name = Column(String, index=True)
    trace_id = Column(String, index=True)
    run_id = Column(String, unique=True, index=True)
    parent_run_id = Column(String, nullable=True)
    run_type = Column(String)
    metadata_json = Column(Text)
    inputs_json = Column(Text)
    outputs_json = Column(Text, nullable=True)
    scores_json = Column(Text, nullable=True) # 存储评分结果
    start_time = Column(Float)
    end_time = Column(Float, nullable=True)

class SettingsDB(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String)

Base.metadata.create_all(bind=engine)

# 初始化默认设置
def init_settings():
    db = SessionLocal()
    try:
        if not db.query(SettingsDB).filter(SettingsDB.key == "auto_cleanup").first():
            db.add(SettingsDB(key="auto_cleanup", value="false"))
        if not db.query(SettingsDB).filter(SettingsDB.key == "retention_days").first():
            db.add(SettingsDB(key="retention_days", value="7"))
        if not db.query(SettingsDB).filter(SettingsDB.key == "server_port").first():
            db.add(SettingsDB(key="server_port", value="8000"))
        db.commit()
    finally:
        db.close()

init_settings()

def perform_auto_cleanup():
    db = SessionLocal()
    try:
        auto_cleanup = db.query(SettingsDB).filter(SettingsDB.key == "auto_cleanup").first()
        if auto_cleanup and auto_cleanup.value == "true":
            retention_days = db.query(SettingsDB).filter(SettingsDB.key == "retention_days").first()
            days = int(retention_days.value) if retention_days else 7
            cutoff_time = time.time() - (days * 24 * 3600)
            db.query(TraceDB).filter(TraceDB.start_time < cutoff_time).delete()
            db.commit()
    except Exception as e:
        print(f"Auto cleanup error: {e}")
    finally:
        db.close()

class TraceRecord(BaseModel):
    project_name: str
    trace_id: str
    run_id: str
    parent_run_id: Optional[str] = None
    run_type: str
    metadata: Optional[Dict[str, Any]] = None
    inputs: Optional[Any] = None
    outputs: Optional[Any] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

@app.post("/traces")
async def receive_trace(record: TraceRecord, background_tasks: BackgroundTasks):
    # 每次收到新数据时，尝试执行自动清理
    background_tasks.add_task(perform_auto_cleanup)
    
    db = SessionLocal()
    try:
        # 检查是否已存在（处理 start/end 分开上报的情况）
        existing = db.query(TraceDB).filter(TraceDB.run_id == record.run_id).first()
        is_first_output = False
        
        if existing:
            # 记录是否是第一次获得输出
            if record.outputs and not existing.outputs_json:
                is_first_output = True
                
            if record.outputs: existing.outputs_json = json.dumps(record.outputs)
            if record.end_time: existing.end_time = record.end_time
            if record.start_time: existing.start_time = record.start_time
            if record.metadata: 
                # 合并 metadata
                old_meta = json.loads(existing.metadata_json) if existing.metadata_json else {}
                old_meta.update(record.metadata)
                existing.metadata_json = json.dumps(old_meta)
        else:
            # 新建记录
            db_record = TraceDB(
                id=record.run_id,
                project_name=record.project_name,
                trace_id=record.trace_id,
                run_id=record.run_id,
                parent_run_id=record.parent_run_id,
                run_type=record.run_type,
                metadata_json=json.dumps(record.metadata or {}),
                inputs_json=json.dumps(record.inputs or {}),
                outputs_json=json.dumps(record.outputs) if record.outputs else None,
                start_time=record.start_time or time.time(),
                end_time=record.end_time
            )
            db.add(db_record)
            if record.outputs:
                is_first_output = True
        
        db.commit()
        
        # 如果这是第一次获得输出，并且是根节点，触发评估
        if is_first_output and not record.parent_run_id:
            background_tasks.add_task(run_evaluation, record.trace_id, SessionLocal)
            
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        print(f"Error saving trace: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@app.get("/api/traces")
async def get_traces(project: Optional[str] = None):
    db = SessionLocal()
    try:
        query = db.query(TraceDB)
        if project:
            query = query.filter(TraceDB.project_name == project)
        # 只返回根节点，并按时间倒序
        traces = query.filter(TraceDB.parent_run_id == None).order_by(TraceDB.start_time.desc()).all()
        return traces
    finally:
        db.close()

@app.get("/api/traces/{trace_id}")
async def get_trace_detail(trace_id: str):
    db = SessionLocal()
    try:
        nodes = db.query(TraceDB).filter(TraceDB.trace_id == trace_id).order_by(TraceDB.start_time.asc()).all()
        if not nodes:
            raise HTTPException(status_code=404, detail="Trace not found")
        return nodes
    finally:
        db.close()

class ConnectionTest(BaseModel):
    project_name: str

@app.post("/test-connection")
async def test_connection(data: ConnectionTest):
    return {"status": "success", "message": f"Connected to TraceMind as project: {data.project_name}"}

@app.get("/api/settings")
async def get_settings():
    db = SessionLocal()
    try:
        settings = db.query(SettingsDB).all()
        return {s.key: s.value for s in settings}
    finally:
        db.close()

@app.post("/api/settings")
async def update_settings(settings: Dict[str, str]):
    db = SessionLocal()
    try:
        for k, v in settings.items():
            item = db.query(SettingsDB).filter(SettingsDB.key == k).first()
            if item:
                item.value = v
            else:
                db.add(SettingsDB(key=k, value=v))
        db.commit()
        return {"status": "success"}
    finally:
        db.close()

@app.delete("/api/traces/{trace_id}")
async def delete_trace(trace_id: str):
    db = SessionLocal()
    try:
        db.query(TraceDB).filter(TraceDB.trace_id == trace_id).delete()
        db.commit()
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        print(f"Delete error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """寻找可用的端口"""
    port = start_port
    while port < start_port + max_attempts:
        if not is_port_in_use(port):
            return port
        port += 1
    return start_port

def start_server():
    """读取配置并启动服务器"""
    import uvicorn
    db = SessionLocal()
    try:
        port_setting = db.query(SettingsDB).filter(SettingsDB.key == "server_port").first()
        configured_port = int(port_setting.value) if port_setting else 8000
    except:
        configured_port = 8000
    finally:
        db.close()
    
    # 自动检测端口占用
    actual_port = find_available_port(configured_port)
    
    if actual_port != configured_port:
        print(f"Warning: Port {configured_port} is already in use. Switching to port {actual_port}.")
        # 同步更新数据库
        db = SessionLocal()
        try:
            port_setting = db.query(SettingsDB).filter(SettingsDB.key == "server_port").first()
            if port_setting:
                port_setting.value = str(actual_port)
            else:
                db.add(SettingsDB(key="server_port", value=str(actual_port)))
            db.commit()
        except Exception as e:
            print(f"Failed to update port in database: {e}")
        finally:
            db.close()

    print(f"\n" + "="*50)
    print(f"TraceMind Server is running at:")
    print(f">>> http://localhost:{actual_port}/dashboard <<<")
    print(f"="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=actual_port)

if __name__ == "__main__":
    start_server()
