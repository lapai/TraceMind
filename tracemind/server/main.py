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
import socket

# Helper for port detection
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    port = start_port
    while port < start_port + max_attempts:
        if not is_port_in_use(port):
            return port
        port += 1
    return start_port

# Evaluation metrics
def extract_json_from_text(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
    return None

def call_llm_evaluator(model: str, question: str, answer: str, contexts: List[str], api_key: str = None, base_url: str = None, reasoning: str = None):
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

请先输出你的详细 analysis 过程，最后再输出 JSON 评分，例如：
---
分析报告：...
---
{{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7}}
"""
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    final_base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    if not final_api_key: return None

    try:
        headers = {"Authorization": f"Bearer {final_api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        url = f"{final_base_url.rstrip('/')}/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200: return None
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        extracted = extract_json_from_text(content)
        return {"scores": extracted, "raw_content": content}
    except:
        return None

def run_evaluation(trace_id: str):
    time.sleep(2)
    db = SessionLocal()
    try:
        all_nodes = db.query(TraceDB).filter(TraceDB.trace_id == trace_id).all()
        if not all_nodes: return
        root_nodes = [n for n in all_nodes if n.parent_run_id is None and n.run_type == "chain"]
        if len(root_nodes) > 1: root_nodes.sort(key=lambda n: len(n.inputs_json or ""), reverse=True)
        root_node = next((n for n in root_nodes if n.outputs_json), None)
        if not root_node and root_nodes: root_node = root_nodes[0]
        retriever_node = next((n for n in all_nodes if n.run_type == "retriever"), None)
        if not root_node: return
        
        # Extract Question
        question = ""
        try:
            inputs = json.loads(root_node.inputs_json)
            if isinstance(inputs, dict):
                question = inputs.get("question") or inputs.get("input") or inputs.get("query") or inputs.get("content")
                if not question:
                    str_vals = [v for v in inputs.values() if isinstance(v, str)]
                    if str_vals: question = str_vals[0]
            if not question: question = str(inputs)
        except: question = root_node.inputs_json
        
        # Extract Answer
        answer = ""
        original_reasoning = ""
        try:
            outputs = json.loads(root_node.outputs_json)
            if isinstance(outputs, dict):
                answer = outputs.get("output") or outputs.get("text") or outputs.get("answer") or outputs.get("result")
                original_reasoning = outputs.get("reasoning_content") or outputs.get("thinking") or outputs.get("thought")
                if not answer and "generations" in outputs:
                    try:
                        gen = outputs["generations"][0][0]
                        answer = gen.get("text")
                        if not original_reasoning: original_reasoning = gen.get("message", {}).get("reasoning_content")
                    except: pass
                if not answer:
                    str_vals = [v for v in outputs.values() if isinstance(v, str)]
                    if str_vals: answer = str_vals[0]
            if not answer: answer = root_node.outputs_json or ""
        except: answer = root_node.outputs_json or ""

        # Extract Contexts
        contexts = []
        if retriever_node and retriever_node.outputs_json:
            try:
                ret_outputs = json.loads(retriever_node.outputs_json)
                docs = []
                if isinstance(ret_outputs, dict) and "documents" in ret_outputs: docs = ret_outputs["documents"]
                elif isinstance(ret_outputs, list): docs = ret_outputs
                for doc in docs:
                    if isinstance(doc, dict):
                        content = doc.get("page_content") or doc.get("text") or doc.get("content")
                        if content: contexts.append(content)
                    else: contexts.append(str(doc))
            except: pass

        settings_db = db.query(SettingsDB).all()
        settings = {s.key: s.value for s in settings_db}
        eval_model = settings.get("eval_model") or "gpt-3.5-turbo"
        api_key = settings.get("eval_api_key")
        api_base = settings.get("eval_api_base")
        if not api_key: return
            
        eval_result = call_llm_evaluator(model=eval_model, question=question, answer=answer, contexts=contexts, api_key=api_key, base_url=api_base, reasoning=original_reasoning)
        if eval_result and eval_result.get("scores"):
            scores = eval_result["scores"]
            scores["eval_model"] = eval_model
            scores["reasoning"] = eval_result.get("raw_content")
        else:
            scores = {"faithfulness": 0, "answer_relevancy": 0, "context_precision": 0, "eval_error": True}
        root_node.scores_json = json.dumps(scores)
        db.commit()
    except: pass
    finally: db.close()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DB_PATH = os.path.join(BASE_DIR, "tracemind.db")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="TraceMind Server")

@app.get("/")
async def root(): return RedirectResponse(url="/dashboard")

app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# DB
engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TraceDB(Base):
    __tablename__ = "traces"
    id = Column(String, primary_key=True, index=True)
    project_name = Column(String, index=True)
    trace_id = Column(String, index=True)
    run_id = Column(String, unique=True, index=True)
    parent_run_id = Column(String, nullable=True)
    run_type = Column(String)
    metadata_json = Column(Text)
    inputs_json = Column(Text)
    outputs_json = Column(Text, nullable=True)
    scores_json = Column(Text, nullable=True)
    start_time = Column(Float)
    end_time = Column(Float, nullable=True)

class SettingsDB(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String)

Base.metadata.create_all(bind=engine)

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
    finally: db.close()

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
    except: pass
    finally: db.close()

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
    background_tasks.add_task(perform_auto_cleanup)
    db = SessionLocal()
    try:
        existing = db.query(TraceDB).filter(TraceDB.run_id == record.run_id).first()
        is_first_output = False
        if existing:
            if record.outputs and not existing.outputs_json: is_first_output = True
            if record.outputs: existing.outputs_json = json.dumps(record.outputs)
            if record.end_time: existing.end_time = record.end_time
            if record.start_time: existing.start_time = record.start_time
            if record.metadata: 
                old_meta = json.loads(existing.metadata_json) if existing.metadata_json else {}
                old_meta.update(record.metadata)
                existing.metadata_json = json.dumps(old_meta)
        else:
            if record.outputs: is_first_output = True
            new_trace = TraceDB(
                id=record.run_id, project_name=record.project_name, trace_id=record.trace_id,
                run_id=record.run_id, parent_run_id=record.parent_run_id, run_type=record.run_type,
                metadata_json=json.dumps(record.metadata or {}), inputs_json=json.dumps(record.inputs or {}),
                outputs_json=json.dumps(record.outputs) if record.outputs else None,
                start_time=record.start_time or time.time(), end_time=record.end_time
            )
            db.add(new_trace)
        db.commit()
        if record.parent_run_id is None and record.run_type == "chain" and is_first_output:
            background_tasks.add_task(run_evaluation, record.trace_id)
    except: db.rollback()
    finally: db.close()
    return {"status": "success"}

@app.get("/settings")
async def get_settings():
    db = SessionLocal()
    try:
        settings = db.query(SettingsDB).all()
        return {s.key: s.value for s in settings}
    finally: db.close()

@app.post("/settings")
async def update_settings(settings: Dict[str, str]):
    db = SessionLocal()
    try:
        for key, value in settings.items():
            db_setting = db.query(SettingsDB).filter(SettingsDB.key == key).first()
            if db_setting: db_setting.value = value
            else: db.add(SettingsDB(key=key, value=value))
        db.commit()
        return {"status": "success"}
    except: db.rollback(); raise HTTPException(status_code=500)
    finally: db.close()

def start_server():
    import uvicorn
    db = SessionLocal()
    try:
        port_setting = db.query(SettingsDB).filter(SettingsDB.key == "server_port").first()
        configured_port = int(port_setting.value) if port_setting else 8000
    except: configured_port = 8000
    finally: db.close()
    
    actual_port = find_available_port(configured_port)
    if actual_port != configured_port:
        print(f"Warning: Port {configured_port} is already in use. Switching to port {actual_port}.")
        db = SessionLocal()
        try:
            port_setting = db.query(SettingsDB).filter(SettingsDB.key == "server_port").first()
            if port_setting: port_setting.value = str(actual_port)
            else: db.add(SettingsDB(key="server_port", value=str(actual_port)))
            db.commit()
        except Exception as e: print(f"Failed to update port in database: {e}")
        finally: db.close()
    
    print(f"\n" + "="*50)
    print(f"TraceMind Server is running at:")
    print(f">>> http://localhost:{actual_port}/dashboard <<<")
    print(f"="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=actual_port)

if __name__ == "__main__":
    start_server()