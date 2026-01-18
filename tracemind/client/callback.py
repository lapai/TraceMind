import requests
import time
import uuid
import json
import atexit
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

class TraceMindCallbackHandler(BaseCallbackHandler):
    # 创建一个全局的线程池，并确保在进程退出时等待任务完成
    _executor = ThreadPoolExecutor(max_workers=5)

    def __init__(self, server_url: str = "http://localhost:8000", project_name: str = "Default-Project"):
        self.server_url = server_url
        self.project_name = project_name
        self.trace_id = str(uuid.uuid4())
        
        # 注册优雅停机函数
        atexit.register(self.shutdown)

    def shutdown(self):
        """确保所有后台任务完成"""
        print(f"\n[TraceMind] 正在上报剩余监控数据，请稍候...")
        self._executor.shutdown(wait=True)
        print(f"[TraceMind] 上报完成。")

    def _serialize(self, obj: Any) -> Any:
        """处理 LangChain 特有对象的序列化"""
        if isinstance(obj, Document):
            return {"page_content": obj.page_content, "metadata": obj.metadata}
        if hasattr(obj, "to_json"):
            try:
                return obj.to_json()
            except:
                return str(obj)
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize(i) for i in obj]
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return str(obj)

    def _post_request(self, serialized_record: Dict[str, Any]):
        """执行实际的 HTTP 请求（由线程池调用）"""
        try:
            requests.post(f"{self.server_url}/traces", json=serialized_record, timeout=5)
        except Exception as e:
            pass

    def _send_trace(self, record: Dict[str, Any]):
        """异步发送追踪数据，不阻塞主流程"""
        try:
            # 在主线程先完成序列化，确保数据一致性
            serialized_record = self._serialize(record)
            # 丢进后台线程池执行 HTTP 请求
            self._executor.submit(self._post_request, serialized_record)
        except:
            pass

    def _get_name(self, serialized: Optional[Dict[str, Any]], parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any) -> str:
        node_name = kwargs.get('name', 'laopai')
        if parent_run_id is None and node_name and node_name != 'laopai':
            return node_name

        tags = kwargs.get('tags') or []
        metadata = kwargs.get('metadata') or {}
        run_type = kwargs.get('run_type')

        if not tags and not metadata:
            return node_name

        if tags:
            first_tag = tags[0]
            if first_tag in ['seq:step:1', 'seq:step:2', 'seq:step:4']:
                return node_name
            if first_tag == 'seq:step:3':
                model_name = (
                    metadata.get('ls_model_name') or 
                    kwargs.get('invocation_params', {}).get('model') or 
                    kwargs.get('invocation_params', {}).get('model_name')
                )
                if model_name:
                    return model_name
                return 'DeepSeek LLM'

        if metadata.get('ls_embedding_provider') == 'OpenAIEmbeddings':
            return node_name

        if not metadata and run_type is None:
            return node_name

        return 'laopai'

    def on_chain_start(
        self, serialized: Optional[Dict[str, Any]], inputs: Dict[str, Any], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any
    ) -> Any:
        name = self._get_name(serialized, parent_run_id, **kwargs)
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "run_type": "chain",
            "inputs": inputs,
            "start_time": time.time(),
            "metadata": {"name": name, "tags": kwargs.get("tags", [])}
        }
        self._send_trace(record)

    def on_llm_start(
        self, serialized: Optional[Dict[str, Any]], prompts: List[str], *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any
    ) -> Any:
        name = self._get_name(serialized, parent_run_id, **kwargs)
        if name in ["RAG 核心链路", "Unnamed"]:
            name = "DeepSeek LLM"
            
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "run_type": "llm",
            "inputs": {"prompts": prompts},
            "start_time": time.time(),
            "metadata": {
                "name": name, 
                "tags": kwargs.get("tags", []),
                **kwargs.get("invocation_params", {})
            }
        }
        self._send_trace(record)

    def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs: Any) -> Any:
        outputs = {"generations": [[g.text for g in gen] for gen in response.generations]}
        usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
        elif response.llm_output and "usage" in response.llm_output:
            usage = response.llm_output["usage"]

        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "run_type": "llm",
            "outputs": outputs,
            "end_time": time.time(),
            "metadata": {"usage": usage}
        }
        self._send_trace(record)

    def on_retriever_start(
        self, serialized: Optional[Dict[str, Any]], query: str, *, run_id: uuid.UUID, parent_run_id: Optional[uuid.UUID] = None, **kwargs: Any
    ) -> Any:
        name = self._get_name(serialized, parent_run_id, **kwargs)
        if name in ["RAG 核心链路", "Unnamed", "laopai"]:
            name = "Knowledge Base Retriever"
            
        metadata = {"name": name}
        if serialized and "kwargs" in serialized:
            metadata.update(serialized["kwargs"])
        if "metadata" in kwargs:
            metadata.update(kwargs["metadata"])
        tags = kwargs.get("tags", [])
            
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "run_type": "retriever",
            "inputs": {"query": query},
            "start_time": time.time(),
            "metadata": {**metadata, "tags": tags}
        }
        self._send_trace(record)

    def on_retriever_end(self, documents: List[Document], *, run_id: uuid.UUID, **kwargs: Any) -> Any:
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "run_type": "retriever",
            "outputs": {"documents": [self._serialize(d) for d in documents]},
            "end_time": time.time(),
        }
        self._send_trace(record)

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: uuid.UUID, **kwargs: Any) -> Any:
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "run_type": "chain",
            "outputs": outputs,
            "end_time": time.time(),
        }
        self._send_trace(record)

    def on_chain_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> Any:
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "run_type": "chain",
            "outputs": {"error": str(error)},
            "end_time": time.time(),
        }
        self._send_trace(record)

    def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> Any:
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "run_type": "llm",
            "outputs": {"error": str(error)},
            "end_time": time.time(),
        }
        self._send_trace(record)

    def on_retriever_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> Any:
        record = {
            "project_name": self.project_name,
            "trace_id": self.trace_id,
            "run_id": str(run_id),
            "run_type": "retriever",
            "outputs": {"error": str(error)},
            "end_time": time.time(),
        }
        self._send_trace(record)