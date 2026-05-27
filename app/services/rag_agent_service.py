"""RAG Agent 编排"""

import asyncio
from typing import Any, AsyncGenerator, Dict

from langchain_core.messages import HumanMessage
from loguru import logger
from app.config import config
from app.core.llm_factory import llm_factory
from app.services.session_memory_store import SessionMemoryStore
from app.services.context_builder import ContextBuilder, TokenCounter
from app.services.memory_manager import MemoryManager
from app.tools import get_current_time, retrieve_knowledge
from app.tools.knowledge_tool import current_user_id as knowledge_user_id


class RagAgentService:

    def __init__(self, streaming: bool = True):
        self.model_name = config.rag_model
        self.streaming = streaming
        self.model = llm_factory.create_chat_model(model=self.model_name, temperature=0.7, streaming=streaming)
        self.summary_model = llm_factory.create_chat_model(model=self.model_name, temperature=0.3, streaming=False)
        self.memory_store = SessionMemoryStore()
        self.token_counter = TokenCounter(self.model_name)
        self.context_builder = ContextBuilder(self.memory_store, self.summary_model, self.token_counter)
        self.memory_manager = MemoryManager()
        self.agent = None
        self._agent_ready = False
        self._tools = [retrieve_knowledge, get_current_time]
        logger.info(f"RAG Agent 初始化完成, model={self.model_name}")

    async def _init_agent(self, model: str = ""):
        use_model = model or self.model_name
        if self._agent_ready and self.model_name == use_model:
            return
        # 模型变了，重建 agent
        if use_model != self.model_name:
            self.model = llm_factory.create_chat_model(model=use_model, temperature=0.7, streaming=self.streaming)
            self.model_name = use_model
            self._agent_ready = False
            logger.info(f"切换模型: {use_model}")
        if self._agent_ready:
            return
        from app.agent.rag_agent import create_rag_agent
        self.agent, _ = await create_rag_agent(self.model, self._tools)
        self._agent_ready = True

    async def _check_limits(self, session_id: str):
        stats = await self.memory_store.get_history_stats(session_id)
        turns = stats["message_count"] // 2
        if turns >= config.session_history_message_limit:
            raise ValueError(f"已达对话上限（{config.session_history_message_limit} 轮），请新建会话")

    async def query(self, question: str, session_id: str, user_id: str = "", model: str = "") -> str:
        try:
            await self._init_agent(model)
            await self._check_limits(session_id)
            self.memory_store.touch(session_id)
            context = await self.context_builder.build_messages(session_id)
            messages = [*context, HumanMessage(content=question)]
            messages = await self.memory_manager.inject(messages, user_id, question)
            token = knowledge_user_id.set(user_id)
            try:
                result = await self.agent.ainvoke(input={"messages": messages})
            finally:
                knowledge_user_id.reset(token)
            msgs = result.get("messages", [])
            if not msgs:
                return ""
            last = msgs[-1]
            answer = last.content if hasattr(last, "content") else str(last)
            await self.memory_store.append_message(session_id, role="user", content=question, user_id=user_id)
            await self.memory_store.append_message(session_id, role="assistant", content=answer, user_id=user_id)
            asyncio.create_task(self.context_builder.save_context(session_id))
            if user_id:
                self.memory_manager.schedule_extract(user_id, session_id, question, answer)
            return answer
        except Exception as e:
            logger.error(f"[会话 {session_id}] 查询失败: {e}")
            raise

    async def query_stream(
        self, question: str, session_id: str, user_id: str = "", model: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            await self._init_agent(model)
            await self._check_limits(session_id)
            context = await self.context_builder.build_messages(session_id)
            messages = [*context, HumanMessage(content=question)]
            messages = await self.memory_manager.inject(messages, user_id, question)
            token = knowledge_user_id.set(user_id)
            chunks: list[str] = []
            try:
                async for tok, _ in self.agent.astream(input={"messages": messages}, stream_mode="messages"):
                    if type(tok).__name__ not in ("AIMessage", "AIMessageChunk"):
                        continue
                    text = _extract_text(tok)
                    if text:
                        chunks.append(text)
                        yield {"type": "content", "data": text}
            finally:
                knowledge_user_id.reset(token)
            final = "".join(chunks).strip()
            if final:
                await self.memory_store.append_message(session_id, role="user", content=question, user_id=user_id)
                await self.memory_store.append_message(session_id, role="assistant", content=final, user_id=user_id)
                asyncio.create_task(self.context_builder.save_context(session_id))
            yield {"type": "complete"}
            if final and user_id:
                self.memory_manager.schedule_extract(user_id, session_id, question, final)
        except Exception as e:
            logger.error(f"[会话 {session_id}] 流式查询失败: {e}")
            yield {"type": "error", "data": str(e)}

    async def get_session_history(self, session_id: str) -> list:
        try:
            return [
                {"role": m["role"], "content": m["content"], "timestamp": m["timestamp"]}
                for m in await self.memory_store.get_history(session_id)
            ]
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, {e}")
            return []

    async def clear_session(self, session_id: str) -> bool:
        try:
            await self.memory_store.clear_session(session_id)
            return True
        except Exception as e:
            logger.error(f"清空会话失败: {session_id}, {e}")
            return False


def _extract_text(token) -> str | None:
    blocks = getattr(token, "content_blocks", None)
    if blocks and isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                return b["text"]
    t = getattr(token, "content", None)
    if t and isinstance(t, str) and t.strip():
        return t
    return None


rag_agent_service = RagAgentService(streaming=True)
