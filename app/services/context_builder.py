"""上下文构建模块 - 会话摘要压缩 + 消息组装"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from app.config import config


class TokenCounter:
    """Token 估算"""

    def __init__(self, model_name: str):
        self._encoder = self._init(model_name)

    def _init(self, model_name: str):
        try:
            import tiktoken
            try:
                return tiktoken.encoding_for_model(model_name)
            except Exception:
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                return max(1, len(text.strip()) // 4)
        return max(1, len(text.strip()) // 4)


class ContextBuilder:
    """管理会话上下文的构建与压缩"""

    def __init__(self, memory_store, summary_model, token_counter: TokenCounter):
        self.memory_store = memory_store
        self.summary_model = summary_model
        self.token_counter = token_counter

    async def build_messages(self, session_id: str) -> list[BaseMessage]:
        """构建上下文：摘要 + 未压缩消息（save_context 已保证在 token 预算内）"""
        summary = await self.memory_store.get_summary(session_id)
        compressed = await self.memory_store.get_compressed(session_id)
        all_msgs = await self.memory_store.get_history(session_id)
        recent = all_msgs[compressed:] if compressed else all_msgs

        context: list[BaseMessage] = []
        if summary:
            context.append(SystemMessage(content=f"以下是历史对话摘要：\n{summary}"))
        for msg in recent:
            if msg["role"] == "user":
                context.append(HumanMessage(content=msg["content"]))
            else:
                context.append(AIMessage(content=msg["content"]))
        return context

    async def save_context(self, session_id: str):
        """检查未压缩部分是否超出 token 限制，超出则压缩"""
        compressed = await self.memory_store.get_compressed(session_id)
        all_msgs = await self.memory_store.get_history(session_id)
        uncompresseds = all_msgs[compressed:] if compressed else all_msgs
        if not uncompresseds:
            return
        total_tokens = sum(m["token_count"] + 4 for m in uncompresseds)
        if total_tokens <= config.rag_context_recent_token_keep:
            return
        excess = total_tokens - config.rag_context_recent_token_keep
        to_compress = []
        chunk_tokens = 0
        for msg in uncompresseds:
            to_compress.append(msg)
            chunk_tokens += msg["token_count"] + 4
            if chunk_tokens >= excess:
                break
        old_summary = await self.memory_store.get_summary(session_id)
        new_summary = await self._compress_to_summary(old_summary, to_compress)
        await self.memory_store.set_summary(session_id, new_summary)
        await self.memory_store.set_compressed(session_id, compressed + len(to_compress))
        logger.info(f"压缩完成: {len(to_compress)} 条 → 新摘要")

    async def _compress_to_summary(self, old_summary: str, messages: list) -> str:
        conversation = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in messages
        )
        prompt = "请将以下信息融合成一段简洁连贯的摘要：\n\n"
        if old_summary:
            prompt += f"【已有摘要】\n{old_summary}\n\n"
        prompt += f"【近期对话】\n{conversation}"
        result = await self.summary_model.ainvoke(
            [SystemMessage(content="把对话压缩为摘要。"), HumanMessage(content=prompt)]
        )
        return result.content if hasattr(result, "content") else str(result)
