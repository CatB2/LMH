"""长期记忆管理 - 提取、过滤、存储"""

import asyncio
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage
from loguru import logger

from app.core.user_memory import (
    get_user_profile,
    search_similar_memories,
    extract_memories,
    store_memories,
)


# 个人信息关键词（用于短问跳过过滤）
_PERSONAL_KEYWORDS = [
    "我", "我的", "我是", "我叫", "我有", "我喜欢", "我做", "我用",
    "之前", "以前", "最近", "正在", "打算", "想", "要",
]


class MemoryManager:
    """用户长期记忆管理"""

    def __init__(self):
        self._extract_counters: dict[str, int] = {}

    async def inject(
        self, messages: list, user_id: str, question: str
    ) -> list:
        """将用户长期记忆注入到消息上下文中"""
        if not user_id:
            return messages

        try:
            profile = await get_user_profile(user_id)
            related = await search_similar_memories(user_id, question, top_k=3)

            parts = []
            if profile:
                lines = [m["content"] for m in profile if m.get("content")]
                if lines:
                    parts.append("## 用户画像\n" + "\n".join(f"- {line}" for line in lines))

            if related:
                lines = [m["content"] for m in related if m.get("content")]
                if lines:
                    parts.append("## 相关记忆\n" + "\n".join(f"- {line}" for line in lines))

            if parts:
                msg = SystemMessage(
                    content="以下是关于该用户的长期记忆信息，请参考这些信息来个性化你的回答：\n\n"
                    + "\n\n".join(parts)
                )
                messages.insert(0, msg)
                logger.info(f"已注入 {len(profile)} 条画像 + {len(related)} 条相关记忆")

        except Exception as e:
            logger.warning(f"注入用户记忆失败（非关键错误）: {e}")

        return messages

    async def extract_and_store(
        self, user_id: str, session_id: str, question: str, answer: str
    ):
        """从问答中提取并存储长期记忆（异步非阻塞）"""
        if not user_id or not answer:
            return

        # 过滤 1：答案太短无记忆价值
        if len(answer.strip()) < 60:
            return

        # 过滤 2：短问题且无人称关键词
        q = question.strip()
        if len(q) < 15 and not any(kw in q for kw in _PERSONAL_KEYWORDS):
            return

        # 过滤 3：每 3 轮提取一次
        count = self._extract_counters.get(session_id, 0) + 1
        self._extract_counters[session_id] = count
        if count % 3 != 0:
            logger.debug(f"记忆提取跳过（第 {count} 轮，每 3 轮提取一次）")
            return

        try:
            memories = await extract_memories(question, answer)
            if memories:
                stored = await store_memories(user_id, memories)
                if stored:
                    logger.info(f"从会话中提取并存储了 {stored} 条长期记忆")
        except Exception as e:
            logger.warning(f"存储长期记忆失败（非关键错误）: {e}")

    def schedule_extract(
        self, user_id: str, session_id: str, question: str, answer: str
    ):
        """安排异步提取（不阻塞当前响应）"""
        if not user_id or not answer:
            return
        asyncio.create_task(
            self.extract_and_store(user_id, session_id, question, answer)
        )
