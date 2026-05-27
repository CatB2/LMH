"""多 Agent 管理器 - 统一入口 + 分层意图识别"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Optional

from app.services.aiops_service import AIOpsService
from app.services.rag_agent_service import rag_agent_service
from app.core.intent_detector import detect_intent, Intent
from loguru import logger


class AgentManager:
    """多 Agent 管理器"""

    def __init__(self):
        self._aiops_service = AIOpsService()
        self.agents: Dict[str, Any] = {
            "rag": rag_agent_service,
            "default": rag_agent_service,
            "aiops": self._aiops_service,
        }

    def get_agent(self, name: Optional[str]) -> Any:
        return self.agents.get(name or "default", self.agents["default"])

    def list_agents(self) -> Dict[str, str]:
        return {
            "default": "默认 RAG Agent",
            "rag": "RAG Agent",
            "aiops": "AIOps Agent",
        }

    async def _resolve_agent(
        self, question: str, agent_name: Optional[str]
    ) -> tuple[str, Any]:
        """如果未指定 agent，先做意图识别"""
        if agent_name:
            return agent_name, self.get_agent(agent_name)

        # 分层意图识别
        intent = await detect_intent(question)
        resolved = intent.intent.value
        logger.info(
            f"意图识别: {resolved} (confidence={intent.confidence})"
        )
        return resolved, self.get_agent(resolved)

    async def query(
        self,
        question: str,
        session_id: str,
        agent_name: Optional[str] = None,
        user_id: str = "",
        model: str = "",
    ) -> str:
        resolved_name, agent = await self._resolve_agent(question, agent_name)
        if hasattr(agent, "query"):
            return await agent.query(question, session_id=session_id, user_id=user_id, model=model)
        raise ValueError(f"Agent {resolved_name} 不支持 query 方法")

    async def query_stream(
        self,
        question: str,
        session_id: str,
        agent_name: Optional[str] = None,
        user_id: str = "",
        model: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        resolved_name, agent = await self._resolve_agent(question, agent_name)
        if hasattr(agent, "query_stream"):
            async for chunk in agent.query_stream(question, session_id=session_id, user_id=user_id, model=model):
                yield chunk
            return
        raise ValueError(f"Agent {resolved_name} 不支持 stream 方法")


agent_manager = AgentManager()
