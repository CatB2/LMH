"""分层意图识别

第一层：关键词规则（0ms）
第二层：LLM 分类（200-500ms）
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from app.config import config


class Intent(str, Enum):
    RAG = "rag"          # 知识检索：问文档内容、查资料
    AIOPS = "aiops"      # 系统诊断：查日志、看监控、分析故障
    CHAT = "chat"        # 普通对话：打招呼、闲聊


class IntentResult(BaseModel):
    """意图识别结果"""
    intent: Intent = Field(..., description="识别到的意图")
    confidence: float = Field(default=1.0, ge=0, le=1, description="置信度")


# ── 第一层：关键词规则 ──

_RAG_KEYWORDS = [
    "文档", "知识库", "上传", "资料", "文件", "论文",
    "文档里说", "根据文件", "文档中提到",
    "搜索一下", "查一下", "找找看",
    "retrieve", "knowledge", "document",
]

_AIOPS_KEYWORDS = [
    "CPU", "内存", "磁盘", "日志", "监控",
    "告警", "诊断", "故障", "服务器",
    "系统", "性能", "排查", "分析系统",
    "指标", "metric", "alert", "monitor",
    "慢查询", "连接数", "负载",
]

_HIGH_CONFIDENCE_RAG = [
    "文档里说", "根据文件", "文档中提到",
    "上传的文档", "知识库",
]

_HIGH_CONFIDENCE_AIOPS = [
    "诊断系统", "排查故障", "查看告警",
    "CPU使用率", "内存使用率", "系统监控",
    "分析一下系统", "服务器状态",
]


def _rule_based_detect(question: str) -> dict:
    """第一层：关键词规则识别

    Returns:
        dict: {"certain": bool, "intent": str}
          certain=True  → 可直接使用该 intent
          certain=False → 需要第二层 LLM 确认
    """
    score = {"rag": 0, "aiops": 0, "chat": 0}

    # 高置信度关键词（优先匹配）
    for kw in _HIGH_CONFIDENCE_RAG:
        if kw in question:
            score["rag"] += 3
    for kw in _HIGH_CONFIDENCE_AIOPS:
        if kw in question:
            score["aiops"] += 3

    # 普通关键词
    for kw in _RAG_KEYWORDS:
        if kw in question:
            score["rag"] += 1
    for kw in _AIOPS_KEYWORDS:
        if kw in question:
            score["aiops"] += 1

    max_intent = max(score, key=score.get)
    max_score = score[max_intent]
    sorted_scores = sorted(score.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

    if max_score >= 3 and (max_score - second_score) >= 2:
        # 某个意图明显领先
        return {"certain": True, "intent": max_intent}
    elif max_score >= 1:
        # 有匹配但不明确
        return {"certain": False, "intent": max_intent}
    else:
        # 没有关键词匹配
        return {"certain": False, "intent": "chat"}


# ── 第二层：LLM 分类 ──

_SYSTEM_PROMPT = """你是一个意图分类器。判断用户输入属于以下三类：

- rag：用户询问文档内容、知识库资料、上传的文件，需要从向量数据库检索
- aiops：用户询问系统监控、日志分析、故障诊断，需要调用 MCP 工具
- chat：普通对话、打招呼、闲聊、不需要调用任何工具的问题

请以JSON格式输出，包含intent字段（字符串）和confidence字段（浮点数）。"""


async def _llm_detect(question: str) -> IntentResult:
    """第二层：LLM 精确分类"""
    try:
        from app.core.llm_factory import llm_factory

        llm = llm_factory.create_chat_model(
            model=config.rag_model,
            temperature=0,
            streaming=False,
        )

        structured_llm = llm.with_structured_output(IntentResult)
        result = await structured_llm.ainvoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
        return result
    except Exception as e:
        logger.warning(f"LLM 意图识别失败: {e}，默认 chat")
        return IntentResult(intent=Intent.CHAT, confidence=0)


# ── 对外接口 ──


async def detect_intent(question: str) -> IntentResult:
    """分层意图识别

    1. 关键词规则匹配命中且明确 → 直接返回，0ms
    2. 关键词匹配模糊或未命中 → 走 LLM 分类
    """
    rule_result = _rule_based_detect(question)

    if rule_result["certain"]:
        logger.debug(f"意图识别(规则层): {rule_result['intent']}")
        return IntentResult(intent=Intent(rule_result["intent"]), confidence=1.0)

    # LLM 分类
    result = await _llm_detect(question)
    logger.debug(f"意图识别(LLM层): {result.intent.value} (confidence={result.confidence})")
    return result
