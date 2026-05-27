"""
Planner 节点：制定执行计划
"""

from textwrap import dedent
import json
import re
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from loguru import logger

from app.tools import retrieve_knowledge
from .state import PlanExecuteState

# 由 AIOpsService 在运行前注入（避免放入 state 导致序列化问题）
_tools: list = []
_tools_description: str = ""
_llm = None


class Plan(BaseModel):
    """计划的输出格式"""
    steps: List[str] = Field(
        description="完成任务所需的不同步骤，按顺序执行，每一步都建立在前一步的基础上。"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

                {experience_context}

                对于给定的任务，请创建一个简单的、逐步的计划来完成它。计划应该：
                - 将任务分解为逻辑上独立的步骤
                - 每个步骤应该明确使用哪些工具来获取信息
                - 步骤之间应该有清晰的依赖关系
                - 步骤描述要具体、可操作
                - 如果有相关经验文档，请参考其中的方法和步骤制定计划
                - 以 JSON 格式输出你的计划（步骤列表）
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


def _parse_plan_response(llm_response) -> list:
    """从 LLM 响应中解析计划步骤列表（支持多种 JSON 格式）"""
    content = ""
    if hasattr(llm_response, "content"):
        content = llm_response.content
    elif isinstance(llm_response, dict):
        content = llm_response.get("content", str(llm_response))
    else:
        content = str(llm_response)

    if not content:
        logger.warning("LLM 返回空内容")
        return []

    # 尝试提取 JSON（支持 markdown 代码块和纯 JSON）
    json_str = content.strip()
    # 从 ```json ... ``` 中提取
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
    if code_block:
        json_str = code_block.group(1).strip()
    # 从 { ... } 中提取
    brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"无法解析 LLM 输出为 JSON，尝试直接提取步骤列表...")
        return _extract_steps_from_text(content)

    # 支持多种 JSON 结构
    if isinstance(data, dict):
        steps = data.get("steps") or data.get("plan") or data.get("steps_list")

        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps
        if isinstance(steps, list):
            # steps 是对象列表（每个对象有 step/description/tool 等字段）
            result = []
            for item in steps:
                if isinstance(item, dict):
                    desc = item.get("description") or item.get("task") or ""
                    step_num = item.get("step", "")
                    prefix = f"{step_num}. " if isinstance(step_num, (int, str)) and str(step_num).strip() else ""
                    text = f"{prefix}{desc}".strip() or str(item)
                    result.append(text)
                else:
                    result.append(str(item))
            if result:
                return result
    elif isinstance(data, list):
        if all(isinstance(s, str) for s in data):
            return data
        # 列表中的对象可能有 step/task/description 字段
        result = []
        for item in data:
            if isinstance(item, dict):
                desc = item.get("description") or item.get("task") or item.get("step") or str(item)
                result.append(str(desc))
            else:
                result.append(str(item))
        return result

    logger.warning(f"JSON 结构不符合预期: {data}")
    return []


def _extract_steps_from_text(text: str) -> list:
    """从纯文本中提取步骤列表（按编号或列表标记）"""
    steps = []
    # 匹配 "1. step" 或 "1、step" 或 "- step" 或 "* step"
    for line in text.split("\n"):
        line = line.strip()
        # 跳过空行
        if not line:
            continue
        # 匹配编号列表
        match = re.match(r"^\d+[.、)\s]\s*(.+)$", line)
        if match:
            steps.append(match.group(1).strip())
            continue
        # 匹配无序列表
        match = re.match(r"^[-*]\s+(.+)$", line)
        if match:
            steps.append(match.group(1).strip())
    return steps


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    流程：
    1. 查询内部文档，获取相关经验和最佳实践
    2. 基于经验和可用工具制定执行计划
    """
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    # 使用 AIOpsService 注入的工具列表和 LLM（模块级全局变量，由 service 在运行前设置）
    if not _tools:
        logger.error("Planner 缺少工具列表，无法制定计划")
        return {"plan": ["收集相关信息", "分析数据", "生成报告"]}

    if not _llm:
        logger.error("Planner 缺少 LLM 实例")
        return {"plan": ["收集相关信息", "分析数据", "生成报告"]}

    try:
        # 步骤1: 查询内部文档获取相关经验
        logger.info("查询内部文档，寻找相关经验...")
        experience_docs = ""
        try:
            context_str = await retrieve_knowledge.ainvoke({"query": input_text})
            if context_str and context_str.strip():
                experience_docs = context_str
                logger.info(f"找到相关经验文档，长度: {len(experience_docs)}")
            else:
                logger.info("未找到相关经验文档")
        except Exception as e:
            logger.warning(f"查询内部文档失败: {e}")

        # 格式化经验文档上下文
        if experience_docs:
            experience_context = dedent(f"""
                ## 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：

                {experience_docs}

                ---
            """).strip()
        else:
            experience_context = ""

        # 使用共享 LLM 生成计划（手动 JSON 解析，避免 OutputParserException）
        planner_chain = planner_prompt | _llm

        llm_response = await planner_chain.ainvoke({
            "messages": [("user", input_text)],
            "tools_description": _tools_description,
            "experience_context": experience_context,
        })

        plan_steps = _parse_plan_response(llm_response)
        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        return {"plan": plan_steps}

    except Exception as e:
        logger.error(f"生成计划失败: {type(e).__name__}", exc_info=True)
        return {"plan": ["收集相关信息", "分析数据", "生成报告"]}
