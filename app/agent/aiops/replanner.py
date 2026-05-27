"""
Replanner 节点：重新规划或生成最终响应
"""

import json
import re
from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from loguru import logger

from .state import PlanExecuteState

# 由 AIOpsService 在运行前注入（避免放入 state 导致序列化问题）
_tools: list = []
_tools_description: str = ""
_llm = None


class Response(BaseModel):
    """最终响应的格式"""
    response: str = Field(description="对用户的最终响应")


class Act(BaseModel):
    """重新规划的输出格式"""
    action: str = Field(
        description="""下一步的行动：
        - 'continue': 当前计划合理，继续执行下一个步骤
        - 'replan': 当前计划需要调整
        - 'respond': 信息充足，生成最终响应"""
    )
    new_steps: List[str] = Field(
        default_factory=list,
        description="新步骤列表（action 为 'replan' 时有效）",
    )


replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                你有三个选择，请严格按照以下条件判断：

                **1. 'respond' - 生成最终响应**
                   仅当同时满足以下所有条件时才可选择：
                   - 已执行步骤数 >= 3（至少执行了3个步骤）
                   - 已获取到了足够回答用户问题的关键数据
                   - 剩余步骤不再必要

                **2. 'continue' - 继续执行当前计划**
                   默认选择，除非满足 respond 条件。

                **3. 'replan' - 调整计划**
                   - 只能在当前计划明显不合理时使用
                   - 新步骤数必须 <= 剩余步骤数（只能简化不能加多）
                   - 已执行 >= 5 次时禁止 replan

                决策优先级：continue(默认) > respond(条件满足时) > replan(仅必要时)

                以 JSON 格式输出你的决策（action 和可选的 new_steps）
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
                - 以 JSON 格式输出最终响应
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def _generate_response(state: PlanExecuteState) -> Dict[str, Any]:
    """生成最终响应"""
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])

    if not _llm:
        return {"response": _build_fallback_report(input_text, past_steps)}

    try:
        messages = [
            ("user", f"原始任务: {input_text}"),
            ("user", f"执行历史:\n{execution_history}"),
            ("user", "请基于以上信息生成全面的最终响应"),
        ]

        prompt_result = response_prompt | _llm
        llm_response = await prompt_result.ainvoke({"messages": messages})

        content = ""
        if hasattr(llm_response, "content"):
            content = llm_response.content
        elif isinstance(llm_response, dict):
            content = llm_response.get("content", str(llm_response))
        else:
            content = str(llm_response)

        final_response = _extract_json_response(content)
        if not final_response:
            final_response = _extract_markdown_response(content)

        if not final_response:
            logger.warning("无法从 LLM 输出中提取响应，使用原始内容")
            final_response = content

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")
        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败: {type(e).__name__}")
        return {"response": _build_fallback_report(input_text, past_steps)}


def _parse_act_response(llm_response) -> tuple:
    """从 LLM 响应中解析 replanner 决策 (action, new_steps)"""
    content = ""
    if hasattr(llm_response, "content"):
        content = llm_response.content
    elif isinstance(llm_response, dict):
        content = llm_response.get("content", str(llm_response))
    else:
        content = str(llm_response)

    json_str = content.strip()
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
    if code_block:
        json_str = code_block.group(1).strip()
    brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)

    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            action = data.get("action", "continue")
            new_steps = data.get("new_steps") or data.get("steps") or []
            if isinstance(new_steps, list):
                return action, new_steps
            return action, []
    except json.JSONDecodeError:
        pass

    return "continue", []


def _extract_json_response(content: str) -> str:
    """从 LLM 响应中提取 response 字段"""
    json_str = content.strip()
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
    if code_block:
        json_str = code_block.group(1).strip()
    brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            response_text = data.get("response") or data.get("report") or data.get("content")
            if response_text:
                return str(response_text)
    except json.JSONDecodeError:
        pass
    return ""


def _extract_markdown_response(content: str) -> str:
    """从 LLM 响应中提取 Markdown 内容（去掉 JSON 包装）"""
    lines = content.strip().split("\n")
    # 移除最外层 ``` 标记
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _build_fallback_report(input_text: str, past_steps: list) -> str:
    steps = _format_simple_steps(past_steps)
    return f"""# 任务执行结果

## 原始任务
{input_text}

## 执行的步骤
{steps}

## 说明
由于系统异常，无法生成完整响应。以上是已收集的信息。"""


def _format_simple_steps(past_steps: list) -> str:
    if not past_steps:
        return "无"
    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {preview}\n")
    return "\n".join(formatted)


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {len(past_steps)}")

    # 使用 AIOpsService 注入的工具列表和 LLM（模块级全局变量）
    if not _llm:
        logger.error("Replanner 缺少 LLM 实例")
        return await _generate_response(state)

    # 强制限制
    MAX_STEPS = 8
    MIN_STEPS_BEFORE_RESPOND = 2
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"已执行 {len(past_steps)} 个步骤，超过最大限制，强制生成最终响应")
        return await _generate_response(state)

    # 如果还有剩余计划，进行决策

    # 格式化已执行步骤
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:300]}..."
        for step, result in past_steps
    ])

    # 如果还有剩余计划，进行决策
    if plan:
        try:
            messages = [
                ("user", f"原始任务: {input_text}"),
                ("user", f"已执行的步骤:\n{steps_summary}"),
                ("user", f"剩余计划: {', '.join(plan)}"),
                ("user", f"提示：已执行 {len(past_steps)} 个步骤，还有 {len(plan)} 个步骤待执行。请继续执行剩余步骤，除非已获取足够数据且已执行 >= 3 步。"),
            ]

            replanner_chain = replanner_prompt | _llm
            llm_response = await replanner_chain.ainvoke({
                "messages": messages,
                "tools_description": _tools_description,
            })

            action, new_steps = _parse_act_response(llm_response)
            logger.info(f"Replanner 决策: {action}")

            if action == "respond" and len(past_steps) < MIN_STEPS_BEFORE_RESPOND:
                logger.info(f"已执行 {len(past_steps)} 步 < {MIN_STEPS_BEFORE_RESPOND}，强制继续执行")
                action = "continue"

            if action == "respond":
                return await _generate_response(state)
            elif action == "replan":
                if len(new_steps) > len(plan):
                    logger.warning(f"新步骤数超出限制，截断")
                    new_steps = new_steps[:len(plan)]
                if len(past_steps) >= 5:
                    logger.warning(f"已执行 {len(past_steps)} 个步骤，禁止重新规划")
                    return await _generate_response(state)
                logger.info(f"决定调整计划，新步骤数量: {len(new_steps)}")
                return {"plan": new_steps} if new_steps else {}
            else:  # continue
                logger.info("决定继续执行当前计划")
                return {}
        except Exception as e:
            logger.error(f"重新规划失败: {e}")
            return {}
    else:
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state)
