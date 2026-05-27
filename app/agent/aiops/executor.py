"""
Executor 节点：执行单个步骤

直接调用工具（绕过 LangGraph ToolNode，避免 Runtime 依赖）
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from .state import PlanExecuteState

# 由 AIOpsService 在运行前注入（避免放入 state 导致序列化问题）
_tools: list = []
_llm = None

# 外部状态累加器：LangGraph 多轮迭代后可能丢失状态，以此处为准
# 各节点通过 sys.modules["app.agent.aiops.executor"]._accumulated_state 访问
_accumulated_state: dict = {}


def _update_accumulator(result: dict) -> None:
    """更新累加器，正确处理 past_steps 的追加（模拟 operator.add）"""
    # past_steps 使用 operator.add 语义（追加而非覆盖）
    if "past_steps" in result:
        existing = _accumulated_state.get("past_steps", [])
        new_steps = result["past_steps"]
        if new_steps:
            if not existing or new_steps[0] != existing[-1]:
                result["past_steps"] = existing + new_steps
    _accumulated_state.update(result)


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """执行节点：执行计划中的下一个步骤"""
    logger.info("=== Executor：执行步骤 ===")

    # 防御性处理：state 可能为 None
    if state is None:
        logger.warning("Executor received None state, restoring from accumulator")
        state = dict(_accumulated_state)  # type: ignore
    elif _accumulated_state:
        acc = _accumulated_state
        existing = state.get("past_steps", [])
        acc_steps = acc.get("past_steps", [])
        if len(acc_steps) > len(existing):
            state["past_steps"] = list(acc_steps)  # type: ignore

    plan = state.get("plan", [])
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    task = plan[0]
    logger.info(f"当前任务: {task}")

    # 统一检查依赖
    if not _tools or not _llm:
        error = "缺少工具列表" if not _tools else "缺少 LLM 实例"
        logger.error(f"Executor {error}")
        return {"plan": plan[1:], "past_steps": [(task, f"执行失败: {error}")]}

    try:
        # 构建工具提示（简化）
        tools_str = "\n".join(
            f"  - {t.name}: {t.description}" 
            for t in _tools 
            if hasattr(t, 'name') and hasattr(t, 'description')
        )
        
        # 获取前序步骤（简化）
        past_steps = state.get("past_steps", [])
        past_context = "\n".join(
            f"  步骤{i}: {step}\n  结果: {result[:500]}"
            for i, (step, result) in enumerate(past_steps, 1)
        ) if past_steps else ""
        
        step_number = len(past_steps) + 1
        
        # 构建消息
        messages = [
            SystemMessage(content=f"""你是一个工具调用专家，负责精确执行指定的步骤。

可用工具列表：
{tools_str}

工具选择指南：
- 查询当前时间 → 使用 get_current_time
- 搜索/查询应用日志 → 使用 search_log（需要 start_time 和 end_time 参数，单位：毫秒时间戳）
- 查询CPU使用率指标 → 使用 query_cpu_metrics
- 查询内存使用率指标 → 使用 query_memory_metrics
- 查询知识库/运维文档 → 使用 retrieve_knowledge

执行规则：
1. 仔细分析任务目标，选择合适的工具
2. 利用前序步骤的结果作为参数
3. 调用工具获取真实数据，不要编造
4. 将工具返回的结果整理后输出

禁止行为：
- 不要编造数据
- 不要重复调用同一个工具
- 如果工具调用失败，说明失败原因"""),
        ]
        
        if past_context:
            messages.append(HumanMessage(
                content=f"## 前序步骤信息（步骤 #{step_number}）\n{past_context}"
            ))
        
        messages.append(HumanMessage(
            content=f"请执行任务（步骤 #{step_number}）: {task}"
        ))

        # LLM 决定是否调用工具
        llm_with_tools = _llm.bind_tools(_tools)
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 执行工具调用
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")
            
            # 构建工具查找表
            tools_by_name = {
                t.name: t 
                for t in _tools 
                if hasattr(t, 'name')
            }
            
            # 执行所有工具调用
            tool_messages = []
            for tool_call in llm_response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", "")
                
                tool_fn = tools_by_name.get(tool_name)
                if not tool_fn:
                    logger.warning(f"未找到工具: {tool_name}")
                    tool_messages.append(ToolMessage(
                        content=f"错误: 未找到工具 '{tool_name}'",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                    ))
                    continue
                
                try:
                    logger.info(f"执行工具: {tool_name}({tool_args})")
                    result = await tool_fn.ainvoke(tool_args)
                    result_str = str(result) if not isinstance(result, str) else result
                    tool_messages.append(ToolMessage(
                        content=result_str,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    ))
                    logger.info(f"工具 {tool_name} 执行成功")
                except Exception as e:
                    logger.warning(f"工具 {tool_name} 执行失败: {e}", exc_info=True)
                    tool_messages.append(ToolMessage(
                        content=f"工具调用失败: {type(e).__name__}: {e}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                    ))
            
            # 获取最终结果
            messages.append(llm_response)
            messages.extend(tool_messages)
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        ret = {
            "plan": plan[1:],
            "past_steps": [(task, result)],
        }
        _update_accumulator(ret)
        return ret

    except Exception as e:
        logger.error(f"执行步骤失败: {type(e).__name__}: {e}", exc_info=True)
        ret = {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {type(e).__name__}")],
        }
        _update_accumulator(ret)
        return ret