"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现
"""

from typing import AsyncGenerator, Dict, Any, List
from loguru import logger

from app.config import config
from app.core.llm_factory import llm_factory
from app.agent.aiops import PlanExecuteState, planner, executor, replanner
import sys
from app.agent.aiops.utils import format_tools_description
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        """初始化服务"""
        self.model_name = config.rag_model
        self.llm = llm_factory.create_chat_model(
            model=self.model_name,
            temperature=0,
            streaming=False,
        )

        # 工具列表（延迟加载，仅一次）
        self._tools: List = []
        self._tools_description: str = ""
        self._tools_initialized = False

        logger.info("Plan-Execute-Replan Service 初始化完成")

    async def _ensure_model(self, model: str = ""):
        use_model = model or config.rag_model
        if use_model != self.model_name:
            self.llm = llm_factory.create_chat_model(model=use_model, temperature=0, streaming=False)
            self.model_name = use_model
            logger.info(f"AIOps 切换模型: {use_model}")

    async def _ensure_tools(self):
        """集中初始化工具列表（仅一次）"""
        if self._tools_initialized:
            return

        local_tools = [get_current_time, retrieve_knowledge]
        mcp_tools = []

        # 逐个服务器加载 MCP 工具，避免并发 TaskGroup 超时
        import asyncio
        try:
            mcp_client = await get_mcp_client_with_retry()
            if hasattr(mcp_client, 'connections'):
                for server_name in list(mcp_client.connections.keys()):
                    try:
                        server_tools = await mcp_client.get_tools(server_name=server_name)
                        mcp_tools.extend(server_tools)
                        logger.info(f"AIOps 加载 MCP 服务器 '{server_name}' 工具: {len(server_tools)} 个")
                    except Exception as e:
                        logger.warning(f"AIOps MCP 服务器 '{server_name}' 加载失败: {e}")
            else:
                # fallback: 一次性加载所有工具
                mcp_tools = await mcp_client.get_tools()

            if mcp_tools:
                logger.info(f"AIOps 成功加载 {len(mcp_tools)} 个 MCP 工具")
        except Exception as e:
            logger.warning(f"AIOps MCP 工具加载失败，仅使用本地工具: {e}")

        self._tools = local_tools + mcp_tools
        self._tools_description = format_tools_description(self._tools)
        self._tools_initialized = True

        if self._tools:
            tool_names = [t.name if hasattr(t, "name") else str(t) for t in self._tools]
            logger.info(f"AIOps 可用工具: {', '.join(tool_names)}")

    async def _inject_and_run(
        self, node_fn, state: PlanExecuteState
    ) -> Dict[str, Any]:
        """注入 tools/llm 后运行指定节点函数"""
        if node_fn is planner:
            pm = sys.modules["app.agent.aiops.planner"]
            pm._tools = self._tools
            pm._tools_description = self._tools_description
            pm._llm = self.llm
        elif node_fn is executor:
            em = sys.modules["app.agent.aiops.executor"]
            em._tools = self._tools
            em._llm = self.llm
        elif node_fn is replanner:
            rm = sys.modules["app.agent.aiops.replanner"]
            rm._tools = self._tools
            rm._tools_description = self._tools_description
            rm._llm = self.llm
        return await node_fn(state)

    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """执行 Plan-Execute-Replan 流程（手动循环，避免 LangGraph 多轮 state 问题）"""
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            # 预初始化工具列表（仅首次执行时加载）
            await self._ensure_tools()

            # 初始化状态
            state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": "",
            }

            final_response = ""

            # Step 1: Plan
            plan_output = await self._inject_and_run(planner, state)
            state.update(plan_output)
            yield self._format_planner_event(state)
            logger.info(f"[会话 {session_id}] 计划制定完成，共 {len(state.get('plan', []))} 个步骤")

            # Step 2-4: Execute-Replan 循环
            max_rounds = 8
            for round_idx in range(max_rounds):
                plan = state.get("plan", [])
                if not plan:
                    logger.info(f"[会话 {session_id}] 计划已执行完毕")
                    break

                # Execute next step
                exec_output = await self._inject_and_run(executor, state)
                state.update(exec_output)
                yield self._format_executor_event(state)

                # Replan
                replan_output = await self._inject_and_run(replanner, state)
                state.update(replan_output)

                # Check if we have a final response
                if state.get("response"):
                    final_response = state["response"]
                    yield self._format_replanner_event(state)
                    logger.info(f"[会话 {session_id}] 最终响应已生成")
                    break

                yield self._format_replanner_event(state)
                remaining = len(state.get("plan", []))
                logger.info(f"[会话 {session_id}] 剩余 {remaining} 个步骤")

            # 如果循环结束但没有 response，生成一个
            if not final_response:
                fallback_prompt = (
                    f"基于以下信息生成诊断报告：\n\n"
                    f"原始任务: {user_input}\n\n"
                    f"执行历史:\n"
                )
                past_steps = state.get("past_steps", [])
                for i, (step, result) in enumerate(past_steps, 1):
                    fallback_prompt += f"\n### 步骤 {i}: {step}\n{result[:500]}\n"

                from app.agent.aiops.replanner import _build_fallback_report
                fallback = _build_fallback_report(user_input, past_steps)
                final_response = fallback

            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response,
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {type(e).__name__}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}",
            }

    async def query(
        self,
        question: str,
        session_id: str = "default",
        user_id: str = "",
        model: str = "",
    ) -> str:
        """AIOps 查询接口（非流式，供 AgentManager 调用）

        收集 execute() 生成器的最终 response 返回纯文本
        """
        await self._ensure_model(model)
        final_response = ""
        async for event in self.execute(question, session_id):
            if event.get("type") == "complete":
                final_response = event.get("response", "")
            elif event.get("type") == "error":
                logger.error(f"AIOps 查询失败: {event.get('message', '')}")
                return f"诊断出错: {event.get('message', '')}"
        return final_response or "诊断完成，未生成报告"

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        user_id: str = "",
        model: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """AIOps 流式查询接口（供 AgentManager 调用）"""
        await self._ensure_model(model)
        async for event in self.execute(question, session_id):
            event_type = event.get("type", "")
            if event_type == "plan":
                yield {"type": "debug", "node": "planner", "message_type": "plan"}
            elif event_type == "step_complete":
                yield {"type": "debug", "node": "executor", "message_type": "step"}
            elif event_type == "report":
                yield {"type": "content", "data": event.get("report", "")}
            elif event_type == "complete":
                response = event.get("response", "")
                yield {
                    "type": "complete",
                    "data": {"content": response},
                }
            elif event_type == "error":
                yield {"type": "error", "data": event.get("message", "")}

    async def diagnose(
        self,
        session_id: str = "default",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """AIOps 诊断接口（兼容旧接口）"""
        from textwrap import dedent

        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```markdown
                # 告警分析报告
                ## 活跃告警清单
                ## 告警根因分析
                ## 处理方案
                ## 结论
                ```
                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本
                - 所有内容必须基于工具查询的真实数据，严禁编造""")

        async for event in self.execute(aiops_task, session_id):
            if event.get("type") == "complete":
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {
                        "status": "completed",
                        "report": event.get("response", ""),
                    },
                }
            else:
                yield event

    def _format_planner_event(self, state: Dict | None) -> Dict:
        if not state:
            return {"type": "status", "stage": "planner", "message": "规划节点执行中"}
        plan = state.get("plan", [])
        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan,
        }

    def _format_executor_event(self, state: Dict | None) -> Dict:
        if not state:
            return {"type": "status", "stage": "executor", "message": "执行节点运行中"}
        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])
        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan),
            }
        else:
            return {"type": "status", "stage": "executor", "message": "开始执行步骤"}

    def _format_replanner_event(self, state: Dict | None) -> Dict:
        if not state:
            return {"type": "status", "stage": "replanner", "message": "评估节点运行中"}
        response = state.get("response", "")
        plan = state.get("plan", [])
        if response:
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response,
            }
        else:
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan),
            }


# 全局单例
aiops_service = AIOpsService()
