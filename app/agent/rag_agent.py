"""Agent 初始化模块 - 系统提示词 + MCP 工具加载"""

from langchain.agents import create_agent
from loguru import logger


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的智能业务代理（SuperBizAgent），负责处理企业级业务问题。

## 核心能力

### 1. 知识检索（retrieve_knowledge）【最重要】
当用户提到以下任何情况时，**必须**调用 `retrieve_knowledge` 工具：
- "上传的文档"、"知识库"、"文档里说"、"参考文档"
- 任何需要查阅内部资料、技术文档、操作手册的问题
- 用户问"有哪些论文"、"论文讲了什么"等涉及知识库的问题
- 用户要求"根据资料回答"、"查一下文档"、"搜索一下"
- 你不确定答案是否正确，需要找依据

### 2. 时间查询
- 获取当前时间和日期

### 3. AIOps 诊断
- 通过 MCP 工具进行系统监控和日志分析

## 工作原则
1. **优先使用工具**：当问题涉及知识库内容时，先检索再回答，不要凭记忆
2. 检索到相关信息后，基于检索结果给出准确回答，并注明信息来源
3. 如果检索不到相关信息，诚实地告知用户未找到
4. 不要编造信息，所有回答都应有依据

## 回答要求
- 保持友好、专业的语气
- 回答简洁明了，重点突出
- 引用文档时注明来源文件名"""


async def create_rag_agent(model, tools):
    """创建并初始化 RAG Agent（包括连接 MCP 加载工具）

    Args:
        model: ChatOpenAI 实例
        tools: 本地工具列表

    Returns:
        tuple: (agent, all_tools)
    """
    # 尝试加载 MCP 工具
    try:
        from app.agent.mcp_client import get_mcp_client_with_retry
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")
    except Exception as e:
        logger.warning(f"MCP 服务器连接失败，仅使用本地工具: {e}")
        mcp_tools = []

    all_tools = tools + mcp_tools

    # 预先绑定 tool_choice='auto'，增加模型主动调工具的可靠性
    model = model.bind_tools(all_tools, tool_choice="auto")

    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
    )

    if all_tools:
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in all_tools]
        logger.info(f"可用工具列表: {', '.join(tool_names)}")

    return agent, all_tools
