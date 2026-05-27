"""对话接口

提供基于 RAG Agent 的普通对话和流式对话接口
"""

import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.models.request import ChatRequest, ClearRequest
from app.models.response import SessionInfoResponse
from app.agent.agent_manager import agent_manager
from app.services.rag_agent_service import rag_agent_service
from app.services.auth_service import get_current_user
from app.utils.response import success, error
from app.tools.knowledge_tool import latest_source_files as rag_source_files
from loguru import logger

router = APIRouter()


def _resolve_session(session_id: str | None, user_id: str) -> str:
    """如果未传 session_id，创建新 session 并绑定 user"""
    if session_id:
        return session_id
    new_id = str(uuid.uuid4())
    logger.info(f"创建新会话: {new_id} (user={user_id})")
    return new_id


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """快速对话接口"""
    try:
        session_id = _resolve_session(request.session_id, user["user_id"])

        # 检查 session 所属权
        owner = await rag_agent_service.memory_store.get_session_owner(session_id)
        if owner and owner != user["user_id"]:
            return error(message="无权访问此会话", code=403)

        # 如果 session 不存在，创建
        if not owner:
            await rag_agent_service.memory_store.create_session(user["user_id"], session_id)

        logger.info(f"[会话 {session_id}] 收到快速对话请求: {request.question}, agent={request.agent}")

        answer = await agent_manager.query(
            request.question,
            session_id=session_id,
            agent_name=request.agent,
            user_id=user["user_id"],
            model=request.model or "",
        )

        logger.info(f"[会话 {session_id}] 快速对话完成")

        source_files = list(rag_source_files)
        rag_source_files.clear()

        return success(data={
            "success": True,
            "answer": answer,
            "session_id": session_id,
            "errorMessage": None,
            "source_files": source_files,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话接口错误: {e}")
        sid = None
        try:
            sid = session_id  # noqa: F821
        except NameError:
            pass
        return error(
            message=str(e),
            code=500,
            data={
                "success": False,
                "answer": None,
                "session_id": sid,
                "errorMessage": str(e),
            },
        )


@router.post("/chat_stream")
async def chat_stream(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """流式对话接口（SSE）"""
    session_id = _resolve_session(request.session_id, user["user_id"])

    # 检查 session 所属权
    owner = await rag_agent_service.memory_store.get_session_owner(session_id)
    if owner and owner != user["user_id"]:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    if not owner:
        await rag_agent_service.memory_store.create_session(user["user_id"], session_id)

    logger.info(f"[会话 {session_id}] 收到流式对话请求: {request.question}, agent={request.agent}")

    async def event_generator():
        try:
            async for chunk in agent_manager.query_stream(
                request.question,
                session_id=session_id,
                agent_name=request.agent,
                user_id=user["user_id"],
                model=request.model or "",
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                if chunk_type == "debug":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "debug",
                            "node": chunk.get("node", "unknown"),
                            "message_type": chunk.get("message_type", "unknown"),
                        }, ensure_ascii=False),
                    }
                elif chunk_type == "tool_call":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "tool_call",
                            "data": chunk_data,
                        }, ensure_ascii=False),
                    }
                elif chunk_type == "search_results":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "search_results",
                            "data": chunk_data,
                        }, ensure_ascii=False),
                    }
                elif chunk_type == "content":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "content",
                            "data": chunk_data,
                        }, ensure_ascii=False),
                    }
                elif chunk_type == "complete":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "done",
                            "data": {**chunk_data, "session_id": session_id} if chunk_data else {"session_id": session_id},
                        }, ensure_ascii=False),
                    }
                elif chunk_type == "error":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "error",
                            "data": str(chunk_data),
                        }, ensure_ascii=False),
                    }

            logger.info(f"[会话 {session_id}] 流式对话完成")

        except Exception as e:
            logger.error(f"流式对话接口错误: {e}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "data": str(e),
                }, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/clear")
async def clear_session(
    request: ClearRequest,
    user: dict = Depends(get_current_user),
):
    """清空会话历史"""
    try:
        owner = await rag_agent_service.memory_store.get_session_owner(request.session_id)
        if owner and owner != user["user_id"]:
            return error(message="无权访问此会话", code=403)

        success_flag = await rag_agent_service.clear_session(request.session_id)
        logger.info(f"清空会话: {request.session_id}, 结果: {success_flag}")

        if success_flag:
            return success(message="会话已清空")
        else:
            return error(message="清空会话失败", code=500)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空会话错误: {e}")
        return error(message=str(e), code=500)


@router.get("/chat/session/{session_id}")
async def get_session_info(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """查询会话历史"""
    try:
        owner = await rag_agent_service.memory_store.get_session_owner(session_id)
        if owner and owner != user["user_id"]:
            return error(message="无权访问此会话", code=403)

        history = await rag_agent_service.get_session_history(session_id)

        return success(data={
            "session_id": session_id,
            "message_count": len(history),
            "history": history,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话信息错误: {e}")
        return error(message=str(e), code=500)


@router.get("/chat/agents")
async def list_agents(user: dict = Depends(get_current_user)):
    """获取当前可用 Agent 列表"""
    return success(data=agent_manager.list_agents())


@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    """列出当前用户的所有会话"""
    try:
        sessions = await rag_agent_service.memory_store.get_user_sessions(user["user_id"])
        return success(data={
            "sessions": sessions,
            "total": len(sessions),
        })
    except Exception as e:
        logger.error(f"获取会话列表错误: {e}")
        return error(message=str(e), code=500)
