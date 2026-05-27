"""健康检查接口"""

from typing import Any
from fastapi import APIRouter
from app.config import config
from app.core.milvus_client import milvus_manager
from app.utils.response import success, error
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    
    """健康检查接口
    检查服务状态和数据库连接状态
    
    Returns:
        JSONResponse: 健康检查结果
    """
    # 检查服务基本状态
    health_data: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy"
    }
    
    # 检查 Milvus 连接状态
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status: str = "connected" if milvus_healthy else "disconnected"
        milvus_message: str = "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }
    
    # 判断整体健康状态
    overall_status = "healthy"

    # 如果 Milvus 不可用，服务不可用
    if health_data["milvus"]["status"] != "connected":
        overall_status = "unhealthy"
        health_data["error"] = "数据库不可用"

    health_data["status"] = overall_status

    if overall_status == "healthy":
        return success(data=health_data, message="服务运行正常")
    else:
        return error(data=health_data, message="服务不可用", code=503)
