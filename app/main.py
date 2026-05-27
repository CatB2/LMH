"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

import sys
sys.dont_write_bytecode = True

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops, auth
from app.core.milvus_client import milvus_manager
from app.core.user_memory import decay_memories


async def _warm_bm25():
    """预构建 BM25 索引"""
    try:
        from app.tools.knowledge_tool import _ensure_bm25_index
        await asyncio.to_thread(_ensure_bm25_index)
    except Exception as e:
        logger.warning(f"BM25 预热失败: {e}")


# 记忆衰减任务：每 24 小时清理过期长期记忆
_memory_decay_task: asyncio.Task | None = None
_flush_task: asyncio.Task | None = None


async def _run_flush():
    """每 5 秒执行 batch_flush，增量同步 Redis → SQLite"""
    while True:
        try:
            await asyncio.sleep(5)
            from app.services.rag_agent_service import rag_agent_service
            await rag_agent_service.memory_store.batch_flush()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"batch_flush 异常: {e}")


async def _run_memory_decay():
    while True:
        try:
            await asyncio.sleep(86400)  # 24 小时
            decay_memories()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"记忆衰减任务异常: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"{config.app_name} v{config.app_version} 启动中...")
    logger.info(f"环境: {'开发' if config.debug else '生产'}")
    logger.info(f"监听地址: http://{config.host}:{config.port}")
    logger.info(f"API 文档: http://{config.host}:{config.port}/docs")

    # Milvus 采用惰性连接，VectorStoreManager 在首次使用时自动初始化

    # 预构建 BM25 索引（避免首次检索时卡顿）
    asyncio.create_task(_warm_bm25())

    # 启动 batch_flush（每 5 秒增量回写 SQLite）
    global _flush_task
    _flush_task = asyncio.create_task(_run_flush())

    # 启动记忆衰减后台任务
    global _memory_decay_task
    _memory_decay_task = asyncio.create_task(_run_memory_decay())
    logger.info("长期记忆衰减任务已启动（每 24 小时执行一次）")

    logger.info("=" * 60)

    yield

    # 关闭时执行
    logger.info("正在关闭服务...")
    if _flush_task and not _flush_task.done():
        _flush_task.cancel()
    if _memory_decay_task and not _memory_decay_task.done():
        _memory_decay_task.cancel()
    try:
        milvus_manager.close()
    except Exception:
        pass
    logger.info(f"{config.app_name} 已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(auth.router, prefix="/api", tags=["认证"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
