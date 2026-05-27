"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # LLM 配置
    dashscope_api_key: str = ""  # API Key（DashScope / vLLM / Ollama 共用此字段）
    dashscope_model: str = "qwen-max"  # 模型名
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # OpenAI 兼容 API 地址
    dashscope_embedding_api_key: str = "sk-bdbfccec7c3a4ad58e03bddbddbe8188"  # 嵌入模型 API Key（如果与 LLM 不同，如 VLLM 场景）
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 3000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考
    rag_context_token_limit: int = 3000
    rag_context_recent_token_keep: int = 1200
    rag_context_summary_threshold: int = 2200
    rag_context_summary_batch: int = 1000

    # Redis 缓存配置
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_cache_ttl_days: int = 7

    # 会话历史限制
    session_history_message_limit: int = 100
    session_history_token_limit: int = 50000

    # JWT 鉴权配置
    jwt_secret_key: str = "super-biz-agent-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 腾讯云 CLS 配置
    cls_secret_id: str = ""  # 腾讯云 Secret ID
    cls_secret_key: str = ""  # 腾讯云 Secret Key
    cls_region: str = "ap-beijing"  # CLS 区域
    cls_topic_id: str = ""  # 日志主题 ID
    cls_topic_database: str = ""  # 数据库日志topic_id
    cls_topic_api: str = ""       # API日志topic_id
    cls_topic_sync: str = ""       # 同步日志topic_id
    cls_topic_default: str = ""    # 普通日志topic_id

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
