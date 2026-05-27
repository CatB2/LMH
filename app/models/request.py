"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    question: str = Field(..., description="用户问题")
    session_id: str | None = Field(None, description="会话 ID，不传则服务端创建新会话")
    agent: str | None = Field(None, description="指定 Agent 类型，如 rag、aiops，不传则自动识别意图")
    model: str | None = Field(None, description="指定模型，如 qwen-max / deepseek-r1，不传用默认")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "question": "什么是向量数据库？",
                "session_id": "session-123",
                "agent": "rag"
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True


class AuthRequest(BaseModel):
    """认证基类"""
    username: str = Field(..., min_length=2, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码")


class RegisterRequest(AuthRequest):
    """注册请求"""
    pass


class LoginRequest(AuthRequest):
    """登录请求"""
    pass


class RefreshTokenRequest(BaseModel):
    """刷新 Token 请求"""
    refresh_token: str = Field(..., description="Refresh Token")


class LogoutRequest(BaseModel):
    """登出请求"""
    refresh_token: str = Field(..., description="Refresh Token")
