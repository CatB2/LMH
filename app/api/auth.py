"""认证接口 - 注册、登录、刷新 Token、登出"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.request import RegisterRequest, LoginRequest, RefreshTokenRequest, LogoutRequest
from app.services.auth_service import (
    register_user,
    login_user,
    refresh_access_token,
    verify_refresh_token,
    user_store,
)
from app.utils.response import success, error

router = APIRouter()


@router.post("/auth/register")
async def register(request: RegisterRequest):
    """用户注册"""
    try:
        result = register_user(request.username, request.password)
        logger.info(f"用户注册成功: {request.username}")
        return success(
            data=result,
            message="注册成功",
        )
    except ValueError as e:
        return error(message=str(e), code=409)
    except Exception as e:
        logger.error(f"注册失败: {e}")
        return error(message="注册失败", code=500)


@router.post("/auth/login")
async def login(request: LoginRequest):
    """用户登录"""
    try:
        result = login_user(request.username, request.password)
        logger.info(f"用户登录成功: {request.username}")
        return success(
            data=result,
            message="登录成功",
        )
    except ValueError as e:
        return error(message=str(e), code=401)
    except Exception as e:
        logger.error(f"登录失败: {e}")
        return error(message="登录失败", code=500)


@router.post("/auth/refresh")
async def refresh(request: RefreshTokenRequest):
    """刷新 Access Token"""
    try:
        result = refresh_access_token(request.refresh_token)
        return success(data=result, message="Token 刷新成功")
    except ValueError as e:
        return error(message=str(e), code=401)


@router.post("/auth/logout")
async def logout(request: LogoutRequest):
    """登出 - 吊销 Refresh Token"""
    payload = verify_refresh_token(request.refresh_token)
    if payload and payload.get("jti"):
        user_store.revoke_token(payload["jti"])
        logger.info(f"用户 {payload['sub']} 登出")
    return success(message="登出成功")
