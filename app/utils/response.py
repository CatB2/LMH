"""统一 API 响应工具"""

from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse


def success(
    data: Any = None,
    message: str = "success",
    code: int = 200,
) -> JSONResponse:
    """成功响应"""
    return JSONResponse(
        status_code=code,
        content={
            "code": code,
            "message": message,
            "data": data,
        },
    )


def error(
    message: str = "error",
    code: int = 400,
    data: Any = None,
) -> JSONResponse:
    """错误响应"""
    return JSONResponse(
        status_code=code,
        content={
            "code": code,
            "message": message,
            "data": data,
        },
    )
