"""用户认证服务 - JWT 双 Token + bcrypt 密码"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import bcrypt as _bcrypt
from loguru import logger

from app.config import config


class UserStore:
    """用户存储（内存 + 延迟写 SQLite）"""

    def __init__(self):
        self._conn = None

    def _ensure_conn(self):
        if self._conn is None:
            import sqlite3
            from pathlib import Path

            db_path = Path("data/session_memory.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._ensure_tables()

    def _ensure_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                token_jti TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                revoked INTEGER DEFAULT 0,
                revoked_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_refresh_tokens_jti
                ON refresh_tokens(token_jti);
            CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id
                ON refresh_tokens(user_id);
        """)
        self._conn.commit()

    # ── 用户操作 ──

    def create_user(self, username: str, password_hash: str) -> str:
        self._ensure_conn()
        user_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO users (user_id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, username, password_hash, created_at),
            )
            self._conn.commit()
            logger.info(f"用户创建成功: {username} ({user_id})")
            return user_id
        except Exception as e:
            logger.error(f"用户创建失败: {e}")
            raise

    def get_user_by_username(self, username: str) -> Optional[dict]:
        self._ensure_conn()
        cursor = self._conn.execute(
            "SELECT user_id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        self._ensure_conn()
        cursor = self._conn.execute(
            "SELECT user_id, username, created_at FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ── Refresh Token 操作 ──

    def save_refresh_token(self, user_id: str, jti: str, expires_at: str) -> None:
        self._ensure_conn()
        created_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO refresh_tokens (user_id, token_jti, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (user_id, jti, expires_at, created_at),
        )
        self._conn.commit()

    def is_token_revoked(self, jti: str) -> bool:
        self._ensure_conn()
        cursor = self._conn.execute(
            "SELECT revoked FROM refresh_tokens WHERE token_jti = ?",
            (jti,),
        )
        row = cursor.fetchone()
        return bool(row and row["revoked"])

    def revoke_token(self, jti: str) -> None:
        self._ensure_conn()
        revoked_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE refresh_tokens SET revoked = 1, revoked_at = ? WHERE token_jti = ?",
            (revoked_at, jti),
        )
        self._conn.commit()

    def revoke_all_user_tokens(self, user_id: str) -> None:
        self._ensure_conn()
        revoked_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE refresh_tokens SET revoked = 1, revoked_at = ? WHERE user_id = ? AND revoked = 0",
            (revoked_at, user_id),
        )
        self._conn.commit()


# 全局用户存储单例
user_store = UserStore()


def hash_password(password: str) -> str:
    """bcrypt 哈希密码"""
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return _bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(user_id: str, username: str) -> str:
    """创建 Access Token（15 分钟有效期）"""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=config.jwt_access_token_expire_minutes
    )
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)


def create_refresh_token(user_id: str) -> tuple[str, str, str]:
    """创建 Refresh Token（7 天有效期）并持久化

    Returns:
        tuple: (token_str, jti, expires_at_iso)
    """
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + timedelta(
        days=config.jwt_refresh_token_expire_days
    )
    expires_at = expire.isoformat()
    payload = {
        "sub": user_id,
        "jti": jti,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
    }
    token = jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)
    user_store.save_refresh_token(user_id, jti, expires_at)
    return token, jti, expires_at


def decode_token(token: str) -> Optional[dict]:
    """解码 JWT Token，失败返回 None"""
    try:
        return jwt.decode(
            token, config.jwt_secret_key, algorithms=[config.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def verify_refresh_token(token_str: str) -> Optional[dict]:
    """验证 Refresh Token：签名 + 未吊销

    Returns:
        dict: payload（含 sub/user_id, jti）或 None
    """
    payload = decode_token(token_str)
    if not payload:
        return None
    if payload.get("type") != "refresh":
        return None
    jti = payload.get("jti")
    if not jti or user_store.is_token_revoked(jti):
        return None
    return payload


def register_user(username: str, password: str) -> dict:
    """注册用户并返回 Token"""
    existing = user_store.get_user_by_username(username)
    if existing:
        raise ValueError(f"用户名 '{username}' 已存在")

    password_hash = hash_password(password)
    user_id = user_store.create_user(username, password_hash)

    access_token = create_access_token(user_id, username)
    refresh_token, _, expires_at = create_refresh_token(user_id)

    return {
        "user_id": user_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": config.jwt_access_token_expire_minutes * 60,
    }


def login_user(username: str, password: str) -> dict:
    """登录并返回 Token"""
    user = user_store.get_user_by_username(username)
    if not user:
        raise ValueError("用户名或密码错误")

    if not verify_password(password, user["password_hash"]):
        raise ValueError("用户名或密码错误")

    access_token = create_access_token(user["user_id"], username)
    refresh_token, _, expires_at = create_refresh_token(user["user_id"])

    return {
        "user_id": user["user_id"],
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": config.jwt_access_token_expire_minutes * 60,
    }


def refresh_access_token(refresh_token_str: str) -> dict:
    """用 Refresh Token 换取新的 Access Token"""
    payload = verify_refresh_token(refresh_token_str)
    if not payload:
        raise ValueError("Refresh Token 无效或已吊销")

    user_id = payload["sub"]
    user = user_store.get_user_by_id(user_id)
    if not user:
        raise ValueError("用户不存在")

    access_token = create_access_token(user_id, user["username"])
    return {
        "access_token": access_token,
        "expires_in": config.jwt_access_token_expire_minutes * 60,
    }


from fastapi import Header, HTTPException


def get_current_user(authorization: str = Header(...)) -> dict:
    """FastAPI 依赖：从 Authorization Header 解析当前用户

    Usage:
        @router.post("/chat")
        async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
            ...
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    return {
        "user_id": payload["sub"],
        "username": payload.get("username", ""),
    }
