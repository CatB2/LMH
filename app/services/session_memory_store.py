"""会话存储：Redis 主存 + SQLite 冷备"""

import json
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import redis.asyncio as aioredis
from app.config import config


class SessionMemoryStore:

    def __init__(self, db_path=None):
        db_path = db_path or Path("data/session_memory.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.redis = aioredis.Redis(host=config.redis_host, port=config.redis_port, db=config.redis_db, decode_responses=True)
        self._ensure_schema()

    # ── Redis key ──

    def _mk(self, sid): return f"session:{sid}:messages"
    def _sk(self, sid): return f"session:{sid}:summary"
    def _ck(self, sid): return f"session:{sid}:compressed"
    def _fk(self, sid): return f"session:{sid}:flushed"

    # ── 消息 ──

    async def append_message(self, session_id, role, content, user_id="", timestamp=None, token_count=None):
        msg = json.dumps({"role": role, "content": content, "timestamp": timestamp or datetime.utcnow().isoformat(), "token_count": token_count or max(1, len(content.strip()) // 4), "user_id": user_id})
        await self.redis.rpush(self._mk(session_id), msg)
        ttl = 604800
        for f in (self._mk, self._sk, self._ck, self._fk):
            await self.redis.expire(f(session_id), ttl)

    async def get_history(self, session_id):
        await self._init_from_sqlite(session_id)
        raw = await self.redis.lrange(self._mk(session_id), 0, -1)
        return [json.loads(r) for r in raw] if raw else []

    async def get_history_stats(self, session_id):
        total = await self.redis.llen(self._mk(session_id))
        return {"message_count": total, "total_tokens": total * 10}

    # ── 摘要 + 压缩指针 ──

    async def get_summary(self, session_id):
        return await self.redis.get(self._sk(session_id)) or ""
    async def set_summary(self, session_id, text):
        await self.redis.set(self._sk(session_id), text)
    async def get_compressed(self, session_id):
        return int(await self.redis.get(self._ck(session_id)) or 0)
    async def set_compressed(self, session_id, val):
        await self.redis.set(self._ck(session_id), val)

    # ── 会话元数据 ──

    async def create_session(self, user_id, session_id):
        await self.redis.sadd(f"user:{user_id}:sessions", session_id)
        now = datetime.utcnow().isoformat()
        self.conn.execute("INSERT OR IGNORE INTO session_metadata (session_id, user_id, created_at, last_active) VALUES (?, ?, ?, ?)", (session_id, user_id, now, now))
        # 旧表迁移（生产环境兼容）
        try:
            self.conn.execute('ALTER TABLE session_summaries ADD COLUMN compressed INTEGER NOT NULL DEFAULT 0')
        except Exception: pass
        try:
            self.conn.execute('ALTER TABLE session_summaries ADD COLUMN summary_text TEXT NOT NULL DEFAULT 0')
        except Exception: pass
        self.conn.commit()

    async def get_session_owner(self, session_id):
        row = self.conn.execute("SELECT user_id FROM session_metadata WHERE session_id = ?", (session_id,)).fetchone()
        return row["user_id"] if row else None

    async def get_user_sessions(self, user_id):
        # 获取用户的session_mendata
        cached = await self.redis.get(f"user:{user_id}:sessions_meta")
        if cached:
            return json.loads(cached)
        sids = await self.redis.smembers(f"user:{user_id}:sessions")
        if not sids:
            # Redis 空：从 SQLite 恢复老数据
            rows = self.conn.execute("SELECT session_id FROM session_metadata WHERE user_id = ?", (user_id,)).fetchall()
            sids = [r["session_id"] for r in rows]
            for sid in sids:
                await self.redis.sadd(f"user:{user_id}:sessions", sid)
        if not sids:
            return []
        items = []
        for sid in sids:
            n = await self.redis.llen(self._mk(sid))
            if n == 0:
                continue
            first = json.loads(await self.redis.lindex(self._mk(sid), 0))
            last_raw = await self.redis.lindex(self._mk(sid), -1)
            last = json.loads(last_raw) if last_raw else first
            items.append({"session_id": sid, "first_message": first["content"][:50], "message_count": n, "created_at": first["timestamp"], "last_active": last["timestamp"]})
        items.sort(key=lambda x: x["last_active"], reverse=True)
        await self.redis.set(f"user:{user_id}:sessions_meta", json.dumps(items), ex=60)
        return items

    # ── 批量回写 ──

    _seen = set()

    def touch(self, session_id):
        self._seen.add(session_id)

    async def batch_flush(self):
        if not self._seen:
            return  # 没有需要处理的会话，直接退出
        
        for sid in list(self._seen):
            total = await self.redis.llen(self._mk(sid))
            if total == 0:
                self._seen.discard(sid)  # 无消息则移除
                continue
            
            flushed = int(await self.redis.get(self._fk(sid)) or 0)
            if total <= flushed:
                # 无新消息，清除标记
                self._seen.discard(sid)
                continue
            
            # 增量回写新消息到 SQLite
            new = await self.redis.lrange(self._mk(sid), flushed, -1)
            for raw in new:
                m = json.loads(raw)
                self.conn.execute(
                    "INSERT INTO session_messages(session_id,user_id,role,content,timestamp,token_count) VALUES(?,?,?,?,?,?)",
                    (sid, m.get("user_id",""), m["role"], m["content"], m["timestamp"], m["token_count"])
                )
            self.conn.commit()
            
            # 更新已回写位置
            await self.redis.set(self._fk(sid), total)
            
            # 同步会话摘要
            sum_text = await self.redis.get(self._sk(sid)) or ""
            comp = await self.redis.get(self._ck(sid)) or 0
            self.conn.execute(
                "INSERT OR REPLACE INTO session_summaries(session_id,summary_text,compressed,segment_start,segment_end,token_count,created_at,user_id) VALUES(?,?,?,0,0,0,\"\",\"\")",
                (sid, sum_text, comp)
            )
            self.conn.commit()
            
            # 备份完成，清除标记
            self._seen.discard(sid)

    # ── 删除 ──
    async def clear_session(self, session_id):

        #  从用户会话集合中移除（需要查 user_id）
        uid_row = self.conn.execute("SELECT user_id FROM session_metadata WHERE session_id = ?", (session_id,)).fetchone()
        uid = uid_row["user_id"] if uid_row else None
        if uid:
            await self.redis.srem(f"user:{uid}:sessions", session_id)

            # 删 Redis 会话数据
        for f in (self._mk, self._sk, self._ck, self._fk):
            await self.redis.delete(f(session_id))

        self.conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
        self.conn.commit()
        

    # ── 自愈 ──
    async def _init_from_sqlite(self, session_id):
        if await self.redis.exists(self._mk(session_id)):
            return
        null_key = f"session:{session_id}:null"
        if await self.redis.get(null_key) == "1":
            return
        lock = f"lock:init:{session_id}"
        if not await self.redis.set(lock, "1", nx=True, ex=5):
            import asyncio; await asyncio.sleep(0.2); return
        try:
            if await self.redis.exists(self._mk(session_id)):
                return
            rows = self.conn.execute("SELECT role, content, timestamp, token_count, user_id FROM session_messages WHERE session_id = ? ORDER BY id ASC", (session_id,)).fetchall()
            if not rows:
                if not self.conn.execute("SELECT 1 FROM session_metadata WHERE session_id = ?", (session_id,)).fetchone():
                    await self.redis.set(null_key, "1", ex=60)
                return
            for row in rows:
                await self.redis.rpush(self._mk(session_id), json.dumps({"role": row["role"], "content": row["content"], "timestamp": row["timestamp"], "token_count": row["token_count"], "user_id": dict(row).get("user_id", "")}))
            await self.redis.set(self._fk(session_id), len(rows))
            sr = self.conn.execute("SELECT summary_text, compressed FROM session_summaries WHERE session_id = ?", (session_id,)).fetchone()
            if sr:
                await self.redis.set(self._sk(session_id), sr["summary_text"] or "")
                await self.redis.set(self._ck(session_id), sr["compressed"] or 0)
            base = 604800
            for f in (self._mk, self._sk, self._ck, self._fk):
                await self.redis.expire(f(session_id), base + random.randint(0, 7200))
        finally:
            await self.redis.delete(lock)

    # ── SQLite schema ──

    def _ensure_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS session_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, user_id TEXT NOT NULL DEFAULT '', role TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL, token_count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS session_summaries (session_id TEXT PRIMARY KEY, summary_text TEXT NOT NULL DEFAULT '', compressed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS session_metadata (session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT '', last_active TEXT NOT NULL DEFAULT '', first_message TEXT NOT NULL DEFAULT '', message_count INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS idx_sm_sid ON session_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_sm_uid ON session_messages(user_id);
        """)
        self.conn.commit()

    def close(self):
        self.conn.close()
