# SuperBizAgent 上下文管理 & 存储重构设计

## 一、总体架构

### 存储职责

```
Redis   — 主存储，读写全走这里，毫秒级
SQLite  — 持久化副本，每 5 秒增量同步
Milvus  — 长期记忆，独立系统
```

### Redis 数据结构

```
session:{id}:messages     List    全部消息，索引从 0 开始
                                 元素: {role, content, timestamp, token_count}
                                 TTL: 7 天（防雪崩：加随机偏移）

session:{id}:summary      String  累积摘要文本
                                 TTL: 7 天

session:{id}:compressed   Integer 已压缩的消息数量（build_messages 从该索引开始取）
                                 TTL: 7 天

session:{id}:flushed      Integer 已回写 SQLite 的消息数量（batch_flush 中用作 LRANGE 起点）
                                 TTL: 7 天

user:{uid}:sessions       Set     该用户所有 session_id
                                 无 TTL

user:{uid}:sessions_meta  String  会话列表 JSON 缓存
                                 TTL: 60 秒
```

### SQLite 表结构

```sql
-- 消息表
CREATE TABLE session_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    token_count  INTEGER NOT NULL
);
CREATE INDEX idx_messages_session ON session_messages(session_id, id);

-- 摘要表（每个会话一行）
CREATE TABLE session_summaries (
    session_id   TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL DEFAULT '',
    compressed   INTEGER NOT NULL DEFAULT 0
);

-- 元数据表
CREATE TABLE session_metadata (
    session_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    first_message TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    last_active   TEXT NOT NULL
);
```

### 配置

```
rag_context_token_limit       3000    上下文最大 token
rag_context_recent_token_keep 1200    未压缩消息最多占的 token
MAX_TURNS                     50      单会话最大对话轮数
batch_flush_interval          5       增量回写间隔（秒）
sessions_meta_ttl             60      会话列表缓存（秒）
session_ttl                   604800  Redis 会话 TTL（秒，7 天）
```

---

## 二、核心流程

### query(session_id, question, user_id)

```python
async def query(question, session_id, user_id):
    # 1. 自愈：确保 Redis 有数据
    await init_from_sqlite(session_id)
    touch(session_id)

    # 2. 轮数上限（数据恢复之后判断，拿到真实值）
    total = Redis.LLEN(f"session:{session_id}:messages")
    if total // 2 >= MAX_TURNS:
        raise ValueError("已达对话上限，请新建会话")

    # 3. 构建上下文
    context = await build_messages(session_id)
    context.append(HumanMessage(content=question))
    context = MemoryManager.inject(context, user_id, question)

    # 4. LLM
    answer = await Agent.ainvoke({"messages": context})

    # 5. 写 Redis
    now = datetime.utcnow().isoformat()
    msg_user = {
        "role": "user", "content": question,
        "timestamp": now, "token_count": estimate_tokens(question)
    }
    msg_ai = {
        "role": "assistant", "content": answer,
        "timestamp": now, "token_count": estimate_tokens(answer)
    }
    Redis.RPUSH(f"session:{session_id}:messages", json.dumps(msg_user))
    Redis.RPUSH(f"session:{session_id}:messages", json.dumps(msg_ai))
    Redis.DEL(f"user:{user_id}:sessions_meta")

    # TTL 续期：每次写消息时刷新，避免活跃会话在 7 天后集中过期
    ttl = 604800
    for suffix in ["messages", "summary", "compressed", "flushed"]:
        Redis.EXPIRE(f"session:{session_id}:{suffix}", ttl)

    # 6. 压缩
    await save_context(session_id)

    # 7. 记忆提取（异步）
    MemoryManager.schedule_extract(user_id, session_id, question, answer)

    return answer
```

---

### build_messages(session_id)

```python
async def build_messages(session_id):
    await init_from_sqlite(session_id)

    # 读摘要
    summary = Redis.GET(f"session:{session_id}:summary") or ""

    # 取未压缩部分（save_context 已保证其在 1200 token 以内，直接全取）
    compressed = Redis.GET(f"session:{session_id}:compressed") or 0
    raw_all = Redis.LRANGE(f"session:{session_id}:messages", compressed, -1)
    uncompressed_msgs = [json.loads(r) for r in raw_all]

    # 拼装
    context = []
    if summary:
        context.append(SystemMessage(f"以下是历史对话摘要：\n{summary}"))
    for msg in uncompressed_msgs:
        if msg["role"] == "user":
            context.append(HumanMessage(content=msg["content"]))
        else:
            context.append(AIMessage(content=msg["content"]))
    return context
```

---

### save_context(session_id)

```python
async def save_context(session_id):
    total = Redis.LLEN(f"session:{session_id}:messages")
    compressed = Redis.GET(f"session:{session_id}:compressed") or 0

    # 取未压缩部分
    raw_msgs = Redis.LRANGE(f"session:{session_id}:messages", compressed, -1)
    uncompresseds = [json.loads(m) for m in raw_msgs]

    total_tokens = sum(m["token_count"] + 4 for m in uncompresseds)
    if total_tokens <= rag_context_recent_token_keep:
        return

    # 取超出部分的旧消息
    excess = total_tokens - rag_context_recent_token_keep
    to_compress = []
    chunk_tokens = 0
    for msg in uncompresseds:
        to_compress.append(msg)
        chunk_tokens += msg["token_count"] + 4
        if chunk_tokens >= excess:
            break

    # LLM 压缩
    old_summary = Redis.GET(f"session:{session_id}:summary") or ""
    new_summary = await compress_to_summary(summary_model, old_summary, to_compress)

    # 写入 Redis
    Redis.SET(f"session:{session_id}:summary", new_summary)
    Redis.SET(f"session:{session_id}:compressed", compressed + len(to_compress))


# ── 摘要压缩（独立函数） ──

async def compress_to_summary(summary_model, old_summary, messages):
    """旧摘要 + 新消息 → LLM 融合成一段新摘要"""
    conversation = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
        for m in messages
    )
    prompt = "请将以下信息融合成一段简洁连贯的摘要：\n\n"
    if old_summary:
        prompt += f"【已有摘要】\n{old_summary}\n\n"
    prompt += f"【近期对话】\n{conversation}"
    result = await summary_model.ainvoke(prompt)
    return result.content if hasattr(result, "content") else str(result)
```

---

### init_from_sqlite(session_id)

```python
async def init_from_sqlite(session_id):
    """确保 Redis 有数据。首次调用或 Redis 重启后自动触发。"""
    msg_key = f"session:{session_id}:messages"

    # 快速路径：Redis 已有数据
    if Redis.EXISTS(msg_key):
        return

    # 空值缓存：已确认不存在的 session，60 秒内不再查 SQLite（防穿透）
    if Redis.GET(f"session:{session_id}:null") == "1":
        return

    # SETNX 锁：同一时刻只让一个请求重建缓存（防击穿）
    lock_val = str(uuid.uuid4())
    if not Redis.SET(f"lock:init:{session_id}", lock_val, NX=True, EX=5):
        await asyncio.sleep(0.2)
        return

    try:
        # 双重检查
        if Redis.EXISTS(msg_key):
            return

        rows = SQLite.query(
            "SELECT role, content, timestamp, token_count "
            "FROM session_messages WHERE session_id = ? ORDER BY id ASC",
            session_id
        )

        if not rows:
            # valid session（metadata 存在）保留穿透可能，等 flush 写入后恢复
            if not SQLite.exists("session_metadata", session_id):
                Redis.SET(f"session:{session_id}:null", "1", EX=60)
            return

        for row in rows:
            msg = {"role": row.role, "content": row.content,
                   "timestamp": row.timestamp, "token_count": row.token_count}
            Redis.RPUSH(msg_key, json.dumps(msg))

        Redis.SET(f"session:{session_id}:flushed", len(rows))

        summary_row = SQLite.query(
            "SELECT summary_text, compressed FROM session_summaries WHERE session_id = ?",
            session_id
        )
        if summary_row:
            Redis.SET(f"session:{session_id}:summary", summary_row.summary_text)
            Redis.SET(f"session:{session_id}:compressed", summary_row.compressed)

        # 防雪崩：TTL 加随机偏移
        base_ttl = 604800
        jitter = random.randint(0, 7200)
        Redis.EXPIRE(msg_key, base_ttl + jitter)
        Redis.EXPIRE(f"session:{session_id}:summary", base_ttl + jitter)
        Redis.EXPIRE(f"session:{session_id}:compressed", base_ttl + jitter)
        Redis.EXPIRE(f"session:{session_id}:flushed", base_ttl + jitter)

    finally:
        if Redis.GET(f"lock:init:{session_id}") == lock_val:
            Redis.DEL(f"lock:init:{session_id}")
```

---

### 活跃会话跟踪 & batch_flush

```python
# 内存集合：touch() 往里加，batch_flush 时遍历
_seen_sessions: set = set()

def touch(session_id):
    _seen_sessions.add(session_id)


def batch_flush():
    """每 5 秒执行，增量同步 Redis → SQLite"""
    global _seen_sessions

    # 服务启动时内存为空，用现有 Redis key 预热（仅首次调用时执行）
    if not _seen_sessions:
        keys = Redis.SCAN(match="session:*:messages")
        for k in keys:
            sid = k.split(":")[1]
            _seen_sessions.add(sid)

    deleted = Redis.SMEMBERS("deleted_sessions")

    active = []
    for session_id in list(_seen_sessions):
        # 跳过已删除
        if session_id in deleted:
            _seen_sessions.discard(session_id)
            continue
        # TTL 过期，key 已不存在
        total = Redis.LLEN(f"session:{session_id}:messages")
        if total == 0:
            _seen_sessions.discard(session_id)
            continue
        flushed = int(Redis.GET(f"session:{session_id}:flushed") or 0)
        if total <= flushed:
            active.append(session_id)
            continue

        # 只写增量
        new_msgs = Redis.LRANGE(f"session:{session_id}:messages", flushed, -1)
        with SQLite.transaction():
            for raw in new_msgs:
                msg = json.loads(raw)
                SQLite.execute(
                    "INSERT INTO session_messages (session_id, role, content, timestamp, token_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    session_id, msg["role"], msg["content"], msg["timestamp"], msg["token_count"]
                )
        Redis.SET(f"session:{session_id}:flushed", total)
        active.append(session_id)

        # 同步摘要和压缩指针
        summary = Redis.GET(f"session:{session_id}:summary") or ""
        compressed = Redis.GET(f"session:{session_id}:compressed") or 0
        SQLite.execute(
            "INSERT OR REPLACE INTO session_summaries (session_id, summary_text, compressed) "
            "VALUES (?, ?, ?)",
            session_id, summary, compressed
        )

    _seen_sessions = set(active)
```

---

## 三、长期记忆

### MemoryManager

```python
class MemoryManager:
    def __init__(self):
        self._counters = {}

    def inject(self, messages, user_id, question):
        if not user_id:
            return messages
        profile = Milvus.query(type="profile", user_id=user_id)
        related = Milvus.search(user_id, question, top_k=3)
        if profile or related:
            parts = []
            if profile:
                parts.append(f"## 用户画像\n" + "\n".join(f"- {m['content']}" for m in profile))
            if related:
                parts.append(f"## 相关记忆\n" + "\n".join(f"- {m['content']}" for m in related))
            msg = SystemMessage(content=f"长期记忆：\n\n" + "\n\n".join(parts))
            messages.insert(0, msg)
        return messages

    def schedule_extract(self, user_id, session_id, question, answer):
        if not user_id or not answer:
            return
        if len(answer.strip()) < 60:
            return
        q = question.strip()
        if len(q) < 15 and not any(kw in q for kw in PERSONAL_KEYWORDS):
            return
        count = self._counters.get(session_id, 0) + 1
        self._counters[session_id] = count
        if count % 3 != 0:
            return
        asyncio.create_task(self._extract(user_id, question, answer))

    async def _extract(self, user_id, question, answer):
        memories = await extract_memories(question, answer)
        if memories:
            await store_memories(user_id, memories)
```

### user_memory.py

```python
# 集合: user_memory，IVF_FLAT + IP，nlist=128，1024 维

def extract_memories(question, answer):
    llm = ChatOpenAI(model=config.rag_model, temperature=0.3, response_format="json_object")
    result = llm.invoke(prompt)
    return parse_json_list(result)  # [{"type":"profile","content":"...","importance":0.8}]

def store_memories(user_id, memories):
    existing = search_similar_memories(user_id, "", top_k=100)
    new = dedup(existing, memories)
    if not new:
        return
    vectors = embed([m["content"] for m in new])
    Milvus.insert(entities, vectors)

def search_similar_memories(user_id, query, top_k=3):
    if query:
        vec = embed(query)
        return Milvus.search(vec, expr=f'user_id=="{user_id}"',
                            anns_field="vector", metric_type="IP",
                            params={"nprobe": 10}, limit=top_k)
    else:
        return Milvus.query(expr=f'user_id=="{user_id}"',
                           output_fields=[...], sort_fields=["importance"])

def decay_memories():
    """每天一次：importance *= 0.99，< 0.2 或 30 天未访问 → 删除"""
```

---

## 四、会话管理

### 创建会话
```python
def create_session(user_id, session_id):
    Redis.SADD(f"user:{user_id}:sessions", session_id)
    Redis.DEL(f"user:{user_id}:sessions_meta")
    now = datetime.utcnow().isoformat()
    SQLite.execute(
        "INSERT INTO session_metadata (session_id, user_id, created_at, last_active) "
        "VALUES (?, ?, ?, ?)",
        session_id, user_id, now, now
    )
```

### 获取会话归属
```python
def get_session_owner(session_id):
    row = SQLite.query(
        "SELECT user_id FROM session_metadata WHERE session_id = ?",
        session_id
    )
    return row.user_id if row else None
```

### 删除会话
```python
def delete_session(session_id):
    uid = get_session_owner(session_id)
    if not uid:
        return

    Redis.DEL(f"session:{session_id}:messages")
    Redis.DEL(f"session:{session_id}:summary")
    Redis.DEL(f"session:{session_id}:compressed")
    Redis.DEL(f"session:{session_id}:flushed")

    Redis.SREM(f"user:{uid}:sessions", session_id)
    Redis.DEL(f"user:{uid}:sessions_meta")

    SQLite.execute("DELETE FROM session_messages WHERE session_id = ?", session_id)
    SQLite.execute("DELETE FROM session_summaries WHERE session_id = ?", session_id)
    SQLite.execute("DELETE FROM session_metadata WHERE session_id = ?", session_id)

    Redis.SADD("deleted_sessions", session_id)
    Redis.EXPIRE("deleted_sessions", 3600)
```

### 展示会话消息
```python
async def get_session_messages(session_id):
    await init_from_sqlite(session_id)
    raws = Redis.LRANGE(f"session:{session_id}:messages", 0, -1)
    return [json.loads(r) for r in raws]
```

### 展示会话列表
```python
def get_user_sessions(user_id):
    cached = Redis.GET(f"user:{user_id}:sessions_meta")
    if cached:
        return json.loads(cached)

    session_ids = Redis.SMEMBERS(f"user:{user_id}:sessions")
    if not session_ids:
        return []

    items = []
    for sid in session_ids:
        count = Redis.LLEN(f"session:{sid}:messages")
        if count == 0:
            continue
        raw_first = Redis.LINDEX(f"session:{sid}:messages", 0)
        first = json.loads(raw_first)
        raw_last = Redis.LINDEX(f"session:{sid}:messages", -1)
        last = json.loads(raw_last)
        items.append({
            "session_id": sid,
            "first_message": first["content"][:50],
            "message_count": count,
            "created_at": first["timestamp"],
            "last_active": last["timestamp"]
        })

    items.sort(key=lambda x: x["last_active"], reverse=True)
    Redis.SET(f"user:{user_id}:sessions_meta", json.dumps(items), EX=60)
    return items
```

---

## 五、并发分析

```
并发场景                       结论
────────────────────────────────────────────
同会话两个 query() 同时执行     不会发生
build_messages 和 save_context  读+写，无冲突
append_message 和 save_context  RPUSH + LLEN，无冲突
save_context 和 batch_flush     微小竞态，不影响正确性
init_from_sqlite 和其他操作     SETNX 锁 + 双重检查，安全
delete_session 任意操作         同步删 Redis + SQLite + deleted_sessions 标记

→ 无需加锁
```

---

## 六、完整流程图

```
用户发消息
  │
  ├─ init_from_sqlite → Redis miss 则从 SQLite 恢复
  ├─ touch → 内存标记活跃
  ├─ LLEN check → 轮数上限
  ├─ build_messages
  │   ├─ GET summary
  │   └─ LRANGE 取未压缩部分（save_context 已保证在预算内）
  ├─ MemoryManager.inject → Milvus 查记忆
  ├─ Agent.ainvoke → LLM (20s)
  ├─ RPUSH × 2 到 Redis messages
  ├─ TTL 续期 → 刷新所有 session key
  ├─ save_context
  │   ├─ 未压缩 token > 1200?
  │   └─ ✅ → LLM 压缩 → 追加 summary → 推进 compressed
  └─ schedule_extract → 异步三层过滤 → LLM 提取 → Milvus

后台每 5 秒:
  batch_flush
    ├─ 遍历 _seen_sessions
    ├─ 跳过 TTL 过期和已删除
    ├─ 增量: LRANGE flushed ~ -1 → SQLite
    ├─ SET flushed = total
    └─ SQLite.upsert(summary, compressed)

Redis 崩溃后首次访问:
  init_from_sqlite
    ├─ SETNX 锁 → 只让一个请求重建
    ├─ SQLite SELECT → RPUSH 全量到 Redis
    ├─ SET flushed, compressed, summary
    └─ 释放锁
```
