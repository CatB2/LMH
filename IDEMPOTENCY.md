# 幂等方案实现代码

---

## 一、唯一标识法（Request-ID）

### 数据库实现

```sql
-- 幂等键表
CREATE TABLE idempotency_keys (
    key         TEXT PRIMARY KEY,        -- request-id
    response    TEXT NOT NULL,           -- 第一次请求的结果
    created_at  TEXT NOT NULL,
    expired_at  TEXT NOT NULL            -- 定时任务清理过期记录
);
CREATE INDEX idx_expired ON idempotency_keys(expired_at);
```

```python
async def idempotent_handler(user_id, request_id, handler_func):
    key = f"req:{user_id}:{request_id}"

    # 查缓存
    cached = Redis.GET(key)
    if cached:
        return json.loads(cached)

    # 插入幂等键（转瞬即逝），利用唯一约束
    try:
        SQLite.execute(
            "INSERT INTO idempotency_keys (key, response, created_at, expired_at) "
            "VALUES (?, ?, ?, ?)",
            key, "", now(), now() + timedelta(hours=24)
        )
    except UniqueViolation:
        # 重复请求，返回已存储的结果
        row = SQLite.query(
            "SELECT response FROM idempotency_keys WHERE key = ?", key
        )
        return json.loads(row.response)

    # 第一次请求，正常处理
    result = await handler_func()
    SQLite.execute("UPDATE idempotency_keys SET response = ? WHERE key = ?",
                   json.dumps(result), key)

    # 写入 Redis 缓存（快速路径）
    Redis.SET(key, json.dumps(result), EX=3600)
    return result
```

### Redis 实现（更简单，你项目适用）

```python
async def idempotent_chat(user_id, request_id, question, session_id):
    key = f"idem:{user_id}:{request_id}"

    # SETNX：第一次成功，重复请求失败
    acquired = Redis.SET(key, "processing", NX=True, EX=60)
    if not acquired:
        # 有人处理过或正在处理
        result = Redis.GET(f"{key}:result")
        if result:
            return json.loads(result)    # 已完成，直接返回
        await asyncio.sleep(0.5)
        return json.loads(Redis.GET(f"{key}:result"))

    try:
        result = await query(question, session_id, user_id)
        Redis.SET(f"{key}:result", json.dumps(result), EX=60)
        return result
    finally:
        Redis.DEL(key)    # 标记处理完成
```

---

## 二、乐观锁（版本号）

### 数据库实现

```sql
-- 库存表带版本号
CREATE TABLE inventory (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    stock     INTEGER NOT NULL,
    version   INTEGER NOT NULL DEFAULT 1
);
```

```python
def deduct_stock(product_id, quantity):
    while True:    # 最多重试 3 次
        # 1. 读出当前版本
        row = SQLite.query(
            "SELECT stock, version FROM inventory WHERE id = ?", product_id
        )
        if row.stock < quantity:
            raise ValueError("库存不足")

        new_stock = row.stock - quantity
        new_version = row.version + 1

        # 2. 更新时检查版本没变
        affected = SQLite.execute(
            "UPDATE inventory SET stock = ?, version = ? "
            "WHERE id = ? AND version = ?",
            new_stock, new_version, product_id, row.version
        )

        # 3. affected_rows = 0 → 版本已变，被别的请求抢先了
        if affected == 0:
            continue   # 重试

        return new_stock
```

### Redis 实现（CAS 版本）

```python
# Redis WATCH + MULTI（事务）
def deduct_stock_redis(product_id, quantity):
    while True:
        Redis.WATCH(f"stock:{product_id}:version")
        version = Redis.GET(f"stock:{product_id}:version")
        stock = Redis.GET(f"stock:{product_id}")

        if int(stock) < quantity:
            Redis.UNWATCH()
            raise ValueError("库存不足")

        # 开启事务：WATCH 的 key 被改过则事务自动取消
        pipe = Redis.PIPELINE()
        pipe.MULTI()
        pipe.DECRBY(f"stock:{product_id}", quantity)
        pipe.INCR(f"stock:{product_id}:version")
        result = pipe.EXEC()    # 返回 None 表示事务被取消（冲突了）

        if result is not None:
            return int(stock) - quantity
        # 否则重试
```

---

## 三、悲观锁

### 数据库行锁

```sql
-- 转账：A 扣钱，B 加钱
BEGIN;
    -- FOR UPDATE：锁住这两行，其他事务不能读写
    SELECT balance FROM accounts WHERE id = 'A' FOR UPDATE;
    SELECT balance FROM accounts WHERE id = 'B' FOR UPDATE;

    UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
    UPDATE accounts SET balance = balance + 100 WHERE id = 'B';
COMMIT;
```

```python
def transfer(from_id, to_id, amount):
    with SQLite.transaction():
        # 锁住行
        from_bal = SQLite.query(
            "SELECT balance FROM accounts WHERE id = ? FOR UPDATE", from_id
        )
        to_bal = SQLite.query(
            "SELECT balance FROM accounts WHERE id = ? FOR UPDATE", to_id
        )

        if from_bal < amount:
            raise ValueError("余额不足")

        SQLite.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?",
                       amount, from_id)
        SQLite.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?",
                       amount, to_id)
```

### Redis 分布式锁

```python
def locked_operation(resource_id):
    lock_key = f"lock:{resource_id}"
    lock_val = uuid.uuid4().hex    # 唯一标识，释放时校验

    # 尝试获取锁：最多等 5 秒
    acquired = Redis.SET(lock_key, lock_val, NX=True, EX=10)
    if not acquired:
        time.sleep(0.1)
        acquired = Redis.SET(lock_key, lock_val, NX=True, EX=10)
        if not acquired:
            raise TimeoutError("获取锁超时")

    try:
        return do_work()
    finally:
        # 释放前校验：别把别人的锁删了
        if Redis.GET(lock_key) == lock_val:
            Redis.DEL(lock_key)
```

---

## 四、状态机

### 订单状态流转

```python
class OrderStatus:
    PENDING  = "pending"       # 待处理
    PAID     = "paid"          # 已支付
    SHIPPED  = "shipped"       # 已发货
    COMPLETED = "completed"    # 已完成

# 合法的状态转移
ALLOWED_TRANSITIONS = {
    OrderStatus.PENDING:  [OrderStatus.PAID],
    OrderStatus.PAID:     [OrderStatus.SHIPPED],
    OrderStatus.SHIPPED:  [OrderStatus.COMPLETED],
}

def pay_order(order_id):
    # UPDATE + WHERE + 状态检查 → 原子操作
    affected = SQLite.execute(
        "UPDATE orders SET status = ? WHERE id = ? AND status = ?",
        OrderStatus.PAID, order_id, OrderStatus.PENDING
    )
    # affected_rows = 0 → 不是 pending 状态，已经被支付过
    if affected == 0:
        current = SQLite.query("SELECT status FROM orders WHERE id = ?", order_id)
        raise ValueError(f"订单状态为 {current.status}，不能支付")


def ship_order(order_id):
    affected = SQLite.execute(
        "UPDATE orders SET status = ? WHERE id = ? AND status = ?",
        OrderStatus.SHIPPED, order_id, OrderStatus.PAID
    )
    if affected == 0:
        raise ValueError("只能发货已支付的订单")
```

---

## 五、去重表（消息指纹）

### 消息队列消费防重

```python
async def consume_message(message):
    # 计算消息指纹
    fingerprint = hashlib.sha256(
        f"{message.order_id}{message.user_id}{message.amount}".encode()
    ).hexdigest()

    try:
        SQLite.execute(
            "INSERT INTO idempotency_keys (key, created_at) VALUES (?, ?)",
            fingerprint, datetime.now()
        )
    except UniqueViolation:
        # 重复消费，已处理过
        logger.info(f"重复消息，跳过: {fingerprint}")
        return

    await process_order(message)
```

### 回调通知防重

```python
# 微信支付回调：微信可能会发多次相同的通知
async def payment_callback(request):
    txn_id = request.transaction_id    # 微信交易号

    # Redis 直接标记
    key = f"cb:{txn_id}"
    if not Redis.SET(key, "1", NX=True, EX=86400):
        return "SUCCESS"    # 已处理，响应成功避免微信重发

    try:
        await update_order_status(txn_id)
        return "SUCCESS"
    except Exception:
        Redis.DEL(key)     # 失败了删掉，允许重试
        raise
```

---

## 六、天然幂等

```python
# DELETE：删两次，第二次 affected_rows = 0，结果一样
SQLite.execute("DELETE FROM sessions WHERE id = ?", session_id)

# SET / PUT：设置两次同一个值
SQLite.execute("UPDATE users SET nickname = ? WHERE id = ?", "张三", user_id)

# INSERT ... ON DUPLICATE KEY（存在则更新）
SQLite.execute(
    "INSERT INTO users (id, name) VALUES (?, ?) "
    "ON CONFLICT(id) DO UPDATE SET name = excluded.name",
    user_id, "张三"
)
```

---

## 七、TTL 窗口去重

```python
# 30 秒内同一用户不能重复发同一内容
async def chat_with_window(user_id, question, session_id):
    req_hash = hashlib.md5(f"{user_id}:{question}".encode()).hexdigest()
    key = f"dedup:{user_id}:{req_hash}"

    # 30 秒窗中内相同内容只处理一次
    if not Redis.SET(key, "1", NX=True, EX=30):
        raise ValueError("请勿重复发送相同内容")

    return await query(question, session_id, user_id)
```

---

## 小结

```
方案           实现核心                    适用场景
────────────────────────────────────────────────────────
Request-ID     SETNX + 缓存结果            支付、下单、聊天
乐观锁         WHERE version = old_version 库存扣减、文档编辑
悲观锁         FOR UPDATE / SETNX 锁      转账、秒杀
状态机         WHERE status = expected    订单、审批流
去重表         hash + 唯一约束            消息队列、回调
天然幂等       DELETE / ON CONFLICT        删除、覆盖更新
TTL 窗口       SETNX + EX                 防连点、内容去重
```
