"""长期记忆模块 - 跨 session 的知识提取、存储、检索与衰减

使用 Milvus 存储，按 user_id 隔离。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service


# Milvus collection 名称
USER_MEMORY_COLLECTION = "user_memory"


def _ensure_collection():
    """确保 user_memory collection 存在"""
    from app.core.milvus_client import milvus_manager

    # 复用 milvus_manager 的连接
    milvus_manager.connect()

    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

    if utility.has_collection(USER_MEMORY_COLLECTION):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="importance", dtype=DataType.FLOAT),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="accessed_at", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="visit_count", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]

    schema = CollectionSchema(fields, description="User long-term memory")
    collection = Collection(USER_MEMORY_COLLECTION, schema)

    collection.create_index("vector", {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    })
    collection.create_index("user_id", {"index_type": "TRIE"})
    collection.load()

    logger.info(f"Milvus collection '{USER_MEMORY_COLLECTION}' 已创建")


def _get_collection():
    """获取 Milvus collection 实例"""
    _ensure_collection()
    from pymilvus import Collection
    return Collection(USER_MEMORY_COLLECTION)


def _get_llm():
    """获取用于记忆提取的 LLM 实例"""
    from app.core.llm_factory import llm_factory
    return llm_factory.create_chat_model(
        model=config.rag_model,
        temperature=0.3,
        streaming=False,
        response_format={"type": "json_object"},
    )


# ── 核心接口 ──


_MEMORY_EXTRACTION_PROMPT = """
从以下对话中提取值得长期记住的信息。
如果没有需要长期记住的信息，返回空列表。

规则：
- profile 类：用户的稳定特征（角色、技术栈、设备、工作背景）
  - 只记录新发现的或明确变更的
  - 不再重复已记录过的内容
- memory 类：正在进行的任务、关键事实、明确表达的偏好
- 不要记录一次性的寒暄、感谢
- importance 评分 0-1，越重要越高

你必须返回 JSON 格式的列表，每个元素包含：
{{"type": "profile"|"memory", "content": "记忆内容", "importance": 0.8}}

用户问题: {question}
助手回答: {answer}
"""


async def extract_memories(question: str, answer: str) -> List[Dict[str, Any]]:
    """从一轮对话中提取值得长期记住的信息

    Returns:
        List[Dict]: 每条含 type, content, importance
    """
    prompt = _MEMORY_EXTRACTION_PROMPT.format(question=question, answer=answer)

    try:
        llm = _get_llm()
        result = await llm.ainvoke(
            [
                SystemMessage(content="你是一个长期记忆提取专家。提取对话中值得长期记住的信息。请以JSON格式输出。"),
                HumanMessage(content=prompt),
            ]
        )

        content = result.content if hasattr(result, "content") else str(result)
        import json
        raw = json.loads(content)
        # json_object 模式可能返回 {"memories": [...]} 或直接返回 [...]
        if isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, list):
                    return v
            return []
        if isinstance(raw, list):
            return raw
        return []
    except Exception as e:
        logger.warning(f"记忆提取失败: {e}")
        return []


async def store_memories(user_id: str, memories: List[Dict[str, Any]]) -> int:
    """将提取的记忆向量化后存入 Milvus

    Returns:
        int: 实际存储的数量（去重后）
    """
    if not memories:
        return 0

    collection = _get_collection()
    now = datetime.now(timezone.utc).isoformat()
    stored = 0

    existing = await search_similar_memories(user_id, "", top_k=100)
    existing_contents = {m["content"]: m for m in existing}

    texts_to_embed = []
    new_entries = []

    for mem in memories:
        mem_type = mem.get("type", "memory")
        content = mem.get("content", "").strip()
        importance = max(0.0, min(1.0, float(mem.get("importance", 0.5))))

        if not content:
            continue

        # 去重：检查是否已存在相似内容
        dup = False
        for existing_content, existing_mem in existing_contents.items():
            if content[:50] in existing_content or existing_content[:50] in content:
                # 保留重要性更高的
                if importance > existing_mem.get("importance", 0):
                    from pymilvus import Collection
                    collection.delete(f'id == "{existing_mem["id"]}"')
                else:
                    dup = True
                break
        if dup:
            continue

        mem_id = str(uuid.uuid4())
        texts_to_embed.append(content)
        new_entries.append({
            "id": mem_id,
            "user_id": user_id,
            "type": mem_type,
            "content": content,
            "importance": importance,
            "created_at": now,
            "accessed_at": now,
            "visit_count": 0,
        })

    if not new_entries:
        return 0

    # 批量向量化
    vectors = vector_embedding_service.embed_documents(texts_to_embed)

    # 插入 Milvus
    entities = [
        [e["id"] for e in new_entries],
        [e["user_id"] for e in new_entries],
        [e["type"] for e in new_entries],
        [e["content"] for e in new_entries],
        [e["importance"] for e in new_entries],
        [e["created_at"] for e in new_entries],
        [e["accessed_at"] for e in new_entries],
        [e["visit_count"] for e in new_entries],
        vectors,
    ]

    collection.insert(entities)
    logger.info(f"存储了 {len(new_entries)} 条长期记忆 (user={user_id})")
    return len(new_entries)


async def search_similar_memories(
    user_id: str,
    query: str,
    top_k: int = 3,
    min_importance: float = 0.0,
) -> List[Dict[str, Any]]:
    """检索用户的长期记忆

    按向量相似度排序 + 标量过滤。
    """
    collection = _get_collection()
    collection.load()

    expr = f'user_id == "{user_id}"'
    if min_importance > 0:
        expr += f" and importance >= {min_importance}"

    # 用 query embedding 做向量检索
    if query and query.strip():
        query_vector = vector_embedding_service.embed_query(query)
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "user_id", "type", "content", "importance",
                           "created_at", "accessed_at", "visit_count"],
        )
        hits = []
        for hits_group in results:
            for hit in hits_group:
                entity = hit.entity
                hits.append({
                    "id": entity.get("id"),
                    "user_id": entity.get("user_id"),
                    "type": entity.get("type"),
                    "content": entity.get("content"),
                    "importance": entity.get("importance"),
                    "created_at": entity.get("created_at"),
                    "accessed_at": entity.get("accessed_at"),
                    "visit_count": entity.get("visit_count"),
                    "score": hit.score,
                })
        return hits
    else:
        # 无 query 时按 importance 降序取
        results = collection.query(
            expr=expr,
            output_fields=["id", "user_id", "type", "content", "importance",
                           "created_at", "accessed_at", "visit_count"],
            limit=top_k,
            sort_fields=["importance"],
        )
        return results


async def get_user_profile(user_id: str) -> List[Dict[str, Any]]:
    """获取用户 profile 类记忆（每次对话自动注入）"""
    collection = _get_collection()
    collection.load()
    results = collection.query(
        expr=f'user_id == "{user_id}" and type == "profile"',
        output_fields=["content", "importance", "created_at"],
        limit=10,
    )
    return results


def decay_memories():
    """对所有 memory 类型的记忆做时间衰减

    每天调用一次：
    - importance *= 0.99
    - importance < 0.2 或 accessed_at > 30 天的删除
    """
    try:
        collection = _get_collection()
        collection.load()

        # 查询所有 memory 类型
        results = collection.query(
            expr='type == "memory"',
            output_fields=["id", "importance", "accessed_at"],
            limit=10000,
        )

        now = datetime.now(timezone.utc)
        to_delete = []
        to_update = []

        for r in results:
            mem_id = r["id"]
            new_importance = r["importance"] * 0.99
            accessed = r.get("accessed_at", "")

            # 删除条件：重要性过低或超过 30 天未访问
            is_stale = False
            if new_importance < 0.2:
                is_stale = True
            elif accessed:
                try:
                    accessed_dt = datetime.fromisoformat(accessed)
                    if (now - accessed_dt).days > 30:
                        is_stale = True
                except ValueError:
                    pass

            if is_stale:
                to_delete.append(mem_id)
            else:
                to_update.append((new_importance, mem_id))

        # 删除过期记忆
        if to_delete:
            from pymilvus import Collection
            for mem_id in to_delete:
                collection.delete(f'id == "{mem_id}"')
            logger.info(f"清理了 {len(to_delete)} 条过期长期记忆")

        # 更新重要性
        for importance, mem_id in to_update:
            collection.upsert([{
                "id": mem_id,
                "importance": importance,
            }])

        if to_update:
            logger.info(f"衰减了 {len(to_update)} 条记忆的重要性")

    except Exception as e:
        logger.warning(f"记忆衰减失败: {e}")
