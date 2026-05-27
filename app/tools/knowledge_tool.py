"""知识检索工具 - 从向量数据库中检索相关信息（混合检索：向量 + BM25）"""

import contextvars
import re
import threading
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger
from rank_bm25 import BM25Okapi

from app.config import config
from app.services.vector_store_manager import vector_store_manager


# 当前用户 ID 上下文变量（由 RAG Agent 在调用前设置）
current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_id", default=""
)

# 最近一次检索的源文件列表（供前端展示，外部读取后请重置为空）
latest_source_files: list[str] = []

# BM25 全局索引（惰性加载，线程安全）
_bm25_lock = threading.Lock()
_bm25_index: BM25Okapi | None = None
_bm25_docs: list[Document] = []


def _tokenize(text: str) -> list[str]:
    """中文 + 英文分词"""
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9_.-]+|[一-鿿]", text)
    return [t for t in tokens if len(t) > 1]


def _ensure_bm25_index():
    """惰性加载 BM25 索引"""
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return

    with _bm25_lock:
        if _bm25_index is not None:
            return

        try:
            from app.core.milvus_client import milvus_manager
            from pymilvus import Collection
            import json as json_mod

            milvus_manager.connect()
            collection = Collection("biz")
            collection.load()

            results = collection.query(
                expr="",
                output_fields=["content", "metadata"],
                limit=10000,
            )

            for r in results:
                meta = r["metadata"]
                if isinstance(meta, str):
                    meta = json_mod.loads(meta)
                _bm25_docs.append(Document(
                    page_content=r["content"],
                    metadata=meta,
                ))

            tokenized = [_tokenize(d.page_content) for d in _bm25_docs]
            _bm25_index = BM25Okapi(tokenized)
            logger.info(f"BM25 索引构建完成，共 {len(_bm25_docs)} 篇文档")
        except Exception as e:
            logger.warning(f"BM25 索引构建失败（降级为纯向量检索）: {e}")


def _rrf_merge(
    vector_docs: list[Document],
    bm25_docs: list[Document],
    k: int = 60,
    top_n: int = 20,
) -> list[Document]:
    """Reciprocal Rank Fusion 合并结果，并按文件来源多样化"""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: -x[1])

    # 按 RRF 分数从高到低取，但每篇文件最多 5 块
    file_counts: dict[str, int] = {}
    diversified: list[Document] = []
    MAX_CHUNKS_PER_FILE = 5

    for key, _ in ranked:
        doc = doc_map[key]
        fname = doc.metadata.get("_file_name", "unknown")
        count = file_counts.get(fname, 0)
        if count < MAX_CHUNKS_PER_FILE:
            diversified.append(doc)
            file_counts[fname] = count + 1
        if len(diversified) >= top_n:
            break

    # 如果凑不满 top_n，从被截断的结果里补充
    if len(diversified) < top_n:
        for key, _ in ranked:
            doc = doc_map[key]
            if not any(d.page_content[:100] == key for d in diversified):
                diversified.append(doc)
                if len(diversified) >= top_n:
                    break

    return diversified[:top_n]


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索上传的文档内容来回答问题。当用户提到"算法名称"、"论文"、"上传的文档"、"知识库"、"文档内容"、"资料"、"参考文档"、“根据文件回答”时，必须调用此工具从向量数据库中搜索相关内容。

    使用场景（必须调用）：
    - 用户说"根据我上传的文档回答..."、"文档里提到了..."
    - 用户问的问题涉及公司内部知识、技术文档、操作手册
    - 用户明确要求"查一下资料"、"搜索知识库"
    - 你不确定答案是否正确，需要从文档中找依据

    Args:
        query: 用户的原始问题或需要搜索的关键词

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 从上下文获取当前用户 ID，实现文档隔离
        uid = current_user_id.get()

        search_kwargs = {"k": config.rag_top_k}
        if uid:
            search_kwargs["expr"] = f'metadata["_user_id"] == "{uid}"'
            logger.info(f"知识检索已限定用户: {uid}")

        # 1. 向量检索（按用户隔离）
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        vector_docs = retriever.invoke(query)

        # 2. BM25 检索（按用户过滤 metadata）
        _ensure_bm25_index()
        bm25_docs: list[Document] = []
        if _bm25_index is not None:
            tokenized_query = _tokenize(query)
            if tokenized_query:
                bm25_scores = _bm25_index.get_scores(tokenized_query)
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )[: config.rag_top_k]

                for i in top_indices:
                    if bm25_scores[i] <= 0:
                        continue
                    # BM25 侧也进行用户隔离
                    if uid:
                        doc_uid = _bm25_docs[i].metadata.get("_user_id", "")
                        if doc_uid and doc_uid != uid:
                            continue
                    bm25_docs.append(_bm25_docs[i])

        # 3. RRF 融合
        all_docs = _rrf_merge(vector_docs, bm25_docs, top_n=config.rag_top_k)

        if not all_docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        context = format_docs(all_docs)
        source_files = list(set(d.metadata.get("_file_name", "") for d in all_docs))
        latest_source_files.clear()
        latest_source_files.extend(source_files)
        logger.info(f"检索到 {len(all_docs)} 个相关文档，来自 {len(source_files)} 篇文件（向量 {len(vector_docs)} + BM25 {len(bm25_docs)}）")
        return context, all_docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """格式化文档列表为上下文文本"""
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
