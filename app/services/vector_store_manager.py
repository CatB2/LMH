"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


# 统一使用 biz collection
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器——惰性初始化，避免模块导入时连接 Milvus"""

    def __init__(self):
        """初始化向量存储管理器"""
        self._vector_store = None
        self.collection_name = COLLECTION_NAME
        self._initialized = False

    def _ensure_initialized(self):
        """确保 VectorStore 已初始化（惰性加载）"""
        if self._initialized:
            return

        try:
            # 必须在 PyMilvus / langchain_milvus 访问 Collection 之前建立连接，
            # 否则会出现 ConnectionNotExistException: should create connection first.
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            self._vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            self._initialized = True
            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            self._initialized = False
            raise

    def _reset(self):
        """重置状态，允许下次操作时重新初始化（用于 Milvus 断线重连）"""
        self._vector_store = None
        self._initialized = False

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        self._ensure_initialized()
        try:
            import time
            import uuid
            start_time = time.time()

            ids = [str(uuid.uuid4()) for _ in documents]

            result_ids = self._vector_store.add_documents(documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return result_ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            self._reset()
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        if not self._initialized:
            return 0

        try:
            collection = milvus_manager.get_collection()

            # 转义文件路径中的引号，防止表达式注入
            safe_path = file_path.replace('"', '\\"')
            expr = f'metadata["_source"] == "{safe_path}"'

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0

            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败: {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        self._ensure_initialized()
        return self._vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        self._ensure_initialized()
        try:
            docs = self._vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()
