"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口"""

from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from loguru import logger

from app.config import config


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope Text Embedding (OpenAI 兼容模式)
    
    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        """
        初始化 DashScope Embeddings
        
        Args:
            api_key: DashScope API Key
            model: 嵌入模型名称
            dimensions: 向量维度
        """
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.dimensions = dimensions
        
        # 打印初始化信息
        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"DashScope Embeddings 初始化完成 - "
            f"模型: {model}, 维度: {dimensions}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """掩码 API Key 用于日志"""
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档列表 (LangChain 标准接口)

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        # DashScope text-embedding-v4 每次最多处理 10 条
        BATCH_SIZE = 10
        all_embeddings = []

        try:
            logger.info(f"批量嵌入 {len(texts)} 个文档 (每批次 {BATCH_SIZE} 条)")

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                all_embeddings.extend([item.embedding for item in response.data])
                logger.debug(f"批次 {i // BATCH_SIZE + 1} 嵌入完成, 维度: {len(all_embeddings[-1])}")

            logger.info(f"全部嵌入完成, 共 {len(all_embeddings)} 条")
            return all_embeddings

        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本 (LangChain 标准接口)
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")
        
        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e


class LazyDashScopeEmbeddings(Embeddings):
    """惰性初始化的 DashScope Embeddings 代理

    延迟创建实际的 DashScopeEmbeddings 实例到第一次使用时，
    避免模块导入时因 API Key 未配置而报错。
    """

    def __init__(self):
        self._inner: DashScopeEmbeddings | None = None

    def _ensure(self) -> DashScopeEmbeddings:
        if self._inner is None:
            api_key = config.dashscope_embedding_api_key or config.dashscope_api_key
            if not api_key:
                raise ValueError(
                    "请设置环境变量 DASHSCOPE_API_KEY 或 DASHSCOPE_EMBEDDING_API_KEY，"
                    "参考: https://bailian.console.aliyun.com/#/api-key"
                )
            self._inner = DashScopeEmbeddings(
                api_key=api_key,
                model=config.dashscope_embedding_model,
                dimensions=1024,
            )
        return self._inner

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ensure().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ensure().embed_query(text)


# 全局单例（惰性初始化）
vector_embedding_service = LazyDashScopeEmbeddings()
