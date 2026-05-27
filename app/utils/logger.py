"""日志配置模块

使用 Loguru 配置应用日志
"""

import sys
from loguru import logger
from app.config import config

# 尝试导入腾讯云 CLS SDK
try:
    from tencentcloud.common import credential
    from tencentcloud.cls.v20201016 import cls_client, models
    cls_available = True
except ImportError:
    cls_available = False

def setup_logger():
    """配置日志系统

    按照 Loguru 最佳实践配置全局 logger：
    1. 移除默认处理器
    2. 添加控制台输出（带颜色）
    3. 添加文件输出（按天轮转，自动压缩，异步写入）
    4. 添加腾讯云 CLS 输出（可选）
    """
    # 移除默认处理器
    logger.remove()

    # 添加控制台输出（带颜色格式）
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,  # 显示完整异常栈信息
        diagnose=config.debug,  # Debug 模式下显示变量值
    )

    # 添加文件输出（按天轮转，自动压缩）
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # 每天0点自动切割新日志文件
        retention="7 days",  # 仅保留最近7天的日志
        compression="zip",  # 过期日志自动压缩为zip
        encoding="utf-8",  # 解决中文乱码
        enqueue=True,  # 异步写入，提升性能（避免IO阻塞）
        backtrace=True,  # 显示完整异常栈信息
        diagnose=True,  # 显示变量值，便于调试
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} | {message}",
    )

    # 如果腾讯云 CLS SDK 可用，添加 CLS 输出
if cls_available and config.cls_secret_id and config.cls_secret_key:
    try:
        # 创建腾讯云的客户端
        cred = credential.Credential(config.cls_secret_id, config.cls_secret_key)
        cls_client_instance = cls_client.ClsClient(cred, config.cls_region)

        def database_sink(message):
            """数据库日志sink"""
            req = models.PutLogsRequest()
            req.TopicId = config.cls_topic_database
            req.Logs = [{
                "Time": int(message.record["time"].timestamp()),
                "Contents": [
                    {"Key": "level", "Value": message.record["level"].name},
                    {"Key": "message", "Value": message.record["message"]}
                ]
            }]
            cls_client_instance.PutLogs(req)

        def api_sink(message):
            """API日志sink"""
            req = models.PutLogsRequest()
            req.TopicId = config.cls_topic_api
            req.Logs = [{
                "Time": int(message.record["time"].timestamp()),
                "Contents": [
                    {"Key": "level", "Value": message.record["level"].name},
                    {"Key": "message", "Value": message.record["message"]}
                ]
            }]
            cls_client_instance.PutLogs(req)

        def sync_sink(message):
            """同步日志sink"""
            req = models.PutLogsRequest()
            req.TopicId = config.cls_topic_sync
            req.Logs = [{
                "Time": int(message.record["time"].timestamp()),
                "Contents": [
                    {"Key": "level", "Value": message.record["level"].name},
                    {"Key": "message", "Value": message.record["message"]}
                ]
            }]
            cls_client_instance.PutLogs(req)
        
        def default_sink(message):
            """普通日志sink"""
            req = models.PutLogsRequest()
            req.TopicId = config.cls_topic_default  # 普通日志的默认topic
            req.Logs = [{
                "Time": int(message.record["time"].timestamp()),
                "Contents": [
                    {"Key": "level", "Value": message.record["level"].name},
                    {"Key": "message", "Value": message.record["message"]}
                ]
            }]
            cls_client_instance.PutLogs(req)

        # 按filter分发到不同sink
       # 按filter分发到不同sink
        logger.add(database_sink, filter=lambda record: "【数据库】" in record["message"], level="INFO", enqueue=True)
        logger.add(api_sink, filter=lambda record: "【API】" in record["message"], level="INFO", enqueue=True)
        logger.add(sync_sink, filter=lambda record: "【同步】" in record["message"], level="INFO", enqueue=True)
        logger.add(default_sink, filter=lambda record: "【数据库】" not in record["message"] and "【API】" not in record["message"] and "【同步】" not in record["message"], level="INFO", enqueue=True)
        logger.info("腾讯云CLS日志处理器已配置")

    except Exception as e:
        logger.error(f"配置CLS日志处理器失败: {e}")


setup_logger()
