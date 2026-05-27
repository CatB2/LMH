"""腾讯云 CLS (Cloud Log Service) MCP Server"""

import logging
import functools
import json
import os
import random
import sys
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastmcp import FastMCP

# 确保项目根目录在 sys.path 中，使 from app.config 可正常工作
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import config

try:
    from tencentcloud.common import credential
    from tencentcloud.cls.v20201016 import cls_client, models
    cls_sdk_available = True
except ImportError:
    cls_sdk_available = False
    logging.warning("腾讯云 CLS SDK 未安装")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CLS_MCP_Server")

mcp = FastMCP("CLS")


def log_tool_call(func):
    """装饰器：记录工具调用"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__
        logger.info(f"调用方法: {method_name}, 参数: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"返回状态: SUCCESS")
            return result
        except Exception as e:
            logger.error(f"返回状态: ERROR, 错误: {e}")
            raise
    return wrapper


def read_local_logs(start_time: int, end_time: int, query: str = None) -> list:
    """从本地日志文件读取真实日志"""
    logs = []
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        return logs
    
    for filename in os.listdir(log_dir):
        if filename.startswith("app_") and filename.endswith(".log"):
            filepath = os.path.join(log_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split(' | ', 3)
                        if len(parts) >= 4:
                            time_str = parts[0].strip()
                            level = parts[1].strip()
                            module_info = parts[2].strip()
                            message = parts[3].strip()
                            
                            try:
                                log_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                                log_timestamp = int(log_time.timestamp() * 1000)
                                
                                if start_time <= log_timestamp <= end_time:
                                    if not query or query.lower() in message.lower():
                                        logs.append({
                                            "timestamp": time_str,
                                            "level": level,
                                            "module": module_info,
                                            "message": message,
                                            "source": "file"
                                        })
                            except ValueError:
                                continue
            except Exception as e:
                logger.warning(f"读取文件失败: {e}")
    
    return logs


def query_cls_logs(topic_id: str, start_time: int, end_time: int, query: str = None, limit: int = 100) -> list:
    """从腾讯云CLS查询日志"""
    logs = []
    
    if not cls_sdk_available or not config.cls_secret_id or not config.cls_secret_key:
        logger.warning("腾讯云CLS未配置")
        return logs
    
    try:
        cred = credential.Credential(config.cls_secret_id, config.cls_secret_key)
        client = cls_client.ClsClient(cred, config.cls_region)
        
        req = models.SearchLogRequest()
        req.TopicId = topic_id if topic_id else config.cls_topic_id
        req.From = start_time
        req.To = end_time
        req.Query = query or "*"
        req.Limit = limit
        
        resp = client.SearchLog(req)
        
        for log_group in resp.LogGroups:
            for log in log_group.Logs:
                log_entry = {"timestamp": datetime.fromtimestamp(log.Time).strftime("%Y-%m-%d %H:%M:%S")}
                for content in log.Contents:
                    log_entry[content.Key] = content.Value
                log_entry["source"] = "cls"
                logs.append(log_entry)
        
        logger.info(f"从腾讯云 CLS 获取 {len(logs)} 条日志")
        
    except Exception as e:
        logger.error(f"调用腾讯云 CLS API 失败: {e}")
    
    return logs


def generate_mock_logs(start_time: int, end_time: int, limit: int = 100) -> list:
    """生成模拟日志"""
    logs = []
    current_time_ms = start_time
    count = 0
    
    templates = [
        "【数据同步】开始同步任务",
        "【数据同步】同步完成，共处理 {count} 条记录",
        "【API网关】收到 {method} 请求 {path}",
        "【API网关】请求处理完成",
        "【数据库】连接成功",
        "【数据库】查询执行完成"
    ]
    
    while current_time_ms <= end_time and count < limit:
        log_time = datetime.fromtimestamp(current_time_ms / 1000)
        time_str = log_time.strftime("%Y-%m-%d %H:%M:%S")
        
        template = random.choice(templates)
        message = template.format(count=random.randint(1, 1000), method=random.choice(["GET", "POST"]), path="/api/users")
        
        logs.append({
            "timestamp": time_str,
            "level": random.choice(["INFO", "INFO", "INFO", "WARN"]),
            "module": "mock.module",
            "message": message,
            "source": "mock"
        })
        
        count += 1
        current_time_ms += 60 * 1000
    
    return logs


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（毫秒）"""
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def search_log(
    topic_id: str,
    start_time: int,
    end_time: int,
    query: Optional[str] = None,
    limit: int = 100,
    source: str = "file"
) -> Dict[str, Any]:
    """查询日志
    
    Args:
        topic_id: 主题ID
        start_time: 开始时间戳（毫秒）
        end_time: 结束时间戳（毫秒）
        query: 查询关键词
        limit: 限制数量
        source: 数据源 ("file", "cls", "mock")
    """
    if source == "cls":
        logs = query_cls_logs(topic_id, start_time, end_time, query, limit)
    elif source == "file":
        logs = read_local_logs(start_time, end_time, query)
    else:
        logs = generate_mock_logs(start_time, end_time, limit)
    
    return {
        "topic_id": topic_id,
        "source": source,
        "total": len(logs),
        "logs": logs[:limit],
        "message": f"成功查询 {len(logs)} 条日志"
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")