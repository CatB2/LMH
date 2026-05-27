"""智能运维监控 MCP Server"""

import logging
import functools
import json
import random
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastmcp import FastMCP

# psutil 可选安装，未安装时使用模拟数据
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Monitor_MCP_Server")

mcp = FastMCP("Monitor")


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


def get_real_cpu_data(interval_minutes: int, sample_count: int = 60) -> list:
    """获取真实CPU数据（psutil 不可用时回退到模拟数据）"""
    if not psutil_available:
        logger.warning("psutil 未安装，使用模拟 CPU 数据")
        return get_mock_cpu_data(interval_minutes, sample_count)
    data_points = []
    current_time = datetime.now()
    cpu_percent = psutil.cpu_percent(interval=1)  # 真实数据

    for i in range(min(sample_count, 30)):
        timestamp = current_time - timedelta(minutes=i * interval_minutes)
        data_points.append({
            "timestamp": timestamp.strftime("%H:%M"),
            "value": cpu_percent,
            "source": "real"
        })

    return sorted(data_points, key=lambda x: x["timestamp"])


def get_mock_cpu_data(interval_minutes: int, sample_count: int) -> list:
    """获取模拟CPU数据"""
    data_points = []
    current_time = datetime.now()
    base_cpu = 10.0
    
    for i in range(min(sample_count, 30)):
        timestamp = current_time - timedelta(minutes=i * interval_minutes)
        cpu_value = min(base_cpu + i * 8, 95.0) + random.uniform(-2, 2)
        cpu_value = max(0, min(100, cpu_value))
        
        data_points.append({
            "timestamp": timestamp.strftime("%H:%M"),
            "value": round(cpu_value, 1),
            "source": "mock"
        })
    
    return sorted(data_points, key=lambda x: x["timestamp"])


def get_real_memory_data(interval_minutes: int, sample_count: int = 60) -> list:
    """获取真实内存数据（psutil 不可用时回退到模拟数据）"""
    if not psutil_available:
        logger.warning("psutil 未安装，使用模拟内存数据")
        return get_mock_memory_data(interval_minutes, sample_count)
    data_points = []
    current_time = datetime.now()
    memory_percent = psutil.virtual_memory().percent  # 真实数据

    for i in range(min(sample_count, 30)):
        timestamp = current_time - timedelta(minutes=i * interval_minutes)
        data_points.append({
            "timestamp": timestamp.strftime("%H:%M"),
            "value": memory_percent,
            "source": "real"
        })

    return sorted(data_points, key=lambda x: x["timestamp"])


def get_mock_memory_data(interval_minutes: int, sample_count: int) -> list:
    """获取模拟内存数据"""
    data_points = []
    current_time = datetime.now()
    
    for i in range(min(sample_count, 30)):
        timestamp = current_time - timedelta(minutes=i * interval_minutes)
        memory_value = 50 + random.uniform(-5, 10)
        memory_value = max(0, min(100, memory_value))
        
        data_points.append({
            "timestamp": timestamp.strftime("%H:%M"),
            "value": round(memory_value, 1),
            "source": "mock"
        })
    
    return sorted(data_points, key=lambda x: x["timestamp"])


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（毫秒）"""
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str = "default",
    interval: str = "5m",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    use_real_data: bool = True
) -> Dict[str, Any]:
    """查询CPU指标数据
    
    Args:
        service_name: 服务名称
        interval: 时间间隔 (如 "5m", "1h")
        start_time: 开始时间戳（毫秒）
        end_time: 结束时间戳（毫秒）
        use_real_data: 是否使用真实数据 (True=真实, False=模拟)
    """
    # 解析时间间隔
    interval_minutes = int(interval.replace("m", ""))
    
    # 根据参数选择真实或模拟数据
    if use_real_data:
        data_points = get_real_cpu_data(interval_minutes)
    else:
        sample_count = 30
        data_points = get_mock_cpu_data(interval_minutes, sample_count)
    
    # 计算统计信息
    if data_points:
        values = [d["value"] for d in data_points]
        return {
            "service_name": service_name,
            "metric_name": "cpu_usage_percent",
            "data_source": "real" if use_real_data else "mock",
            "data_points": data_points,
            "statistics": {
                "avg": round(sum(values) / len(values), 2),
                "max": max(values),
                "min": min(values)
            }
        }
    
    return {"service_name": service_name, "data_points": [], "data_source": "none"}


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str = "default",
    interval: str = "5m",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    use_real_data: bool = True
) -> Dict[str, Any]:
    """查询内存指标数据
    
    Args:
        service_name: 服务名称
        interval: 时间间隔 (如 "5m", "1h")
        start_time: 开始时间戳（毫秒）
        end_time: 结束时间戳（毫秒）
        use_real_data: 是否使用真实数据 (True=真实, False=模拟)
    """
    interval_minutes = int(interval.replace("m", ""))
    
    if use_real_data:
        data_points = get_real_memory_data(interval_minutes)
    else:
        sample_count = 30
        data_points = get_mock_memory_data(interval_minutes, sample_count)
    
    if data_points:
        values = [d["value"] for d in data_points]
        return {
            "service_name": service_name,
            "metric_name": "memory_usage_percent",
            "data_source": "real" if use_real_data else "mock",
            "data_points": data_points,
            "statistics": {
                "avg": round(sum(values) / len(values), 2),
                "max": max(values),
                "min": min(values)
            }
        }
    
    return {"service_name": service_name, "data_points": [], "data_source": "none"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")