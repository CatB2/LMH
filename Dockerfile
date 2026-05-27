FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖（部分 Python 包编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖描述文件，利用 Docker 缓存
COPY pyproject.toml ./

# 创建最小占位包（setuptools 需要 app 包存在才能解析依赖）
RUN mkdir -p app && echo '__version__ = "1.0.0"' > app/__init__.py

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[pdf]" \
    && pip install --no-cache-dir "redis[hiredis]"

# 删除占位包，拷贝真实应用代码
RUN rm -rf app
COPY app/ ./app/
COPY static/ ./static/
COPY aiops-docs/ ./aiops-docs/
COPY mcp_servers/ ./mcp_servers/
COPY _start_server.py ./

# 创建运行时目录
RUN mkdir -p /app/data /app/uploads /app/logs

# 暴露端口
# 9900: FastAPI 主服务
# 8003: MCP CLS 服务
# 8004: MCP Monitor 服务
EXPOSE 9900 8003 8004

# 默认启动 FastAPI
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
