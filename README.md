# SuperBizAgent

> 基于 LangGraph 的智能业务代理系统 — RAG 知识库问答 + AIOps 智能运维诊断

[![Python](https://img.shields.io/badge/Python-3.11_|_3.12_|_3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)

## 核心特性

- **RAG 智能对话** — Milvus 向量检索 + BM25 混合检索（RRF 融合），支持流式输出
- **AIOps 故障诊断** — Plan-Execute-Replan 模式，自动制定诊断计划、调用工具、生成报告
- **多 Agent 路由** — AgentManager 统一调度，支持普通对话和 AIOps 诊断两种模式
- **长期记忆系统** — 跨会话用户画像 + 记忆衰减机制，越用越懂你
- **JWT 双 Token 鉴权** — access_token + refresh_token，bcrypt 密码加密
- **会话持久化** — Redis 主存 + SQLite 冷备，支持自愈恢复
- **MCP 工具集成** — CLS 日志查询 + Monitor 系统监控，streamable-http 协议
- **Docker 一键部署** — Docker Compose 编排全套服务，Caddy 自动 HTTPS

## 技术栈

| 层 | 技术 |
|----|------|
| 框架 | FastAPI + LangChain + LangGraph |
| LLM | 阿里云 DashScope（通义千问 qwen-max） |
| Embedding | text-embedding-v4（1024 维） |
| 向量库 | Milvus 2.5（IVF_FLAT 索引） |
| 缓存 | Redis 7 |
| 协议 | MCP (Model Context Protocol) |
| 部署 | Docker Compose + Caddy |

## 快速开始

### 环境要求

- Python 3.11+
- Docker Desktop（运行 Milvus + Redis）
- 阿里云百炼 API Key（[申请地址](https://bailian.console.aliyun.com/)）

### 本地开发（Windows）

```powershell
# 1. 克隆项目
git clone git@github.com:CatB2/LMH.git
cd LMH

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 配置 API Key
# 编辑 .env，填入你的 DASHSCOPE_API_KEY

# 4. 启动基础设施（Milvus + Redis）
docker compose -f vector-database.yml up -d
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 5. 启动 MCP 服务（两个终端窗口）
python mcp_servers/cls_server.py      # 端口 8003
python mcp_servers/monitor_server.py  # 端口 8004

# 6. 启动主服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

或者用一键脚本：`.\start-windows.bat`

### 访问

- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs
- **健康检查**: http://localhost:9900/health

## 项目结构

```
LMH/
├── app/
│   ├── main.py                  # FastAPI 入口（lifespan 管理）
│   ├── config.py                # Pydantic Settings 配置
│   ├── api/
│   │   ├── chat.py              # 对话接口（普通/流式）
│   │   ├── aiops.py             # AIOps 诊断接口（SSE）
│   │   ├── auth.py              # 认证接口（注册/登录/刷新/登出）
│   │   ├── file.py              # 文件上传 + 自动索引
│   │   └── health.py            # 健康检查
│   ├── services/
│   │   ├── rag_agent_service.py         # RAG Agent 编排
│   │   ├── aiops_service.py             # AIOps Plan-Execute-Replan
│   │   ├── auth_service.py              # JWT + bcrypt 鉴权
│   │   ├── session_memory_store.py      # Redis + SQLite 会话存储
│   │   ├── context_builder.py           # 上下文压缩 + Token 管理
│   │   ├── memory_manager.py            # 长期记忆注入/提取
│   │   ├── vector_store_manager.py      # Milvus LangChain 封装
│   │   ├── vector_embedding_service.py  # DashScope Embeddings
│   │   ├── vector_index_service.py      # 文件索引流水线
│   │   ├── vector_search_service.py     # 原始 pymilvus 检索
│   │   └── document_splitter_service.py # 文档分割
│   ├── agent/
│   │   ├── agent_manager.py     # 多 Agent 路由调度
│   │   ├── rag_agent.py         # RAG Agent 定义 + MCP 工具加载
│   │   ├── mcp_client.py        # MultiServerMCPClient 管理
│   │   └── aiops/               # Plan-Execute-Replan 三节点
│   │       ├── planner.py       # 制定诊断计划
│   │       ├── executor.py      # 执行诊断步骤
│   │       └── replanner.py     # 评估 + 重规划
│   ├── core/
│   │   ├── llm_factory.py       # ChatOpenAI 工厂
│   │   ├── milvus_client.py     # Milvus 连接管理
│   │   ├── user_memory.py       # 长期记忆（Milvus 存储）
│   │   └── intent_detector.py   # 意图检测（关键词 + LLM）
│   ├── tools/
│   │   ├── knowledge_tool.py    # 知识库检索（向量 + BM25 RRF）
│   │   └── time_tool.py         # 时间查询
│   ├── models/                  # Pydantic 数据模型
│   └── utils/                   # 日志 + 响应工具
├── mcp_servers/
│   ├── cls_server.py            # CLS 日志查询 MCP 服务
│   └── monitor_server.py        # 监控数据 MCP 服务
├── static/                      # Web 前端（原生 JS）
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── aiops-docs/                  # 运维知识库文档（5 篇）
├── tests/                       # 单元测试
├── Dockerfile                   # 应用镜像
├── docker-compose.yml           # 生产环境编排
├── Caddyfile                    # HTTPS 反向代理
├── vector-database.yml          # Milvus 编排
├── .env.docker                  # 生产环境配置模板
├── pyproject.toml               # 依赖管理
└── DEPLOY.md                    # 部署指南
```

## API 接口

### 认证接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 用户注册 |
| POST | `/api/auth/login` | 用户登录（返回双 token） |
| POST | `/api/auth/refresh` | 刷新 access_token |
| POST | `/api/auth/logout` | 登出（吊销 refresh_token） |

### 核心接口（需认证）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 普通对话 |
| POST | `/api/chat_stream` | 流式对话（SSE） |
| POST | `/api/chat/clear` | 清空会话 |
| GET | `/api/chat/session/{id}` | 查询会话历史 |
| GET | `/api/chat/agents` | 获取可用 Agent 列表 |
| GET | `/api/sessions` | 获取用户所有会话 |
| POST | `/api/upload` | 上传文档到知识库 |
| POST | `/api/aiops` | AIOps 智能诊断（SSE） |

### 公开接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康检查 |

### 使用示例

```bash
# 注册
curl -X POST "http://localhost:9900/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"123456"}'

# 登录
curl -X POST "http://localhost:9900/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"123456"}'

# 对话（使用返回的 access_token）
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{"question":"什么是 CPU 高负载？"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{"question":"帮我分析系统状态"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{}' \
  --no-buffer
```

## 配置说明

通过 `.env` 文件配置，关键项：

```bash
# LLM（必填）
DASHSCOPE_API_KEY=sk-xxx              # 阿里云百炼 API Key
DASHSCOPE_MODEL=qwen-max              # 对话模型
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4  # 嵌入模型

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT（生产环境务必修改）
JWT_SECRET_KEY=change-me-to-random-string

# MCP 服务
MCP_CLS_URL=http://localhost:8003/mcp
MCP_MONITOR_URL=http://localhost:8004/mcp
```

## Docker 部署

见 [DEPLOY.md](./DEPLOY.md)

```bash
docker compose --env-file .env.docker up -d --build
```

## 常用命令

| 操作 | 命令 |
|------|------|
| 启动全部服务 | `make start` 或 `.\start-windows.bat` |
| 停止全部服务 | `make stop` 或 `.\stop-windows.bat` |
| 开发模式 | `make dev`（热重载） |
| 健康检查 | `make check` 或 `curl localhost:9900/health` |
| 代码格式化 | `make format` |
| 代码检查 | `make lint` |
| 运行测试 | `pytest` |

## 许可证

MIT License

Author: chief
