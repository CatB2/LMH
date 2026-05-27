# 面向学术写作的多智能体协同框架

> 基于 LangGraph 的多智能体协同系统 — 支持 RAG 知识库问答、AIOps 智能运维诊断、跨会话长期记忆

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)

## ✨ 核心特性

- 🤖 **多智能体协同** — AgentManager 统一调度，支持 RAG 问答 / AIOps 诊断两种 Agent 模式
- 📚 **RAG 知识库问答** — Milvus 向量检索 + BM25 关键词检索，RRF 混合融合，支持文档上传自动建立向量索引
- 🧠 **长期记忆系统** — 跨会话用户画像记忆，自动提取 + 注入 + 衰减，越用越懂你
- 🔧 **AIOps 智能诊断** — Plan-Execute-Replan 三节点协作，自动故障诊断和根因分析
- 🌐 **Web 界面** — 现代化 UI，支持快速问答 / 流式对话两种模式，Markdown 渲染 + 代码高亮
- 🔌 **MCP 工具集成** — CLS 日志查询 + Monitor 系统监控，streamable-http 协议
- 🔐 **JWT 双 Token 鉴权** — access_token + refresh_token，bcrypt 密码加密，SQLite 用户存储
- 🐳 **Docker 一键部署** — Docker Compose 编排全套服务，Caddy 自动 HTTPS

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| 框架 | FastAPI + LangChain + LangGraph |
| LLM | 阿里云 DashScope（通义千问 qwen-max） |
| Embedding | text-embedding-v4（1024 维） |
| 向量库 | Milvus 2.5（IVF_FLAT 索引） |
| 缓存 / 会话 | Redis 7（主存） + SQLite（冷备） |
| 工具协议 | MCP (Model Context Protocol) |
| 部署 | Docker Compose + Caddy（自动 HTTPS） |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker Desktop（运行 Milvus + Redis）
- 阿里云 DashScope API Key（[获取地址](https://dashscope.aliyun.com/)）

### 安装和启动

#### Linux/macOS 环境

```bash
# 1. 克隆项目
git clone git@github.com:CatB2/LMH.git
cd LMH

# 2. 安装依赖（推荐使用 uv）
# 方式 1: 使用 uv（推荐，更快）
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 方式 2: 使用 pip
pip install -e .

# 3. 启动基础设施（Milvus + Redis）
docker compose -f vector-database.yml up -d
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 4. 编辑配置文件
# 首次使用需要编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
vim .env

# 5. 启动 MCP 服务（需要两个终端）
python mcp_servers/cls_server.py      # 端口 8003
python mcp_servers/monitor_server.py  # 端口 8004

# 6. 启动主服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 7. 上传知识库文档（可选）
for f in aiops-docs/*.md; do
  curl -X POST http://localhost:9900/api/upload -F "file=@$f"
done
```

#### Windows 环境（PowerShell/CMD）

如果 Windows 不支持 `make` 命令，可以手动执行以下步骤以启动服务：

```powershell
# 1. 克隆项目
git clone git@github.com:CatB2/LMH.git
cd LMH

# 2. 创建虚拟环境并安装依赖
# 方式 1: 使用 uv（推荐，更快）
pip install uv
uv venv
.venv\Scripts\activate
uv pip install -e .

# 方式 2: 使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 编辑配置文件
# 使用记事本或其他编辑器打开 .env 文件，填入你的 DASHSCOPE_API_KEY
notepad .env

# 4. 启动 Docker Desktop
# 确保 Docker Desktop 已安装并正在运行

# 5. 启动 Milvus 向量数据库（Docker Compose）
docker compose -f vector-database.yml up -d

# 6. 启动 Redis 缓存
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 7. 等待 Milvus 启动完成（约 5-10 秒）
timeout /t 10

# 8. 启动 MCP 服务
# 启动 CLS 日志查询服务（新开一个 PowerShell 窗口）
python mcp_servers/cls_server.py

# 启动 Monitor 监控服务（新开一个 PowerShell 窗口）
python mcp_servers/monitor_server.py

# 9. 启动 FastAPI 主服务（新开一个 PowerShell 窗口）
# 注意：日志会自动输出到 logs\app_YYYY-MM-DD.log
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 10. 上传文档到向量库（新开一个 PowerShell 窗口）
# 等待服务启动完成后执行
timeout /t 5
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"
```

**Windows 一键启动脚本**（推荐）

使用启动脚本：

```powershell
# 启动所有服务
.\start-windows.bat

# 停止所有服务
.\stop-windows.bat
```

### 访问服务

- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs
- **健康检查**: http://localhost:9900/health

## 📡 API 接口

### 认证接口（无需登录）

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 用户注册 | POST | `/api/auth/register` | bcrypt 密码加密 |
| 用户登录 | POST | `/api/auth/login` | 返回双 token |
| 刷新 Token | POST | `/api/auth/refresh` | 用 refresh_token 换新 access_token |
| 登出 | POST | `/api/auth/logout` | 吊销 refresh_token |

### 核心接口（需认证，Header 带 `Authorization: Bearer <access_token>`）

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| 清空会话 | POST | `/api/chat/clear` | 清空指定会话历史 |
| 会话历史 | GET | `/api/chat/session/{id}` | 查询指定会话消息 |
| Agent 列表 | GET | `/api/chat/agents` | 获取可用 Agent |
| 用户会话 | GET | `/api/sessions` | 列出用户全部会话 |
| 文件上传 | POST | `/api/upload` | 上传文档到知识库（自动索引） |
| AIOps 诊断 | POST | `/api/aiops` | 智能故障诊断（SSE 流式） |

### 公开接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | GET | `/health` | 服务 + Milvus 连通性 |

### 使用示例

```bash
# 注册
curl -X POST "http://localhost:9900/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"123456"}'

# 登录（获取 token）
curl -X POST "http://localhost:9900/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"123456"}'

# 普通对话（带认证）
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

# 健康检查（无需认证）
curl http://localhost:9900/health
```

## 📁 项目结构

```
LMH/
├── app/                                    # 应用核心
│   ├── __init__.py                         # 包初始化（自动加载日志配置）
│   ├── main.py                             # FastAPI 应用入口（lifespan 管理）
│   ├── config.py                           # 配置管理（环境变量、MCP 服务器配置）
│   ├── api/                                # API 路由层
│   │   ├── __init__.py
│   │   ├── chat.py                         # 对话接口（普通 + 流式）
│   │   ├── aiops.py                        # AIOps 诊断接口（SSE 流式）
│   │   ├── auth.py                         # 认证接口（注册/登录/刷新/登出）
│   │   ├── file.py                         # 文件管理（上传 + 自动索引）
│   │   └── health.py                       # 健康检查（服务 + Milvus 状态）
│   ├── services/                           # 业务服务层
│   │   ├── __init__.py
│   │   ├── rag_agent_service.py            # RAG Agent 编排（LangGraph）
│   │   ├── aiops_service.py                # AIOps 服务（Plan-Execute-Replan）
│   │   ├── auth_service.py                 # JWT 双 Token + bcrypt 鉴权
│   │   ├── session_memory_store.py         # Redis 主存 + SQLite 冷备 + 自愈
│   │   ├── context_builder.py              # 上下文压缩（Token 管理 + 摘要）
│   │   ├── memory_manager.py               # 长期记忆注入/提取/衰减
│   │   ├── vector_store_manager.py         # 向量存储 LangChain 封装
│   │   ├── vector_embedding_service.py     # DashScope Embeddings
│   │   ├── vector_index_service.py         # 文件索引流水线（分块→嵌入→入库）
│   │   ├── vector_search_service.py        # pymilvus 原始检索
│   │   └── document_splitter_service.py    # 文档分割（双窗口策略）
│   ├── agent/                              # Agent 模块
│   │   ├── __init__.py
│   │   ├── agent_manager.py                # 多 Agent 路由调度
│   │   ├── rag_agent.py                    # RAG Agent 定义 + MCP 工具加载
│   │   ├── mcp_client.py                   # MultiServerMCPClient 管理（重试拦截器）
│   │   └── aiops/                          # AIOps Plan-Execute-Replan
│   │       ├── __init__.py
│   │       ├── planner.py                  # 计划制定器（LLM 生成步骤）
│   │       ├── executor.py                 # 步骤执行器（工具调用）
│   │       ├── replanner.py                # 重规划器（评估→继续/调整/报告）
│   │       ├── state.py                    # PlanExecuteState 类型定义
│   │       └── utils.py                    # 工具描述格式化
│   ├── core/                               # 核心组件
│   │   ├── __init__.py
│   │   ├── llm_factory.py                  # LLM 工厂（ChatOpenAI 统一创建）
│   │   ├── milvus_client.py                # Milvus 客户端管理器（连接池）
│   │   ├── intent_detector.py              # 意图检测（关键词 + LLM 双层）
│   │   └── user_memory.py                  # 长期记忆存储（Milvus user_memory 集合）
│   ├── tools/                              # Agent 工具集
│   │   ├── __init__.py
│   │   ├── knowledge_tool.py               # 知识库查询（向量 + BM25 RRF 混合）
│   │   └── time_tool.py                    # 当前时间工具
│   ├── models/                             # Pydantic 数据模型
│   │   ├── __init__.py
│   │   ├── aiops.py                        # AIOps 请求/响应模型
│   │   ├── document.py                     # 文档分块模型
│   │   ├── request.py                      # 请求模型（含认证）
│   │   └── response.py                     # 统一响应格式
│   └── utils/                              # 工具类
│       ├── __init__.py
│       ├── logger.py                       # Loguru 配置（控制台 + 文件轮转）
│       └── response.py                     # 统一 JSONResponse 工具
├── static/                                 # Web 前端（纯静态，原生 JS）
│   ├── index.html                          # 主页面
│   ├── app.js                              # 前端逻辑（SSE 解析 + Markdown 渲染）
│   └── styles.css                          # Google Material Design 风格
├── mcp_servers/                            # MCP 服务器（独立进程）
│   ├── cls_server.py                       # CLS 日志查询服务（端口 8003）
│   ├── monitor_server.py                   # 系统监控服务（端口 8004）
│   └── README.md                           # MCP 服务说明
├── aiops-docs/                             # 运维知识库文档（5 篇 Markdown）
│   ├── cpu_high_usage.md                   # CPU 高负载排查
│   ├── memory_high_usage.md                # 内存高负载排查
│   ├── disk_high_usage.md                  # 磁盘高负载排查
│   ├── slow_response.md                    # 服务响应慢排查
│   └── service_unavailable.md              # 服务不可用排查
├── tests/                                  # 单元测试
│   ├── __init__.py
│   ├── conftest.py                         # Pytest 配置
│   └── test_fine_tuning_service.py         # 微调服务测试
├── data/                                   # 运行时数据目录
│   └── session_memory.db                   # SQLite 用户 + 会话 + Token 存储
├── logs/                                   # 日志目录（Loguru 自动创建）
│   └── app_YYYY-MM-DD.log                  # 按天轮转的日志文件
├── uploads/                                # 上传文件临时目录
├── volumes/                                # Milvus 数据持久化目录
├── .env                                    # 环境变量配置（需手动创建，不提交）
├── .env.docker                             # Docker 部署配置模板
├── Dockerfile                              # 应用容器镜像
├── docker-compose.yml                      # 生产环境全套编排
├── Caddyfile                               # HTTPS 反向代理（自动证书）
├── Makefile                                # 项目管理命令（Linux/macOS）
├── start-windows.bat                       # Windows 一键启动脚本
├── stop-windows.bat                        # Windows 一键停止脚本
├── vector-database.yml                     # Milvus Docker Compose 配置
├── pyproject.toml                          # 项目配置（依赖、元数据、工具链）
├── uv.lock                                 # uv 依赖锁定文件
├── pyrightconfig.json                      # Pyright 类型检查配置
└── README.md                               # 项目说明
```

## ⚙️ 配置说明

通过 `.env` 文件配置，也可以参考 `.env.docker`（Docker 部署用）：

```bash
# LLM 配置（必填）
# 秘钥管理：https://bailian.console.aliyun.com/
DASHSCOPE_API_KEY=sk-your-api-key          # 阿里云百炼 API Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max                   # 对话模型
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4  # 嵌入模型
DASHSCOPE_EMBEDDING_API_KEY=sk-your-api-key  # 嵌入 API Key（可与上面相同）

# Milvus 向量数据库
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TIMEOUT=10000

# Redis 缓存
REDIS_ENABLED=True
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_CACHE_TTL_DAYS=7

# RAG 检索
RAG_TOP_K=20
RAG_MODEL=qwen-max

# 文档分块
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100

# MCP 服务
MCP_CLS_TRANSPORT=streamable-http
MCP_CLS_URL=http://localhost:8003/mcp
MCP_MONITOR_TRANSPORT=streamable-http
MCP_MONITOR_URL=http://localhost:8004/mcp

# JWT 鉴权（生产环境务必修改）
JWT_SECRET_KEY=change-me-to-random-string
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# 腾讯云 CLS（可选）
CLS_SECRET_ID=
CLS_SECRET_KEY=
CLS_REGION=ap-beijing
CLS_TOPIC_ID=
```

## 🎯 多智能体协同架构

### RAG 知识库 Agent

```
用户提问 → 意图检测 → 知识库检索（BM25 + 向量 RRF） → LLM 生成 → 流式输出
                                ↑
                    长期记忆注入（用户画像 + 相关记忆）
                                ↓
                    新知识提取 → 记忆衰减机制
```

**检索策略**：
- **BM25** — 关键词精确匹配，擅长术语检索
- **向量检索** — 语义相似度，擅长同义表达
- **RRF 融合** — Reciprocal Rank Fusion，综合两种策略的优点

### AIOps 诊断 Agent — Plan-Execute-Replan

基于 LangGraph 的三节点协作：

```
1. Planner 制定计划 → 生成 4-6 个诊断步骤
2. Executor 执行步骤 → 调用 MCP 工具（日志查询、监控数据）
3. Replanner 评估结果 → 决定继续 / 调整计划 / 生成报告
4. 输出诊断报告 → 根因分析 + 运维建议
```

**可用工具**：
| 工具 | 来源 | 用途 |
|------|------|------|
| `retrieve_knowledge` | 本地 | 查询运维知识库 |
| `get_current_time` | 本地 | 获取当前时间 |
| `search_logs` | MCP CLS | 查询系统日志 |
| `query_metrics` | MCP Monitor | 获取 CPU/内存/磁盘指标 |

### 快速测试

```bash
# 访问 Web 界面，点击「智能运维与诊断工具」
# 或使用 API
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{}' \
  --no-buffer
```

## 🐳 Docker 部署

```bash
# 拉取项目
git clone git@github.com:CatB2/LMH.git
cd LMH

# 修改配置
nano .env.docker
# 必改：DASHSCOPE_API_KEY、JWT_SECRET_KEY

# 一键启动（首次构建约 5-10 分钟）
docker compose --env-file .env.docker up -d --build

# 查看状态
docker compose ps
curl http://localhost:9900/health
```

详细部署文档见 [DEPLOY.md](./DEPLOY.md)

## 📝 开发指南

### 常用命令

```bash
# 项目管理
make init              # 一键初始化（Docker + 服务 + 文档）
make start             # 启动所有服务
make stop              # 停止所有服务
make restart           # 重启所有服务

# 依赖管理
make install-dev       # 安装开发依赖
make sync              # 同步依赖

# Docker 管理
make up                # 启动 Docker 容器
make down              # 停止 Docker 容器

# 代码质量
make format            # 格式化代码
make lint              # 代码检查
```

## 🐛 常见问题

### Windows 环境问题

#### 1. `make` 命令不可用

Windows 不支持 `make` 命令，请使用提供的批处理脚本：

```powershell
.\start-windows.bat   # 启动服务
.\stop-windows.bat    # 停止服务
```

#### 2. PowerShell 执行策略限制

如果遇到 "无法加载文件，因为在此系统上禁止运行脚本" 错误：

```powershell
# 临时允许脚本执行（管理员权限）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 或者使用 CMD 而不是 PowerShell
cmd
.\start-windows.bat
```

#### 3. 端口被占用（Windows）

```powershell
# 查看占用端口的进程
netstat -ano | findstr :9900

# 结束进程（替换 PID 为实际进程 ID）
taskkill /F /PID <PID>
```

### 通用问题

#### API Key 错误

```bash
# 检查环境变量
cat .env | grep DASHSCOPE_API_KEY    # Linux/macOS
type .env | findstr DASHSCOPE_API_KEY  # Windows
```

#### Milvus 连接失败

```bash
# 确保本机有 Docker 服务并且已经启动（可以使用 Docker Desktop）

# 检查 Milvus 状态
docker ps | grep milvus

# 重启 Milvus（使用 docker compose）
docker compose -f vector-database.yml restart

# 或者重启单个服务
docker compose -f vector-database.yml restart standalone
```

#### Redis 连接失败

```bash
# 确保 Redis 容器在运行
docker ps | grep redis

# 如果没有，启动 Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

#### 服务无法启动

**Linux/macOS:**

```bash
# 查看服务日志
tail -f logs/app_$(date +%Y-%m-%d).log  # FastAPI 主服务（Loguru 日志）
tail -f mcp_cls.log                      # CLS MCP 服务
tail -f mcp_monitor.log                  # Monitor MCP 服务

# 检查端口占用
lsof -i :9900  # FastAPI
lsof -i :8003  # CLS MCP
lsof -i :8004  # Monitor MCP
```

**Windows:**

```powershell
# 查看服务日志（获取今天的日期）
$today = Get-Date -Format "yyyy-MM-dd"
type logs\app_$today.log  # FastAPI 主服务（Loguru 日志）
type mcp_cls.log          # CLS MCP 服务
type mcp_monitor.log      # Monitor MCP 服务

# 或者查看最新的日志文件
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50

# 检查端口占用
netstat -ano | findstr :9900  # FastAPI
netstat -ano | findstr :8003  # CLS MCP
netstat -ano | findstr :8004  # Monitor MCP
```

## 📚 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph Plan-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [阿里云 DashScope](https://dashscope.aliyun.com/)
- [MCP 协议](https://modelcontextprotocol.io/)

## 📄 许可证

author： chief

MIT License
