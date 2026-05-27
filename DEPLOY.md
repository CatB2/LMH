# SuperBizAgent 部署指南

支持两种部署方式：**Docker 一键部署**（推荐生产）和**本地手动启动**（开发调试）。

---

## 方式一：Docker Compose 部署（推荐）

### 1. 准备服务器

- 系统：Ubuntu 22.04 / CentOS 7+
- 配置：**4核8G 起步**（Milvus + PyTorch 内存开销较大）
- 开放端口：80、443（如需 SSH 还开放 22）

### 2. 安装 Docker

```bash
curl -fsSL https://get.docker.com | bash
apt install docker-compose-plugin -y
```

### 3. 拉取项目

```bash
git clone git@github.com:CatB2/LMH.git
cd LMH
```

### 4. 修改配置

```bash
nano .env.docker
```

必改项：
```
DASHSCOPE_API_KEY=你的百炼API密钥
JWT_SECRET_KEY=随机生成一个长字符串（生产环境务必修改）
```

### 5. 启动

```bash
docker compose --env-file .env.docker up -d --build
```

首次构建约 5-10 分钟（下载镜像 + 安装 PyTorch/Transformers）。启动后：

```bash
# 检查状态
docker compose ps
curl http://localhost:9900/health

# 查看日志
docker compose logs -f app
```

### 6. 配置域名（可选）

如果你有域名（如 `liumeihua.com`），编辑 `Caddyfile` 改为你的域名：

```
your-domain.com {
    reverse_proxy app:9900
}
```

然后将域名的 DNS A 记录指向服务器公网 IP。Caddy 会自动申请和续期 Let's Encrypt HTTPS 证书。

### Docker 架构

```
Caddy (:80/443) → app (:9900) → Redis (:6379)
                              → Milvus (:19530)
                              → MCP CLS (:8003)
                              → MCP Monitor (:8004)
```

### 运维命令

```bash
docker compose ps                    # 查看服务状态
docker compose logs -f app           # 查看应用日志
docker compose restart app           # 重启应用
docker compose down                  # 停止全部
git pull && docker compose up -d --build app  # 更新代码
```

---

## 方式二：本地手动启动（开发）

### 环境

- Python 3.11+
- Docker Desktop（运行 Milvus + Redis）

### 步骤

```powershell
# 1. 克隆
git clone git@github.com:CatB2/LMH.git
cd LMH

# 2. 安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 编辑 .env，填入 API Key

# 4. 启动基础设施
docker compose -f vector-database.yml up -d
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 5. 启动 MCP 服务（两个终端窗口）
python mcp_servers/cls_server.py
python mcp_servers/monitor_server.py

# 6. 启动主服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

访问 http://localhost:9900

---

## 注意事项

1. `.env` 含 API Key，**不要提交到 Git**（已在 .gitignore）
2. `.env.docker` 是 Docker 部署的配置模板，不含真实密钥
3. Caddy 自动管理 HTTPS 证书，无需手动操作
4. Milvus 数据存在 Docker Volume 中，`docker compose down` 不会丢失
5. 生产环境的 `JWT_SECRET_KEY` 务必改成随机强密码
