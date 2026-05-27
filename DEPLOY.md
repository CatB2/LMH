# SuperBizAgent 阿里云 ECS 部署指南

## 前置条件

- 阿里云账号（已购买 `liumeihua.com` 域名）
- 一台 ECS 实例（**4核8G 起步**，推荐 Ubuntu 22.04，按量/包月均可）
- 安全组开放端口：**22** (SSH)、**80** (HTTP)、**443** (HTTPS)

---

## 第一步：购买 ECS + 配置安全组

1. 打开 [阿里云 ECS 控制台](https://ecs.console.aliyun.com)
2. 创建实例：
   - 镜像：Ubuntu 22.04
   - 规格：4 vCPU / 8 GiB 内存
   - 系统盘：40 GiB
   - 勾选 "分配公网 IPv4"
3. 在安全组规则中添加：
   | 方向 | 端口 | 协议 | 来源 |
   |------|------|------|------|
   | 入方向 | 22 | TCP | 0.0.0.0/0 |
   | 入方向 | 80 | TCP | 0.0.0.0/0 |
   | 入方向 | 443 | TCP | 0.0.0.0/0 |

> 记下 ECS 的**公网 IP**（例如 `47.xx.xx.xx`）。

---

## 第二步：SSH 登录 ECS，安装 Docker

```bash
ssh root@<你的ECS公网IP>

# 安装 Docker
curl -fsSL https://get.docker.com | bash

# 安装 Docker Compose
apt install docker-compose-plugin -y

# 验证
docker --version
docker compose version
```

---

## 第三步：上传项目到 ECS

**方式 A：从本机直接 scp 上传**
```bash
# 在你的 Windows 终端（PowerShell）
scp -r D:\AIModel\super_biz_agent_py-release-2026-03-21 root@<ECS_IP>:/opt/superbizagent
```

**方式 B：用 git 仓库（如果代码已在 git）**
```bash
# 在 ECS 上
cd /opt
git clone <你的仓库地址> superbizagent
```

---

## 第四步：修改配置

在 ECS 上编辑配置文件：

```bash
cd /opt/superbizagent
```

**4.1 修改 `.env.docker`**：
```bash
nano .env.docker
```
重点修改这 3 项：
```
DASHSCOPE_API_KEY=你的百炼API密钥
JWT_SECRET_KEY=<随机生成一个长字符串>
```
其余保持默认即可。

**4.2 修改 `Caddyfile`**（如果域名不同）：
确认里面的域名是 `liumeihua.com`。

---

## 第五步：启动服务

```bash
cd /opt/superbizagent

# 构建并启动全部服务
docker compose --env-file .env.docker up -d --build

# 查看启动日志
docker compose logs -f

# 看到这一行说明成功：
# sba-app | Uvicorn running on http://0.0.0.0:9900
```

首次构建约需 **5-10 分钟**（安装 PyTorch/Transformers 等依赖）。

---

## 第六步：配置域名 DNS

1. 打开 [阿里云域名控制台](https://dc.console.aliyun.com)
2. 找到 `liumeihua.com` → **解析设置**
3. 添加两条记录：

| 记录类型 | 主机记录 | 记录值 |
|----------|----------|--------|
| A | @ | `<ECS 公网 IP>` |
| A | www | `<ECS 公网 IP>` |

4. 等待 DNS 生效（通常 1-10 分钟）

---

## 第七步：验证

```bash
# 检查健康状态
curl http://localhost:9900/health

# 用域名访问（DNS 生效后）
curl https://liumeihua.com/health

# Caddy 会自动申请 Let's Encrypt HTTPS 证书
```

浏览器打开 `https://liumeihua.com` 即可看到 Web 界面。

---

## 常用运维命令

```bash
# 查看所有服务状态
docker compose ps

# 查看某个服务的日志
docker compose logs -f app
docker compose logs -f caddy

# 重启某个服务
docker compose restart app

# 更新代码后重新部署
git pull                          # 如果用了 git
docker compose up -d --build app  # 重建 app 相关服务

# 停止所有服务
docker compose down

# 备份数据
tar -czf backup.tar.gz app_data/ app_uploads/ redis_data/ milvus_data/
```

---

## 注意事项

1. **JWT_SECRET_KEY** — 上线前务必改成随机的强密码
2. **DASHSCOPE_API_KEY** — 不要提交到公开仓库
3. **Milvus 数据** — 默认存在 Docker volume 中，`.env.docker` 中的 API 模型变更后可能需要重建索引
4. **HTTPS 证书** — Caddy 自动管理，无需手动续期
5. **4C8G 最低配置** — Milvus + PyTorch 内存占用较高，低于此配置可能 OOM
