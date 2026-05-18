# K8s Smart Diagnosis Operator

基于大语言模型的多Agent编排的 Kubernetes 智能排障系统。通过 4 个专用 Agent（调度/查询/诊断/巡检）协作，自动完成集群故障的意图识别、数据采集、多轮推理诊断和定时巡检。

---

## 快速开始

⚠️ 注意 / 免责声明

提示词尚在优化中：当前 查询 Agent 与 诊断 Agent 的提示词仍需进一步调优，在复杂故障场景下可能存在推理路径偏差。

性能与成本考量：由于故障排查流程涉及多次 LLM 调用（多 Agent 编排、多轮推理），单次排障耗时较长、Token 消耗较高，与运维场景中“快速定位故障”的初衷存在一定背离。


```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM API Key（编辑 config/llm_config.json，填入 api_key）
# 3. 启动服务
python main.py
```

服务启动后，用 curl 测试：

```bash
# 发送诊断请求（将 <your_message> 替换为你的问题）
curl -X POST http://127.0.0.1:1314/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我查一下 default 命名空间下所有 Pod 的状态", "session_id": "test-001"}'
```

> ⚠️ **启动前必做**：编辑 `config/llm_config.json`，在 `api_key` 字段填入你的 LLM API Key（支持火山引擎等 OpenAI 兼容 API）。

---

## 核心特性

### 🤖 四 Agent 协作

| Agent | 职责 | 核心技术 |
|:---|:---|:---|
| **调度 Agent (Dispatcher)** | 意图识别（7 种）、任务分发、多结果整合 | LLM 推理 + asyncio 并行调度 |
| **查询 Agent (Query)** | K8s 集群实时数据查询 | MCP 协议调用 K8s API + Prometheus |
| **诊断 Agent (Diagnosis)** | 多轮循环推理，逐步收敛根因 | 3 轮策略选择 → 并行采集 → 推理判断 |
| **巡检 Agent (Inspector)** | 定时集群巡检，生成结构化报告 | 30 分钟定时器 → MCP 工具采集 → LLM 整合 |

### 🔧 17 个 MCP 工具

通过 MCP（Model Context Protocol）统一调用：

- **K8s 查询（11 个）：** 节点/Pod/命名空间查询、事件查看、控制器管理、容器日志、资源用量 top
- **Prometheus 监控（6 个）：** 即时查询、范围查询、节点/Pod CPU/内存使用率趋势

### 🧠 RAG 知识库

- 基于 ChromaDB + LlamaIndex 的持久化向量存储
- 两个独立知识库：**巡检报告** 和 **工作记录**
- 启动自动去重注入，运行时实时写入与检索
- 支持按日期范围、标题关键词过滤

### 💬 会话管理

- 持久化对话历史，支持多会话隔离
- 自动创建新会话，LLM 可识别 `new_session` 意图
- 会话上下文传递给下游 Agent，实现连续对话

### 📊 日志体系

```
logs/agent/
├── all.log            # 完整日志（所有 Agent 汇总）
├── dispatcher.log     # 调度 Agent 日志
├── query.log          # 查询 Agent 日志
├── inspector.log      # 巡检 Agent 日志
└── diagnosis.log      # 诊断 Agent 日志
```

支持运行时动态切换 DEBUG 级别：

```python
from logger import set_agent_debug
set_agent_debug("query")    # 只看查询 Agent 的 debug 日志
set_agent_debug()           # 所有 Agent 开启 debug
set_agent_info("query")     # 恢复为 INFO
```

### ⚡ 优雅关闭

Ctrl+C 时系统会：
1. 停止巡检定时器
2. 等待正在进行的文件写入完成（最多 30 秒）
3. 安全退出，不丢失数据

---

## 系统架构

```
用户请求
    │
    ▼
┌──────────────────┐
│  调度 Agent      │  ← 意图识别（7种）+ 任务分发 + 结果整合
│  DispatcherAgent │
└───────┬──────────┘
        │
        ▼ 并行执行
┌───────┴────────┐  ┌───────────────┐  ┌───────────────┐
│  查询 Agent    │  │  诊断 Agent   │  │  巡检 Agent   │
│  QueryAgent    │  │ DiagnosisAgent│  │ InspectorAgent│
├────────────────┤  ├───────────────┤  ├───────────────┤
│ MCP 协议调用   │  │ 3轮循环推理   │  │ 每30分钟      │
│ K8s API(11个)  │  │ 策略选择→采集 │  │ 自动巡检      │
│ Prometheus(6个)│  │ →推理判断→收敛 │  │ 生成报告      │
└────────────────┘  └───────────────┘  └───────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  RAG 知识库   │
                                        │  ChromaDB     │
                                        │  检索+持久化   │
                                        └──────────────┘

内嵌服务:
  HTTP API : 0.0.0.0:1314 (接收用户请求)
  MCP Server: 127.0.0.1:1315 (Agent 内部调用工具)
```

---

## API 接口

### POST /chat

```bash
curl -X POST http://127.0.0.1:1314/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "最近集群不稳定，帮我排查一下",
    "session_id": "optional-session-id"
  }'
```

**响应格式：**

```json
{
  "reply": "正在为您诊断集群...",
  "session_id": "2026-05-18-16:13",
  "work_record_file": "2026-05-18-16:13-xxx工作记录.md"
}
```

### GET /healthz

```bash
curl http://127.0.0.1:1314/healthz
# 返回: ok
```

---

## 7 种意图识别

调度 Agent 支持以下操作类型，可多意图同时激活：

| 意图 | 触发场景 | 调度目标 |
|:---|:---|:---|
| `new_session` | 用户要求重置/新对话 | 创建新会话 |
| `chat` | 闲聊、寒暄、感谢 | 纯 LLM 聊天 |
| `simple_query` | 查 Pod/节点/事件/日志/监控指标 | 查询 Agent |
| `complex_diagnosis` | 分析故障根因、多维度诊断 | 诊断 Agent |
| `inspection` | 要求生成巡检报告 | 巡检 Agent |
| `rag_work_record` | 查询历史工作记录/诊断档案 | RAG 知识库 |
| `rag_inspection` | 查询历史巡检报告 | RAG 知识库 |

---

## 项目结构

```
├── agents/
│   ├── dispatcher_agent.py    # 调度 Agent — 意图识别/任务分发/结果整合
│   ├── diagnosis_agent.py     # 诊断 Agent — 多轮推理诊断
│   ├── inspector_agent.py     # 巡检 Agent — 定时集群巡检
│   └── query_agent.py         # 查询 Agent — K8s 数据查询
├── tools/
│   ├── k8s_tools.py           # 11 个 K8s 查询工具
│   ├── prometheus_tools.py    # 6 个 PromQL 工具
│   ├── mcp_server.py          # MCP Server 统一入口（注册 17 个工具）
│   └── mcp_utils.py           # MCP 客户端工具（发现/调用）
├── utils/
│   ├── llm.py                 # LLM 模型工厂（AgentScope）
│   ├── llm_utils.py           # LLM 响应解析工具
│   ├── rag.py                 # ChromaDB 知识库管理
│   ├── rag_retrieval.py       # RAG 检索封装
│   ├── conversation.py        # 会话管理
│   ├── async_utils.py         # 异步并行执行封装
│   ├── write_guard.py         # 文件写入防并发
│   └── task_context.py        # Agent 间任务上下文
├── config/
│   ├── llm_config.json        # LLM 模型配置（填写 API Key）
│   ├── ChromaDB/              # 向量数据持久化目录
│   ├── conversations/         # 会话历史 JSON 文件
│   ├── inspector_reports/     # 巡检报告 .md 文件
│   └── work_records/          # 诊断工作记录 .md 文件
├── config.py                  # 项目配置常量
├── logger.py                  # 5 文件日志系统
├── main.py                    # HTTP 服务入口（aiohttp）
├── requirements.txt           # Python 依赖
└── .gitignore
```

---

## 配置说明

### LLM 配置

编辑 `config/llm_config.json`：

```json
{
  "llm": {
    "dispatcher": {
      "api_key": "your-api-key-here",
      "model_name": "Doubao-Seed-2.0-pro",
      "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
      "temperature": 0.3
    }
  },
  "embedding": {
    "inspection_reports": {
      "model": {
        "api_key": "your-api-key-here",
        "model_name": "doubao-embedding-vision",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3"
      }
    }
  }
}
```

> 支持任意 OpenAI 兼容 API，更换地址和模型名即可。

### 环境变量

| 变量 | 用途 | 默认值 |
|:---|:---|:---|
| `PROMETHEUS_URL` | Prometheus 地址 | 集群内 DNS 自动探测 |

---

## 技术栈

| 技术 | 用途 |
|:---|:---|
| Python 3.11+ | 核心开发语言 |
| AgentScope | LLM 模型接入框架 |
| aiohttp | HTTP 异步 Web 服务 |
| MCP / FastMCP | 工具协议（17 个集群工具） |
| ChromaDB + LlamaIndex | 向量知识库 |
| Kubernetes SDK | K8s API 调用 |
| Prometheus API Client | PromQL 查询 |
| 火山引擎 Doubao | LLM + Embedding 模型 |

---

## 许可证

MIT
