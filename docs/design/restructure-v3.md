# AuraForge 重构方案 v3

## 1. 定位

v2 的方向是对的，但动作过大。

当前项目的核心问题不是目录结构，而是三件事：

1. 用户入口不清晰：`chat`、`dispatch`、`committee__submit` 混在一起。
2. MCP 生命周期不合理：会话启动时预连接，噪声进入终端。
3. Committee 自动化闭环不够硬：部分阶段只存在于 prompt 中，没有清晰的运行时落点。

因此，v3 的原则是：

- 先修产品路径，再修自动化闭环，最后才整理包结构。
- 先做减法，不新增抽象层，不一次性大搬家。
- 任何重构都必须服务于 `submit -> daemon -> watch -> delivery` 这条主路径。

---

## 2. v3 设计原则

1. **不为整理而整理。** 目录重组不能优先于闭环修复。
2. **保留现有运行骨架。** `ControlHub`、`AgentRunner`、`Dispatcher`、`Engine` 先不拆。
3. **提交入口去 LLM。** 项目级任务直接写入 Committee 信箱。
4. **观察入口唯一化。** 用户只看 `watch`，不看零散状态命令拼图。
5. **运行时行为必须有代码落点。** Prompt 可以驱动 agent，但关键阶段切换不能只靠 prompt。
6. **兼容优先。** 先保留 `OpKind.CHAT`、`subagent__run`、`runtime/` 包，避免无收益迁移。

---

## 3. 明确保留什么

这些模块先视为稳定骨架，不在 v3 首阶段重构：

- `runtime/control/hub.py`
- `runtime/control/runner.py`
- `runtime/control/dispatcher.py`
- `runtime/engine.py`
- `runtime/engine_agno_async.py`
- `runtime/capability/`
- `runtime/context/`
- `runtime/sandbox/`
- `runtime/signal/`
- `runtime/event_log/`
- `runtime/registry/`

理由：

- 它们已经形成真实运行链路。
- 当前痛点不在这里的模块边界，而在入口设计和若干关键流程未接线。
- 先搬目录只会扩大回归面。

---

## 4. 明确暂不删除什么

v2 中以下删除项，v3 暂缓：

### 4.1 `Dispatcher`

不能删。

当前 `TASK_ASSIGNED -> 创建 sandbox -> 发信号` 的硬逻辑在 `runtime/control/dispatcher.py`。
在出现等价替代路径前，`Dispatcher` 是必需运行部件，而不是历史包袱。

### 4.2 `subagent__run` 与 `subagents/`

不能立即删。

当前引擎仍真实注册并处理 `subagent__run`，包括 approval passthrough 和 resume 流程。它不是孤儿代码。

### 4.3 `OpKind.CHAT`

不能立即改名。

虽然语义上更像 “agent turn”，但 Runner 当前就是通过 `OpKind.CHAT` 驱动 session。v3 先保兼容，不做协议迁移。

### 4.4 `runtime/` 顶层包

不能第一阶段拆散。

只要主路径还没稳定，目录重组就是额外风险源。

---

## 5. v3 真正要解决的问题

### 5.1 用户交互统一

目标：用户只需要理解三条路径。

- `aura submit`：提交项目任务
- `aura daemon`：启动/停止自动化引擎
- `aura watch`：观察全过程

`aura chat` 退居次入口，仅保留：

- 快速问答
- 小范围代码阅读
- 小改动

它不再承担“提交任务给市场”的主入口职责。

### 5.2 MCP 生命周期收敛

目标：

- 会话启动时不预连接全部 MCP
- MCP stderr 不进入用户终端
- 连接建立和工具发现拆开处理

v3 不要求立刻做复杂连接池，只做两件事：

1. stderr 落到 `.aura/logs/mcp_<server>.log`
2. 仅在真正要执行该 server 工具时建立连接

### 5.3 Committee 闭环硬接线

当前问题：

- `project_request`、`bid_check`、`task_completed` 虽然能进入 Committee prompt，
  但部分阶段仍缺少明确的运行时 side effect。

v3 要补齐三段：

1. `submit -> committee`
2. `bid_check -> collect/evaluate -> TASK_ASSIGNED`
3. `task_completed -> verify -> delivery/notification`

这里的重点不是“让 LLM 更聪明”，而是：

- 哪个阶段由谁触发
- 谁负责持久化结果
- 谁负责发出下一条信号

---

## 6. v3 目标架构

v3 不是新的包结构，而是新的产品路径：

```text
User
  ├── aura submit    -> SignalBus -> Committee
  ├── aura daemon    -> ControlHub
  └── aura watch     -> Dashboard snapshot

ControlHub
  └── AgentRunner
        ├── Committee session
        └── Worker sessions

Committee runtime responsibilities
  ├── decompose request
  ├── wake candidates
  ├── evaluate bids
  ├── assign worker via Dispatcher
  └── verify completion and notify delivery
```

重点：

- 结构上仍复用现有 `runtime/`
- 行为上改成明确的单一路径

---

## 7. CLI 策略

### 7.1 `aura submit`

新增，作为唯一任务提交入口。

行为：

- 不走 LLM
- 直接构造 `project_request` payload
- 通过 `SignalBus.send(... to_agent="committee")` 写入 Committee 信箱
- 若 daemon 未运行，明确提示用户执行 `aura daemon start`

### 7.2 `aura daemon`

保留现有能力，只做轻整理：

- `start`
- `stop`
- `status`

不改内部机制，不额外发明新控制面。

### 7.3 `aura watch`

新增，作为唯一推荐观察入口。

第一版只要求：

- daemon 状态
- 活跃 issue 列表
- 活跃 agent 列表
- 最近 signal log

先不要做很重的 TUI 交互设计。第一版可以先是定时刷新文本视图或轻量 `rich` 表格。

### 7.4 `aura chat`

降级为辅助入口。

保留原因：

- 对代码库问答仍有价值
- 做小改动仍有价值
- 调试引擎仍有价值

但它不再是市场主入口。

---

## 8. Committee 闭环设计

### 8.1 Decompose

输入：`project_request`

输出：

- Linear project / issue
- 候选 agent WAKE 信号
- request/activity 记录

这里可以继续主要由 prompt 驱动。

### 8.2 Bid Check

输入：定时 `bid_check` 或显式通知

当前缺口：`BiddingService` 存在，但没有形成清晰的运行主路径。

v3 目标：

- Committee 读取投标来源
- 调用 `BiddingService.collect()`
- 调用 `BiddingService.evaluate()`
- 若结果为 `assign`，调用 `Dispatcher.dispatch(... TASK_ASSIGNED ...)`

也就是说：

- 选标逻辑由 `BiddingService` 承担
- sandbox 创建和任务分发继续由 `Dispatcher` 承担

不要重复造轮子。

### 8.3 Completion

输入：worker 的 `task_completed`

v3 目标：

- Committee 进行验证
- 若通过，生成清晰的用户可见交付记录
- 如已有通知系统可复用，则先复用 `NotificationStore`
- Signal 仍作为 agent 间轻通知，不强行替代所有面向用户的投递

这里和 v2 不同：

- v3 不急着删除 `notifications.py`
- 先把“交付可见”做稳，再考虑是否统一到 Signal

---

## 9. MCP 改造范围

v3 只做最小必要修改。

### Phase M1：日志隔离

- 所有 MCP subprocess stderr 落盘到 `.aura/logs/`
- 用户终端不再看到 MCP 连接日志

### Phase M2：执行时连接

- 不在 chat / engine 主循环开始时统一 `ensure_mcp_connections()`
- 至少将“工具元数据发现”和“真实连接建立”拆开
- 能延后连接就延后

### 暂不做

- 不做连接池
- 不做跨 session 复用
- 不做复杂 breaker 重构

---

## 10. Dashboard 策略

v2 的 dashboard 设计过满，v3 做减法。

### 第一版只展示

- daemon 运行状态
- agents 概览
- issues 概览
- recent signals

### 第一版不强求

- 多级详情页面
- 键盘导航系统
- 复杂 audit tail drill-down
- 自定义 phase 推导状态机

原因：

- 当前已有 `control/dashboard.py` 能提供足够的数据基础
- 先做一个稳定的观察面，比一次做完整 TUI 更重要

---

## 11. 迁移策略

### Phase 1：产品入口收敛

目标：先解决用户路径混乱。

任务：

1. 新增 `aura submit`
2. 新增 `aura watch`
3. 保留 `aura status` 和 `dispatch` 作为调试命令
4. 将 `aura chat` 文案降级为轻量问答/小改动

验收：

- 用户可以不经过 LLM 提交任务
- 用户有统一的实时观察入口

### Phase 2：MCP 降噪

目标：解决当前最差体验点。

任务：

1. stderr 落盘
2. 改造连接时机

验收：

- chat 不再看到 MCP 噪声
- 非 MCP 操作启动更快

### Phase 3：Committee 闭环补线

目标：把自动推进从“主要靠 prompt”提升到“关键阶段有代码级支撑”。

任务：

1. `bid_check -> BiddingService -> Dispatcher`
2. `task_completed -> verify -> delivery`

验收：

- 竞标到分配自动推进
- 完成到交付自动推进

### Phase 4：结构整理

目标：在主路径稳定后，再进行包结构整理。

允许动作：

- 按模块搬迁文件
- 合并零散工具文件
- 收紧导入路径

前置条件：

- `submit -> daemon -> watch -> delivery` 已稳定
- 关键流程有测试覆盖

---

## 12. 明确不做什么

v3 明确不做以下事情：

- 不新增统一 `Agent` 抽象类
- 不新增新的 runtime 包装层
- 不第一阶段删除 `runtime/`
- 不第一阶段删除 `Dispatcher`
- 不第一阶段删除 `subagent__run`
- 不第一阶段重命名 `OpKind.CHAT`
- 不一次性删除大量工具和目录，只因为它们“看起来旧”

---

## 13. 判断标准

如果一个改动满足以下任意一条，就应该优先：

- 让用户少记一个入口
- 让自动化链路少一个手工步骤
- 让终端少一类噪声
- 让关键阶段从“靠提示词”变成“有代码支撑”

如果一个改动只是在做这些事，就应该后置：

- 文件名更整齐
- 包层次更漂亮
- 目录更“架构化”
- 旧代码看起来不顺眼

---

## 14. 最终建议

v3 的本质不是“再设计一个新架构”，而是把当前系统收束成一个明确产品：

- 用户用 `submit` 提交
- 系统用 `daemon` 自动推进
- 用户用 `watch` 观察
- `chat` 退居辅助

在这个前提下：

- 保留当前有效骨架
- 修掉最差体验点
- 补上关键闭环
- 最后再做目录整理

这才是对当前 AuraForge 成本最低、风险最低、收益最高的重构路径。
