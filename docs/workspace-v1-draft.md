# Aura Workspace v1 草案

## 1. 定义

Workspace 是 Aura 中用于交付型多代理协作的“审计化工作上下文容器”。

它绑定三件事：
- 一个 issue（默认 Linear）
- 一个代码仓库（默认 GitHub）
- 一组隔离 workbench（每个 agent 实例一个）

它的职责是隔离、追踪、审计，不负责调度策略本身。

## 2. 目标与边界

### 2.1 目标
- 支持上百 agent 并发执行且文件隔离。
- 每个产出都可回溯到 issue/branch/commit/PR/CI/tool_call。
- 统一外部动作出口（Linear/GitHub）并形成审计链。

### 2.2 非目标
- 不做 agent-to-agent 通信层（A2A-lite 另管）。
- 不做执行编排仲裁（activator/executor/committee 另管）。
- 不做本地 merge 策略（仅 PR-only）。
- 不替代 Git/VCS 本身。

## 3. 已确认策略

- `merge_policy = pr_only`
- `staging_enabled = true`
- `push_policy = integrator_only`

解释：
- Worker 只能本地改动和 commit。
- Integrator 负责 push、开 PR、合并。
- 大量并发时走 `agent/* -> staging/* -> main`。

## 4. 核心对象

### 4.1 IssueWorkspace（逻辑容器）

建议字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `workspace_id` | string | 例：`ws_PROJ-123` |
| `issue_ref` | object | `provider,id,key,url` |
| `repo_ref` | object | `provider,owner,repo` |
| `base_branch` | string | 默认 `main` |
| `staging_branch` | string | 例：`staging/PROJ-123` |
| `merge_policy` | string | 固定 `pr_only` |
| `push_policy` | string | 固定 `integrator_only` |
| `state` | string | 见状态机 |
| `created_at` | int | ms 时间戳 |
| `updated_at` | int | ms 时间戳 |
| `revision` | int | 乐观并发版本号 |

### 4.2 Workbench（物理工作台）

建议字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `workbench_id` | string | 例：`wb_PROJ-123_backend-developer_0007` |
| `workspace_id` | string | 归属 issue workspace |
| `agent_id` | string | 市场 agent id |
| `instance_id` | string | 运行实例 id |
| `role` | string | `worker`/`integrator`/`reviewer` |
| `worktree_path` | string | 本地隔离目录 |
| `branch` | string | 例：`agent/PROJ-123/backend-developer/i7f3ab` |
| `base_ref` | string | 默认 `staging/<issue_key>` |
| `state` | string | 见状态机 |
| `lease_until` | int\|null | 运行租约到期时间 |
| `last_heartbeat_at` | int\|null | 心跳时间 |
| `revision` | int | 乐观并发版本号 |

### 4.3 Submission（交付证据）

每次 worker 提交一个“可审计单元”，建议写入 JSONL：

| 字段 | 必填 | 说明 |
|---|---|---|
| `submission_id` | 是 | 唯一 id |
| `workspace_id` | 是 | 所属 workspace |
| `workbench_id` | 是 | 来源 workbench |
| `instance_id` | 是 | 来源实例 |
| `agent_id` | 是 | 来源 agent |
| `branch` | 是 | 产出分支 |
| `commit_sha` | 是 | 产出 commit |
| `changed_files` | 是 | 变更文件列表 |
| `tool_call_ids` | 是 | 外部动作 tool call 证据链 |
| `status` | 是 | `submitted/accepted/rejected/integrated` |
| `pr_url` | 否 | PR 链接 |
| `ci_url` | 否 | CI 链接 |
| `notes` | 否 | 人工备注 |
| `created_at` | 是 | 创建时间 |

## 5. 存储布局（对齐现有 `.aura`）

- `.aura/state/workspaces/<workspace_id>.json`
- `.aura/state/workbenches/<workbench_id>.json`
- `.aura/index/workspace_submissions.jsonl`
- `.aura/index/workspace_timeline.jsonl`（可选）
- `.aura/artifacts/workspaces/<workspace_id>/...`（可选）

worktree 建议目录：
- `.aura/tmp/worktrees/<workspace_id>/<workbench_id>/`

## 6. 命名规范

- `workspace_id = ws_<issue_key>`
- `workbench_id = wb_<issue_key>_<agent_slug>_<seq4>`
- `staging_branch = staging/<issue_key>`
- `agent_branch = agent/<issue_key>/<agent_slug>/<instance_short>`

## 7. 状态机

### 7.1 IssueWorkspace

状态：
- `draft`
- `active`
- `integrating`
- `done`
- `blocked`
- `archived`

关键迁移：
- `draft -> active`：staging 分支准备完成，至少一个 workbench ready。
- `active -> integrating`：有 submission 被 integrator 接收并进入 PR 流。
- `integrating -> active`：回到继续开发阶段。
- `active|integrating -> blocked`：关键依赖阻塞（CI 持续红、权限失败等）。
- `blocked -> active`：阻塞解除。
- `active|integrating -> done`：最终 PR 合并到 `base_branch`。
- `done|blocked -> archived`：归档（TTL 或人工）。

### 7.2 Workbench

状态：
- `provisioning`
- `ready`
- `running`
- `submitted`
- `integrated`
- `blocked`
- `abandoned`
- `closed`
- `gc`

关键迁移：
- `provisioning -> ready`：worktree + branch 创建成功。
- `ready -> running`：agent 实例被执行器绑定启动。
- `running -> submitted`：完成并登记 submission。
- `submitted -> integrated`：integrator 合并其对应 PR。
- `running|submitted -> blocked`：测试/权限/冲突等硬失败。
- `blocked -> ready`：人工或自动修复后重试。
- `integrated|abandoned -> closed`：生命周期结束。
- `closed -> gc`：清理 worktree 与临时资产。

## 8. 并发、幂等、锁

### 8.1 幂等键
- Workspace 创建幂等键：`(issue_ref, repo_ref, base_branch)`。
- Workbench 创建幂等键：`(workspace_id, agent_id, instance_id)`。
- Submission 幂等键：`(workspace_id, workbench_id, commit_sha)`。

### 8.2 更新语义
- 状态对象带 `revision`，写入时做 compare-and-swap。
- 同一对象写入失败时应自动重读并重试（有限次数）。

### 8.3 运行租约
- `running` workbench 必须周期心跳。
- 超过 `lease_until` 未续约可回收为 `ready` 或转 `blocked`。
- 恢复时必须保留失败原因和原 `instance_id` 关联。

## 9. 与现有内核集成

### 9.1 执行前注入（必须）
- 为每个 agent 实例先分配 workbench。
- 强制 `cwd = worktree_path`。
- 注入最小 `workspace_context` 到请求上下文。

### 9.2 WorkSpec 强约束（复用现有能力）
- `WorkSpec.resource_scope.workspace_roots = [worktree_path_relative]`
- 仅允许在该 root 内 `project__*` 文件工具写入。
- domain/file_type allowlist 继续走现有 ToolRuntime 检查。

### 9.3 事件与审计
- v1 不强制新增 EventKind。
- 先复用 `operation_*` / `tool_call_*`，在 payload 增加：
  - `workspace_id`
  - `workbench_id`
  - `submission_id`（如有）
  - `external_ref`（`pr_url`/`ci_url`/`linear_comment_id`）

## 10. 工具面（v1 最小集合）

对所有 agent 暴露：
- `workspace__context`：读取当前 workspace/workbench 关键上下文。
- `workspace__register_submission`：登记 commit/pr/ci/tool_call 证据。

仅 integrator 暴露：
- `workspace__accept_submission`
- `workspace__open_pr`
- `workspace__merge_pr`
- `workspace__advance_issue_state`

执行器内部调用（不直接给普通 agent）：
- `workspace__create_or_get`
- `workspace__provision_workbench`
- `workspace__heartbeat_workbench`
- `workspace__close_workbench`
- `workspace__close_workspace`

## 11. 权限模型

Worker：
- 允许：本地编辑、测试、本地 commit、登记 submission。
- 禁止：push、开 PR、merge、改 Linear 状态。

Integrator / Committee：
- 允许：push、开 PR、merge、issue 状态推进、submission 采纳/驳回。
- 责任：保证最终交付链条闭环（issue -> branch -> PR -> CI -> merge）。

## 12. 失败恢复与 GC

- Git push/PR 创建失败：workbench -> `blocked`，附错误码与重试次数。
- CI 失败：submission 标记 `rejected` 或 `needs_fix`，回流对应 workbench。
- 进程重启：从 `.aura/state/*` 恢复状态机，按 `lease_until` 扫描失联运行。
- GC 默认策略：
  - `closed` 且超过 7 天的 workbench 清理 worktree。
  - `done` 且超过 30 天的 workspace 可归档压缩。

## 13. 验收标准（进入实现前）

- 50+ 并发 workbench 下无路径踩踏。
- worker 无法执行 push/open_pr/merge（权限硬拦截）。
- 每个最终合并变更可追溯到 `submission_id` 和 `tool_call_id`。
- 进程重启后可恢复 `active/running/submitted` 的正确状态。
- 可通过一个查询接口列出某 issue 的全链路审计信息。

## 14. 分阶段实现建议

Phase 1（最小闭环）：
- IssueWorkspace/Workbench 持久化模型 + 状态迁移
- executor 注入 workbench + WorkSpec workspace_roots
- `workspace__context` / `workspace__register_submission`

Phase 2（集成闭环）：
- integrator-only push/open_pr/merge 工具
- submission 采纳流 + staging 集成流
- Linear 状态推进工具接入

Phase 3（规模化与治理）：
- 租约恢复、冲突自动重试、GC
- 指标与看板（阻塞率、吞吐、集成时延）
