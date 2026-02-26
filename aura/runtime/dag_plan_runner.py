from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

from .dag.graph import DAG
from .dag.scheduler import Scheduler
from .plan import PlanItem, PlanState, PlanStore, StepStatus


@dataclass
class DAGPlanRunner:
    """
    DAG计划执行器：管理PlanStore → Scheduler的转换和状态同步。

    职责：
    - 从PlanStore构建DAG图并初始化Scheduler
    - 跟踪已完成/运行节点，自动更新PlanStore状态
    - 提供就绪节点查询接口
    """

    plan_store: PlanStore
    max_parallel: int = 3

    _scheduler: Scheduler[str] | None = field(default=None, init=False, repr=False)
    _last_plan_hash: str | None = field(default=None, init=False, repr=False)

    def refresh_from_store(self) -> None:
        """
        从PlanStore重建DAG和Scheduler。

        调用时机：
        - 首次调用get_dispatchable_nodes()
        - PlanStore被外部修改后（如主Agent接受了Proposal并调用update_plan）
        """

        plan_state = self.plan_store.get()
        current_hash = self._compute_plan_hash(plan_state)
        if self._scheduler is None or current_hash != self._last_plan_hash:
            dag = self._build_dag_from_items(plan_state.plan)
            self._scheduler = Scheduler(dag=dag, max_parallel=self.max_parallel)
            self._last_plan_hash = current_hash

        self._sync_scheduler_state(plan_state)

    def get_dispatchable_nodes(self) -> list[PlanItem]:
        """
        返回当前可派发的节点列表。

        Returns:
            list[PlanItem]: 就绪且未在运行的节点
        """

        self.refresh_from_store()
        if self._scheduler is None:
            return []

        next_node_ids = self._scheduler.get_next_nodes()
        if not next_node_ids:
            return []

        plan_state = self.plan_store.get()
        items_by_id = {item.id: item for item in plan_state.plan}
        dispatchable: list[PlanItem] = []
        for node_id in next_node_ids:
            item = items_by_id.get(node_id)
            if item is None:
                continue
            if item.status is StepStatus.PENDING:
                dispatchable.append(item)

        return dispatchable

    def mark_completed(self, node_id: str, *, node_result: dict[str, Any] | None = None) -> None:
        """
        标记节点完成，同时更新Scheduler和PlanStore。
        """

        if self._scheduler is None:
            self.refresh_from_store()
        if self._scheduler is None:
            raise RuntimeError("Scheduler not initialized.")

        self._scheduler.mark_completed(node_id)

        plan_state = self.plan_store.get()
        updated_items: list[PlanItem] = []
        found = False
        for item in plan_state.plan:
            if item.id == node_id:
                found = True
                new_meta = dict(item.metadata)
                if node_result is not None:
                    new_meta["node_result"] = node_result
                if item.status is StepStatus.COMPLETED and new_meta == item.metadata:
                    updated_items.append(item)
                else:
                    updated_items.append(replace(item, status=StepStatus.COMPLETED, metadata=new_meta))
            else:
                updated_items.append(item)
        if not found:
            raise KeyError(f"PlanItem not found: {node_id}")

        self.plan_store.set(updated_items, goal=plan_state.goal, explanation=plan_state.explanation)
        self._last_plan_hash = self._compute_plan_hash(
            PlanState(plan=updated_items, goal=plan_state.goal, explanation=plan_state.explanation, updated_at=plan_state.updated_at)
        )

    def mark_failed(self, node_id: str, error: str, *, node_result: dict[str, Any] | None = None) -> None:
        """标记节点失败，保留错误信息。"""
        if self._scheduler is None:
            self.refresh_from_store()
        if self._scheduler is None:
            raise RuntimeError("Scheduler not initialized.")

        self._scheduler.mark_completed(node_id)  # 从调度器移除

        plan_state = self.plan_store.get()
        updated_items: list[PlanItem] = []
        found = False
        for item in plan_state.plan:
            if item.id == node_id:
                found = True
                new_trace = list(item.error_trace) + [{"error": error}]
                new_meta = dict(item.metadata)
                if node_result is not None:
                    new_meta["node_result"] = node_result
                updated_items.append(replace(item, status=StepStatus.FAILED, error_trace=new_trace, metadata=new_meta))
            else:
                updated_items.append(item)
        if not found:
            raise KeyError(f"PlanItem not found: {node_id}")

        self.plan_store.set(updated_items, goal=plan_state.goal, explanation=plan_state.explanation)
        self._last_plan_hash = self._compute_plan_hash(
            PlanState(plan=updated_items, goal=plan_state.goal, explanation=plan_state.explanation, updated_at=plan_state.updated_at)
        )

    def get_goal(self) -> str | None:
        """获取当前计划的全局目标。"""
        return self.plan_store.get().goal

    def get_progress_summary(self) -> str:
        """生成当前进度摘要，用于注入到subagent context。"""
        items = self.plan_store.get().plan
        completed = [it for it in items if it.status is StepStatus.COMPLETED]
        failed = [it for it in items if it.status is StepStatus.FAILED]
        lines = [f"✅ 已完成 {len(completed)}/{len(items)}"]
        if completed:
            lines.append("最近完成: " + ", ".join(it.id for it in completed[-3:]))
        if failed:
            lines.append(f"❌ 失败 {len(failed)}: " + ", ".join(it.id for it in failed))
        return "\n".join(lines)

    def is_all_done(self) -> bool:
        """
        检查DAG是否全部完成。
        """

        plan_state = self.plan_store.get()
        return all(item.status in (StepStatus.COMPLETED, StepStatus.FAILED) for item in plan_state.plan)

    def _build_dag_from_items(self, items: list[PlanItem]) -> DAG[str]:
        dag: DAG[str] = DAG()
        for item in items:
            dag.add_node(item.id)
        for item in items:
            for dep_id in item.depends_on:
                dag.add_edge(dep_id, item.id)
        return dag

    def _compute_plan_hash(self, plan_state: PlanState) -> str:
        node_ids = sorted(item.id for item in plan_state.plan)
        edges: list[str] = []
        for item in plan_state.plan:
            for dep in item.depends_on:
                edges.append(f"{dep}->{item.id}")
        edges.sort()
        signature = "|".join(node_ids) + "||" + "|".join(edges)
        return sha256(signature.encode("utf-8")).hexdigest()

    def _sync_scheduler_state(self, plan_state: PlanState) -> None:
        if self._scheduler is None:
            return

        dag = self._scheduler.dag
        completed = {item.id for item in plan_state.plan if item.status is StepStatus.COMPLETED}
        running = {item.id for item in plan_state.plan if item.status is StepStatus.IN_PROGRESS}
        running |= set(self._scheduler._running)
        if completed & running:
            running = running - completed

        in_degree = {node: 0 for node in dag.nodes()}
        for src in dag.nodes():
            for dst in dag.successors(src):
                if src in completed:
                    continue
                in_degree[dst] = in_degree.get(dst, 0) + 1

        ready = [
            node
            for node in dag.nodes()
            if in_degree.get(node, 0) == 0 and node not in completed and node not in running
        ]

        # Sync scheduler internal state with PlanStore statuses.
        self._scheduler._completed = set(completed)
        self._scheduler._running = set(running)
        self._scheduler._in_degree = in_degree
        self._scheduler._ready = deque(ready)
