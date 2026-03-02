from __future__ import annotations

from .registry import ToolRegistry
from .runtime import (
    InspectionDecision,
    InspectionResult,
    PlannedToolCall,
    ToolApprovalMode,
    ToolExecutionContext,
    ToolRuntime,
    ToolRuntimeError,
)
from typing import Any
import importlib

__all__ = [
    "ToolRegistry",
    "ToolRuntime",
    "ToolRuntimeError",
    "InspectionDecision",
    "InspectionResult",
    "PlannedToolCall",
    "ToolApprovalMode",
    "ToolExecutionContext",
    # Lazily imported tool implementations (see __getattr__).
    "BrowserRunTool",
    "ProjectAIGCDetectTool",
    "ProjectApplyEditsTool",
    "ProjectApplyPatchTool",
    "ProjectGlobTool",
    "ProjectListDirTool",
    "ProjectPatchTool",
    "ProjectReadTextManyTool",
    "ProjectReadTextTool",
    "ProjectSearchTextTool",
    "ProjectTextStatsTool",
    "SessionExportTool",
    "SessionSearchTool",
    "ShellRunTool",
    "SkillListTool",
    "SkillLoadTool",
    "SkillReadFileTool",
    "SnapshotCreateTool",
    "SnapshotDiffTool",
    "SnapshotListTool",
    "SnapshotReadTextTool",
    "SnapshotRollbackTool",
    "SpecApplyTool",
    "SpecGetAssetTool",
    "SpecGetTool",
    "SpecListAssetsTool",
    "SpecProposeTool",
    "SpecQueryTool",
    "SpecSealTool",
    "WebFetchTool",
    "WebSearchTool",
    "WorkspaceAwardClaimTool",
    "WorkspaceAcceptSubmissionTool",
    "WorkspaceAuditChainTool",
    "WorkspaceAppendSubmissionEvidenceTool",
    "WorkspaceAdvanceIssueStateTool",
    "WorkspaceCloseWorkbenchTool",
    "WorkspaceCloseWorkspaceTool",
    "WorkspaceContextTool",
    "WorkspaceCreateOrGetTool",
    "WorkspaceGcWorkbenchTool",
    "WorkspaceHeartbeatWorkbenchTool",
    "WorkspaceListAwardsTool",
    "WorkspaceListClaimsTool",
    "WorkspaceListHeartbeatsTool",
    "WorkspaceListWorkbenchesTool",
    "WorkspaceListSubmissionsTool",
    "WorkspacePublishHeartbeatTool",
    "WorkspaceProvisionWorkbenchTool",
    "WorkspaceRecoverExpiredWorkbenchesTool",
    "WorkspaceRegisterSubmissionTool",
    "WorkspaceSubmitClaimTool",
    "WorkspaceTimelineTool",
    "WorkspaceTransitionWorkbenchStateTool",
    "WorkspaceWakeAwardedAgentTool",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ProjectReadTextTool": (".builtins", "ProjectReadTextTool"),
    "ProjectSearchTextTool": (".builtins", "ProjectSearchTextTool"),
    "ShellRunTool": (".builtins", "ShellRunTool"),
    "BrowserRunTool": (".browser", "BrowserRunTool"),
    "ProjectGlobTool": (".discovery", "ProjectGlobTool"),
    "ProjectListDirTool": (".discovery", "ProjectListDirTool"),
    "ProjectReadTextManyTool": (".discovery", "ProjectReadTextManyTool"),
    "SkillListTool": (".skills", "SkillListTool"),
    "SkillLoadTool": (".skills", "SkillLoadTool"),
    "SkillReadFileTool": (".skills", "SkillReadFileTool"),
    "SpecApplyTool": (".spec_workflow", "SpecApplyTool"),
    "SpecGetAssetTool": (".spec_assets", "SpecGetAssetTool"),
    "SpecGetTool": (".spec_workflow", "SpecGetTool"),
    "SpecListAssetsTool": (".spec_assets", "SpecListAssetsTool"),
    "SpecProposeTool": (".spec_workflow", "SpecProposeTool"),
    "SpecQueryTool": (".spec_workflow", "SpecQueryTool"),
    "SpecSealTool": (".spec_workflow", "SpecSealTool"),
    "SnapshotCreateTool": (".snapshot_tools", "SnapshotCreateTool"),
    "SnapshotDiffTool": (".snapshot_tools", "SnapshotDiffTool"),
    "SnapshotListTool": (".snapshot_tools", "SnapshotListTool"),
    "SnapshotReadTextTool": (".snapshot_tools", "SnapshotReadTextTool"),
    "SnapshotRollbackTool": (".snapshot_tools", "SnapshotRollbackTool"),
    "SessionExportTool": (".session_tools", "SessionExportTool"),
    "SessionSearchTool": (".session_tools", "SessionSearchTool"),
    "ProjectTextStatsTool": (".text_stats", "ProjectTextStatsTool"),
    "ProjectAIGCDetectTool": (".aigc_detect", "ProjectAIGCDetectTool"),
    "WebFetchTool": (".web", "WebFetchTool"),
    "WebSearchTool": (".web", "WebSearchTool"),
    "ProjectApplyPatchTool": (".apply_patch_tool", "ProjectApplyPatchTool"),
    "ProjectApplyEditsTool": (".apply_edits_tool", "ProjectApplyEditsTool"),
    "ProjectPatchTool": (".patch_tool", "ProjectPatchTool"),
    "WorkspaceAwardClaimTool": (".workspace_tools", "WorkspaceAwardClaimTool"),
    "WorkspaceAcceptSubmissionTool": (".workspace_tools", "WorkspaceAcceptSubmissionTool"),
    "WorkspaceAuditChainTool": (".workspace_tools", "WorkspaceAuditChainTool"),
    "WorkspaceAppendSubmissionEvidenceTool": (".workspace_tools", "WorkspaceAppendSubmissionEvidenceTool"),
    "WorkspaceAdvanceIssueStateTool": (".workspace_tools", "WorkspaceAdvanceIssueStateTool"),
    "WorkspaceCloseWorkbenchTool": (".workspace_tools", "WorkspaceCloseWorkbenchTool"),
    "WorkspaceCloseWorkspaceTool": (".workspace_tools", "WorkspaceCloseWorkspaceTool"),
    "WorkspaceCreateOrGetTool": (".workspace_tools", "WorkspaceCreateOrGetTool"),
    "WorkspaceProvisionWorkbenchTool": (".workspace_tools", "WorkspaceProvisionWorkbenchTool"),
    "WorkspaceContextTool": (".workspace_tools", "WorkspaceContextTool"),
    "WorkspaceGcWorkbenchTool": (".workspace_tools", "WorkspaceGcWorkbenchTool"),
    "WorkspaceRegisterSubmissionTool": (".workspace_tools", "WorkspaceRegisterSubmissionTool"),
    "WorkspaceHeartbeatWorkbenchTool": (".workspace_tools", "WorkspaceHeartbeatWorkbenchTool"),
    "WorkspaceListAwardsTool": (".workspace_tools", "WorkspaceListAwardsTool"),
    "WorkspaceListClaimsTool": (".workspace_tools", "WorkspaceListClaimsTool"),
    "WorkspaceListHeartbeatsTool": (".workspace_tools", "WorkspaceListHeartbeatsTool"),
    "WorkspaceListWorkbenchesTool": (".workspace_tools", "WorkspaceListWorkbenchesTool"),
    "WorkspaceListSubmissionsTool": (".workspace_tools", "WorkspaceListSubmissionsTool"),
    "WorkspacePublishHeartbeatTool": (".workspace_tools", "WorkspacePublishHeartbeatTool"),
    "WorkspaceRecoverExpiredWorkbenchesTool": (".workspace_tools", "WorkspaceRecoverExpiredWorkbenchesTool"),
    "WorkspaceSubmitClaimTool": (".workspace_tools", "WorkspaceSubmitClaimTool"),
    "WorkspaceTimelineTool": (".workspace_tools", "WorkspaceTimelineTool"),
    "WorkspaceTransitionWorkbenchStateTool": (".workspace_tools", "WorkspaceTransitionWorkbenchStateTool"),
    "WorkspaceWakeAwardedAgentTool": (".workspace_tools", "WorkspaceWakeAwardedAgentTool"),
}


def __getattr__(name: str) -> Any:
    """
    Keep `aura.runtime.tools` import fast by deferring tool implementation imports.

    This is important because `aura.runtime.tools.runtime` is imported very early
    (e.g. during CLI startup), and Python loads the package `__init__` first.
    """

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))
