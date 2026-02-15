"""Agent communication protocol for structured handoffs between agents."""

import logging
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    REVIEWER = "reviewer"
    REFACTORER = "refactorer"
    VERIFIER = "verifier"
    TEST_RUNNER = "test_runner"


@dataclass
class HandoffContext:
    """
    Context passed during agent handoffs.

    Contains all information needed for the receiving agent to continue work.
    """
    pr_number: int
    repository: str
    branch: str

    files: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)

    original_code: dict[str, str] = field(default_factory=dict)
    refactored_code: dict[str, str] = field(default_factory=dict)

    commits_made: list[str] = field(default_factory=list)
    rollback_points: list[str] = field(default_factory=list)

    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    handoff_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pr_number": self.pr_number,
            "repository": self.repository,
            "branch": self.branch,
            "files": self.files,
            "violations": self.violations,
            "original_code": self.original_code,
            "refactored_code": self.refactored_code,
            "commits_made": self.commits_made,
            "rollback_points": self.rollback_points,
            "started_at": self.started_at,
            "handoff_chain": self.handoff_chain,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HandoffContext":
        """Create from dictionary."""
        return cls(
            pr_number=data["pr_number"],
            repository=data["repository"],
            branch=data["branch"],
            files=data.get("files", []),
            violations=data.get("violations", []),
            original_code=data.get("original_code", {}),
            refactored_code=data.get("refactored_code", {}),
            commits_made=data.get("commits_made", []),
            rollback_points=data.get("rollback_points", []),
            started_at=data.get("started_at", datetime.now(timezone.utc).isoformat()),
            handoff_chain=data.get("handoff_chain", []),
        )

    def add_to_chain(self, agent: AgentRole) -> None:
        """Record an agent in the handoff chain."""
        self.handoff_chain.append(f"{agent.value}@{datetime.now(timezone.utc).isoformat()}")

    def add_rollback_point(self, commit_sha: str) -> None:
        """Add a commit SHA as a rollback point."""
        self.rollback_points.append(commit_sha)

    def get_last_rollback_point(self) -> Optional[str]:
        """Get the most recent rollback point."""
        return self.rollback_points[-1] if self.rollback_points else None
