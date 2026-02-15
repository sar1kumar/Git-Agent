"""Agent communication protocol for structured message passing."""

import uuid
import logging
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages exchanged between agents."""
    # Requests
    REVIEW_REQUEST = "review_request"
    REFACTOR_REQUEST = "refactor_request"
    VERIFY_REQUEST = "verify_request"
    TEST_REQUEST = "test_request"
    ROLLBACK_REQUEST = "rollback_request"
    
    # Responses
    REVIEW_RESULT = "review_result"
    REFACTOR_RESULT = "refactor_result"
    VERIFY_RESULT = "verify_result"
    TEST_RESULT = "test_result"
    ROLLBACK_RESULT = "rollback_result"
    
    # Control
    HANDOFF = "handoff"
    ACKNOWLEDGE = "acknowledge"
    ERROR = "error"
    STATUS_UPDATE = "status_update"


class AgentRole(str, Enum):
    """Roles of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    REVIEWER = "reviewer"
    REFACTORER = "refactorer"
    VERIFIER = "verifier"
    TEST_RUNNER = "test_runner"


class Priority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    """
    Structured message for agent-to-agent communication.
    
    This provides a consistent protocol for all inter-agent messaging,
    enabling traceability, error handling, and state management.
    """
    message_type: MessageType
    sender: AgentRole
    recipient: AgentRole
    payload: dict = field(default_factory=dict)
    
    # Metadata
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None  # Links related messages
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    priority: Priority = Priority.NORMAL
    
    # State
    requires_response: bool = True
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender": self.sender.value,
            "recipient": self.recipient.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "requires_response": self.requires_response,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            message_type=MessageType(data["message_type"]),
            sender=AgentRole(data["sender"]),
            recipient=AgentRole(data["recipient"]),
            payload=data.get("payload", {}),
            message_id=data.get("message_id", str(uuid.uuid4())),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            priority=Priority(data.get("priority", "normal")),
            requires_response=data.get("requires_response", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )
    
    def create_response(
        self,
        message_type: MessageType,
        sender: AgentRole,
        payload: dict,
    ) -> "AgentMessage":
        """Create a response message linked to this message."""
        return AgentMessage(
            message_type=message_type,
            sender=sender,
            recipient=self.sender,
            payload=payload,
            correlation_id=self.message_id,
            priority=self.priority,
            requires_response=False,
        )
    
    def create_error_response(
        self,
        sender: AgentRole,
        error: str,
        details: Optional[dict] = None,
    ) -> "AgentMessage":
        """Create an error response."""
        return AgentMessage(
            message_type=MessageType.ERROR,
            sender=sender,
            recipient=self.sender,
            payload={
                "error": error,
                "original_message_type": self.message_type.value,
                "details": details or {},
            },
            correlation_id=self.message_id,
            priority=Priority.HIGH,
            requires_response=False,
        )


@dataclass
class HandoffContext:
    """
    Context passed during agent handoffs.
    
    Contains all information needed for the receiving agent to continue work.
    """
    pr_number: int
    repository: str
    branch: str
    
    # Files being worked on
    files: list[str] = field(default_factory=list)
    
    # Analysis results
    violations: list[dict] = field(default_factory=list)
    
    # Refactoring context
    original_code: dict[str, str] = field(default_factory=dict)  # filename -> code
    refactored_code: dict[str, str] = field(default_factory=dict)
    
    # State
    commits_made: list[str] = field(default_factory=list)
    rollback_points: list[str] = field(default_factory=list)
    
    # Metadata
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    handoff_chain: list[str] = field(default_factory=list)  # Track agent sequence
    
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


class MessageBus:
    """
    Simple in-memory message bus for agent communication.
    
    In production, this could be replaced with Redis, RabbitMQ, etc.
    """
    
    def __init__(self):
        self._queues: dict[AgentRole, list[AgentMessage]] = {
            role: [] for role in AgentRole
        }
        self._handlers: dict[AgentRole, dict[MessageType, Any]] = {
            role: {} for role in AgentRole
        }
        self._message_log: list[AgentMessage] = []
    
    def send(self, message: AgentMessage) -> None:
        """Send a message to an agent."""
        logger.debug(f"Message {message.message_id}: {message.sender.value} -> {message.recipient.value} [{message.message_type.value}]")
        self._message_log.append(message)
        self._queues[message.recipient].append(message)
    
    def receive(self, agent: AgentRole, block: bool = False) -> Optional[AgentMessage]:
        """Receive the next message for an agent."""
        queue = self._queues[agent]
        if queue:
            # Priority queue - sort by priority then timestamp
            queue.sort(key=lambda m: (
                {"critical": 0, "high": 1, "normal": 2, "low": 3}[m.priority.value],
                m.timestamp
            ))
            return queue.pop(0)
        return None
    
    def register_handler(
        self,
        agent: AgentRole,
        message_type: MessageType,
        handler: Any,
    ) -> None:
        """Register a handler for a message type."""
        self._handlers[agent][message_type] = handler
    
    def process_messages(self, agent: AgentRole) -> list[AgentMessage]:
        """Process all pending messages for an agent."""
        responses = []
        
        while True:
            message = self.receive(agent)
            if not message:
                break
            
            handler = self._handlers[agent].get(message.message_type)
            if handler:
                try:
                    response = handler(message)
                    if response:
                        responses.append(response)
                        self.send(response)
                except Exception as e:
                    error_response = message.create_error_response(
                        agent,
                        str(e),
                    )
                    responses.append(error_response)
                    self.send(error_response)
            else:
                logger.warning(f"No handler for {message.message_type.value} in {agent.value}")
        
        return responses
    
    def get_message_log(self) -> list[dict]:
        """Get the full message log."""
        return [m.to_dict() for m in self._message_log]
    
    def clear(self) -> None:
        """Clear all queues and logs."""
        for queue in self._queues.values():
            queue.clear()
        self._message_log.clear()


# Factory functions for common messages

def create_review_request(
    pr_number: int,
    repository: str,
    files: list[str],
) -> AgentMessage:
    """Create a review request message."""
    return AgentMessage(
        message_type=MessageType.REVIEW_REQUEST,
        sender=AgentRole.ORCHESTRATOR,
        recipient=AgentRole.REVIEWER,
        payload={
            "pr_number": pr_number,
            "repository": repository,
            "files": files,
        },
        priority=Priority.NORMAL,
    )


def create_refactor_request(
    context: HandoffContext,
    violations: list[dict],
) -> AgentMessage:
    """Create a refactor request with handoff context."""
    return AgentMessage(
        message_type=MessageType.REFACTOR_REQUEST,
        sender=AgentRole.REVIEWER,
        recipient=AgentRole.REFACTORER,
        payload={
            "context": context.to_dict(),
            "violations": violations,
        },
        priority=Priority.HIGH,
    )


def create_verify_request(
    context: HandoffContext,
    refactored_files: dict[str, str],
) -> AgentMessage:
    """Create a verification request."""
    return AgentMessage(
        message_type=MessageType.VERIFY_REQUEST,
        sender=AgentRole.REFACTORER,
        recipient=AgentRole.VERIFIER,
        payload={
            "context": context.to_dict(),
            "refactored_files": refactored_files,
        },
        priority=Priority.HIGH,
    )


def create_test_request(
    context: HandoffContext,
    test_files: Optional[list[str]] = None,
) -> AgentMessage:
    """Create a test execution request."""
    return AgentMessage(
        message_type=MessageType.TEST_REQUEST,
        sender=AgentRole.VERIFIER,
        recipient=AgentRole.TEST_RUNNER,
        payload={
            "context": context.to_dict(),
            "test_files": test_files,
        },
        priority=Priority.HIGH,
    )


def create_rollback_request(
    context: HandoffContext,
    reason: str,
) -> AgentMessage:
    """Create a rollback request."""
    return AgentMessage(
        message_type=MessageType.ROLLBACK_REQUEST,
        sender=AgentRole.VERIFIER,
        recipient=AgentRole.ORCHESTRATOR,
        payload={
            "context": context.to_dict(),
            "reason": reason,
            "rollback_to": context.get_last_rollback_point(),
        },
        priority=Priority.CRITICAL,
    )
