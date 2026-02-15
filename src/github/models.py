"""Data models for GitHub PR review."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """Severity levels for violations."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ViolationCategory(str, Enum):
    """Categories of code violations."""
    STYLE = "style"
    QUALITY = "quality"
    SECURITY = "security"
    BEST_PRACTICES = "best_practices"


@dataclass
class FileChange:
    """Represents a changed file in a PR."""
    filename: str
    status: str  # added, modified, removed, renamed
    additions: int
    deletions: int
    patch: Optional[str] = None
    contents: Optional[str] = None
    
    @property
    def is_deleted(self) -> bool:
        return self.status == "removed"


@dataclass
class HunkLine:
    """Represents a single line in a diff hunk."""
    line_number: int
    content: str
    change_type: str  # 'added', 'removed', 'context'
    
    @property
    def is_added(self) -> bool:
        return self.change_type == "added"


@dataclass
class DiffHunk:
    """Represents a hunk in a diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[HunkLine] = field(default_factory=list)
    header: str = ""


@dataclass
class ParsedDiff:
    """Parsed diff for a file."""
    filename: str
    hunks: list[DiffHunk] = field(default_factory=list)
    
    def get_added_lines(self) -> list[HunkLine]:
        """Get all added lines across all hunks."""
        added = []
        for hunk in self.hunks:
            added.extend([line for line in hunk.lines if line.is_added])
        return added


@dataclass
class Violation:
    """Represents a code violation found during review."""
    rule_id: str
    rule_name: str
    category: ViolationCategory
    severity: Severity
    file: str
    line: int
    column: int = 0
    message: str = ""
    suggestion: str = ""
    code_snippet: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet,
            "confidence": self.confidence,
        }


@dataclass
class ReviewComment:
    """Represents a review comment to post on GitHub."""
    file: str
    line: int
    body: str
    severity: Severity
    rule_id: Optional[str] = None
    suggestion_code: Optional[str] = None
    
    def format_body(self) -> str:
        """Format the comment body with severity badge and suggestion."""
        severity_emoji = {
            Severity.CRITICAL: "🚨",
            Severity.ERROR: "❌",
            Severity.WARNING: "⚠️",
            Severity.INFO: "ℹ️",
        }
        
        emoji = severity_emoji.get(self.severity, "📝")
        formatted = f"{emoji} **{self.severity.value.upper()}**"
        
        if self.rule_id:
            formatted += f" [{self.rule_id}]"
        
        formatted += f"\n\n{self.body}"
        
        if self.suggestion_code:
            formatted += f"\n\n**Suggested fix:**\n```suggestion\n{self.suggestion_code}\n```"
        
        return formatted


@dataclass
class PullRequest:
    """Represents a GitHub Pull Request."""
    number: int
    title: str
    body: str
    base_branch: str
    head_branch: str
    author: str
    files: list[FileChange] = field(default_factory=list)
    commits_count: int = 0
    
    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)
    
    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)


@dataclass
class RefactorResult:
    """Result from the refactoring agent."""
    file: str
    original_code: str
    refactored_code: str
    changes_description: str
    violations_fixed: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ReviewSummary:
    """Summary of the code review."""
    total_files: int
    total_violations: int
    critical_count: int
    error_count: int
    warning_count: int
    info_count: int
    files_with_issues: list[str] = field(default_factory=list)
    refactoring_performed: bool = False
    refactoring_files: list[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate markdown summary for PR comment."""
        lines = [
            "## 🔍 AI Code Review Summary\n",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Files Reviewed | {self.total_files} |",
            f"| Total Issues | {self.total_violations} |",
            f"| 🚨 Critical | {self.critical_count} |",
            f"| ❌ Errors | {self.error_count} |",
            f"| ⚠️ Warnings | {self.warning_count} |",
            f"| ℹ️ Info | {self.info_count} |",
        ]
        
        if self.files_with_issues:
            lines.append("\n### Files with Issues\n")
            for f in self.files_with_issues:
                lines.append(f"- `{f}`")
        
        if self.refactoring_performed:
            lines.append("\n### 🔧 Automatic Refactoring Applied\n")
            for f in self.refactoring_files:
                lines.append(f"- `{f}`")
        
        if self.total_violations == 0:
            lines.append("\n✅ **No issues found! Great job!**")
        
        return "\n".join(lines)
