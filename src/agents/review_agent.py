"""Code review agent for analyzing PRs."""

import logging
from typing import Optional

from src.github.models import (
    PullRequest,
    FileChange,
    Violation,
    ReviewComment,
    Severity,
    ReviewSummary,
    ParsedDiff,
)
from src.github.client import GitHubClient
from src.analysis.diff_parser import DiffParser
from src.analysis.rules_engine import RulesEngine
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ReviewAgent:
    """Agent for performing code reviews on pull requests."""
    
    # File extensions to analyze
    SUPPORTED_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
        ".rb", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs",
    }
    
    # Files to skip
    SKIP_PATTERNS = {
        "package-lock.json", "yarn.lock", "Pipfile.lock",
        ".min.js", ".min.css", ".map",
    }
    
    def __init__(
        self,
        github_client: GitHubClient,
        rules_engine: Optional[RulesEngine] = None,
        ollama_client: Optional[OllamaClient] = None,
        use_llm: bool = True,
    ):
        """
        Initialize the review agent.
        
        Args:
            github_client: GitHub API client.
            rules_engine: Rules engine for static analysis.
            ollama_client: Ollama client for LLM analysis.
            use_llm: Whether to use LLM for enhanced analysis.
        """
        self.github = github_client
        self.rules = rules_engine or RulesEngine()
        self.diff_parser = DiffParser()
        self.use_llm = use_llm
        
        if use_llm:
            self.llm = ollama_client or OllamaClient()
        else:
            self.llm = None
        
        logger.info(f"Initialized ReviewAgent (use_llm={use_llm})")
    
    def review_pr(self, pr_number: int) -> tuple[list[Violation], list[ReviewComment], ReviewSummary]:
        """
        Perform a complete code review on a PR.
        
        Args:
            pr_number: Pull request number.
            
        Returns:
            Tuple of (violations, comments, summary).
        """
        logger.info(f"Starting review of PR #{pr_number}")
        
        # Fetch PR details
        pr = self.github.get_pull_request(pr_number)
        
        all_violations: list[Violation] = []
        all_comments: list[ReviewComment] = []
        files_with_issues: list[str] = []
        
        # Analyze each file
        for file_change in pr.files:
            if not self._should_analyze(file_change):
                logger.debug(f"Skipping {file_change.filename}")
                continue
            
            logger.info(f"Analyzing {file_change.filename}")
            
            # Parse diff
            parsed_diff = self.diff_parser.parse_file_diff(file_change)
            
            # Get full file contents for context
            content = self.github.get_file_contents(
                file_change.filename,
                ref=pr.head_branch,
            )
            
            if content is None and file_change.is_deleted:
                continue
            
            # Static analysis with rules engine
            if content:
                violations = self.rules.analyze(
                    file_change.filename,
                    content,
                    parsed_diff,
                )
                all_violations.extend(violations)
                
                if violations:
                    files_with_issues.append(file_change.filename)
            
            # LLM-based analysis for added code
            if self.use_llm and self.llm and parsed_diff:
                llm_violations = self._llm_analyze(file_change, parsed_diff, content)
                all_violations.extend(llm_violations)
        
        # Generate review comments from violations
        for violation in all_violations:
            comment = self._violation_to_comment(violation)
            all_comments.append(comment)
        
        # Generate summary
        summary = self._generate_summary(all_violations, files_with_issues, len(pr.files))
        
        logger.info(f"Review complete: {len(all_violations)} violations, {len(all_comments)} comments")
        
        return all_violations, all_comments, summary
    
    def _should_analyze(self, file_change: FileChange) -> bool:
        """Check if a file should be analyzed."""
        filename = file_change.filename
        
        # Skip deleted files
        if file_change.is_deleted:
            return False
        
        # Skip files without patches
        if not file_change.patch:
            return False
        
        # Skip known non-code files
        for pattern in self.SKIP_PATTERNS:
            if pattern in filename:
                return False
        
        # Check extension
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        if ext and ext not in self.SUPPORTED_EXTENSIONS:
            return False
        
        return True
    
    def _llm_analyze(
        self,
        file_change: FileChange,
        parsed_diff: ParsedDiff,
        full_content: Optional[str],
    ) -> list[Violation]:
        """Perform LLM-based analysis on changed code."""
        violations = []
        
        if not self.llm:
            return violations
        
        # Get added code chunks
        chunks = self.diff_parser.get_added_code_chunks(parsed_diff, min_chunk_size=3)
        
        for chunk in chunks:
            code = chunk["code"]
            start_line = chunk["start_line"]
            
            # Skip very small chunks
            if len(code.split("\n")) < 3:
                continue
            
            try:
                analysis = self.llm.analyze_code(
                    code,
                    context=f"File: {file_change.filename}",
                    focus_areas=["security", "bugs", "best practices"],
                )
                
                for issue in analysis.get("issues", []):
                    severity_str = issue.get("severity", "warning").lower()
                    try:
                        severity = Severity(severity_str)
                    except ValueError:
                        severity = Severity.WARNING
                    
                    # Calculate actual line number
                    issue_line = issue.get("line")
                    if issue_line:
                        actual_line = start_line + issue_line - 1
                    else:
                        actual_line = start_line
                    
                    violations.append(Violation(
                        rule_id="LLM001",
                        rule_name="llm_analysis",
                        category=self._infer_category(issue.get("description", "")),
                        severity=severity,
                        file=file_change.filename,
                        line=actual_line,
                        message=issue.get("description", "Issue detected by AI analysis"),
                        suggestion=issue.get("suggestion", ""),
                        code_snippet=code[:200] if len(code) > 200 else code,
                        confidence=0.8,  # LLM findings have slightly lower confidence
                    ))
                    
            except Exception as e:
                logger.warning(f"LLM analysis failed for chunk: {e}")
        
        return violations
    
    def _infer_category(self, description: str) -> "ViolationCategory":
        """Infer violation category from description."""
        from src.github.models import ViolationCategory
        
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["security", "vulnerability", "injection", "xss", "secret"]):
            return ViolationCategory.SECURITY
        elif any(word in description_lower for word in ["style", "naming", "format", "indent"]):
            return ViolationCategory.STYLE
        elif any(word in description_lower for word in ["complexity", "performance", "memory", "efficiency"]):
            return ViolationCategory.QUALITY
        else:
            return ViolationCategory.BEST_PRACTICES
    
    def _violation_to_comment(self, violation: Violation) -> ReviewComment:
        """Convert a violation to a review comment."""
        # Optionally enhance with LLM
        body = violation.message
        suggestion_code = None
        
        if self.use_llm and self.llm and violation.severity in (Severity.CRITICAL, Severity.ERROR):
            try:
                enhanced = self.llm.generate_review_comment(
                    violation.code_snippet,
                    violation.rule_name,
                    violation.message,
                )
                body = enhanced
            except Exception as e:
                logger.debug(f"Could not enhance comment: {e}")
        
        if violation.suggestion:
            body += f"\n\n**Suggestion:** {violation.suggestion}"
        
        return ReviewComment(
            file=violation.file,
            line=violation.line,
            body=body,
            severity=violation.severity,
            rule_id=violation.rule_id,
            suggestion_code=suggestion_code,
        )
    
    def _generate_summary(
        self,
        violations: list[Violation],
        files_with_issues: list[str],
        total_files: int,
    ) -> ReviewSummary:
        """Generate a review summary."""
        summary = ReviewSummary(
            total_files=total_files,
            total_violations=len(violations),
            critical_count=sum(1 for v in violations if v.severity == Severity.CRITICAL),
            error_count=sum(1 for v in violations if v.severity == Severity.ERROR),
            warning_count=sum(1 for v in violations if v.severity == Severity.WARNING),
            info_count=sum(1 for v in violations if v.severity == Severity.INFO),
            files_with_issues=list(set(files_with_issues)),
        )
        
        return summary
    
    def should_delegate_to_refactor(self, violations: list[Violation]) -> bool:
        """Check if violations warrant delegation to refactoring agent."""
        return self.rules.should_delegate(violations)
    
    def post_review(
        self,
        pr_number: int,
        comments: list[ReviewComment],
        summary: ReviewSummary,
    ) -> bool:
        """
        Post review comments and summary to the PR.
        
        Args:
            pr_number: Pull request number.
            comments: List of review comments.
            summary: Review summary.
            
        Returns:
            True if successful.
        """
        summary_md = summary.to_markdown()
        
        # Try to post as a proper review
        success = self.github.post_review(pr_number, comments, summary_md)
        
        if not success:
            # Fall back to just posting summary
            self.github.post_summary_comment(pr_number, summary_md)
        
        return success
