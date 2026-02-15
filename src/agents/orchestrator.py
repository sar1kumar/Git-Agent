"""LangGraph orchestrator for multi-agent code review workflow."""

import logging
import json
from datetime import datetime, timezone
from typing import TypedDict, Annotated, Literal
from pathlib import Path

from langgraph.graph import StateGraph, END

from src.github.models import (
    PullRequest,
    FileChange,
    Violation,
    ReviewComment,
    ReviewSummary,
    RefactorResult,
)
from src.github.client import GitHubClient
from src.analysis.rules_engine import RulesEngine
from src.llm.ollama_client import OllamaClient
from src.agents.review_agent import ReviewAgent

logger = logging.getLogger(__name__)


class ReviewState(TypedDict):
    """State for the review workflow."""
    # Input
    pr_number: int
    
    # PR data
    pr: PullRequest | None
    files: list[FileChange]
    
    # Analysis results
    violations: list[Violation]
    comments: list[ReviewComment]
    
    # Delegation
    needs_refactor: bool
    refactor_results: list[RefactorResult]
    
    # Output
    summary: ReviewSummary | None
    
    # Logging
    logs: list[dict]
    
    # Workflow control
    error: str | None


def add_log(state: ReviewState, event: str, **kwargs) -> dict:
    """Create a log entry."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }


class ReviewOrchestrator:
    """LangGraph-based orchestrator for the code review workflow."""
    
    def __init__(
        self,
        github_client: GitHubClient,
        rules_engine: RulesEngine | None = None,
        ollama_client: OllamaClient | None = None,
        use_llm: bool = True,
        log_path: Path | str | None = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            github_client: GitHub API client.
            rules_engine: Rules engine for static analysis.
            ollama_client: Ollama client for LLM.
            use_llm: Whether to use LLM for analysis.
            log_path: Path to save JSONL logs.
        """
        self.github = github_client
        self.rules = rules_engine or RulesEngine()
        self.ollama = ollama_client
        self.use_llm = use_llm
        self.log_path = Path(log_path) if log_path else None
        
        # Initialize agents
        self.review_agent = ReviewAgent(
            github_client=self.github,
            rules_engine=self.rules,
            ollama_client=self.ollama,
            use_llm=self.use_llm,
        )
        
        # Build the workflow graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        logger.info("Initialized ReviewOrchestrator")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create graph with state schema
        graph = StateGraph(ReviewState)
        
        # Add nodes
        graph.add_node("fetch_pr", self._fetch_pr)
        graph.add_node("analyze_code", self._analyze_code)
        graph.add_node("check_delegation", self._check_delegation)
        graph.add_node("refactor", self._refactor)
        graph.add_node("post_comments", self._post_comments)
        graph.add_node("generate_summary", self._generate_summary)
        
        # Set entry point
        graph.set_entry_point("fetch_pr")
        
        # Add edges
        graph.add_edge("fetch_pr", "analyze_code")
        graph.add_edge("analyze_code", "check_delegation")
        
        # Conditional edge for delegation
        graph.add_conditional_edges(
            "check_delegation",
            self._should_refactor,
            {
                "refactor": "refactor",
                "skip_refactor": "post_comments",
            }
        )
        
        graph.add_edge("refactor", "post_comments")
        graph.add_edge("post_comments", "generate_summary")
        graph.add_edge("generate_summary", END)
        
        return graph
    
    def _fetch_pr(self, state: ReviewState) -> dict:
        """Fetch PR details from GitHub."""
        pr_number = state["pr_number"]
        logs = state.get("logs", [])
        
        logs.append(add_log(state, "review_started", pr=pr_number))
        
        try:
            pr = self.github.get_pull_request(pr_number)
            logger.info(f"Fetched PR #{pr_number}: {len(pr.files)} files")
            
            logs.append(add_log(
                state,
                "pr_fetched",
                files_count=len(pr.files),
                additions=pr.total_additions,
                deletions=pr.total_deletions,
            ))
            
            return {
                "pr": pr,
                "files": pr.files,
                "logs": logs,
            }
        except Exception as e:
            logger.error(f"Failed to fetch PR: {e}")
            logs.append(add_log(state, "error", message=str(e)))
            return {
                "error": str(e),
                "logs": logs,
            }
    
    def _analyze_code(self, state: ReviewState) -> dict:
        """Analyze code and find violations."""
        pr_number = state["pr_number"]
        logs = state.get("logs", [])
        
        if state.get("error"):
            return {}
        
        try:
            violations, comments, summary = self.review_agent.review_pr(pr_number)
            
            # Log each rule trigger
            for violation in violations:
                logs.append(add_log(
                    state,
                    "rule_triggered",
                    rule_id=violation.rule_id,
                    file=violation.file,
                    line=violation.line,
                    severity=violation.severity.value,
                    confidence=violation.confidence,
                ))
            
            logger.info(f"Found {len(violations)} violations")
            
            return {
                "violations": violations,
                "comments": comments,
                "summary": summary,
                "logs": logs,
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logs.append(add_log(state, "error", message=str(e)))
            return {
                "violations": [],
                "comments": [],
                "error": str(e),
                "logs": logs,
            }
    
    def _check_delegation(self, state: ReviewState) -> dict:
        """Check if refactoring agent should be invoked."""
        violations = state.get("violations", [])
        logs = state.get("logs", [])
        
        needs_refactor = self.review_agent.should_delegate_to_refactor(violations)
        
        if needs_refactor:
            logs.append(add_log(
                state,
                "delegation",
                reason="threshold_exceeded",
                target="refactor_agent",
            ))
            logger.info("Delegating to refactor agent")
        
        return {
            "needs_refactor": needs_refactor,
            "logs": logs,
        }
    
    def _should_refactor(self, state: ReviewState) -> Literal["refactor", "skip_refactor"]:
        """Conditional edge function for refactoring decision."""
        if state.get("needs_refactor", False):
            return "refactor"
        return "skip_refactor"
    
    def _refactor(self, state: ReviewState) -> dict:
        """Apply automated refactoring."""
        violations = state.get("violations", [])
        pr = state.get("pr")
        logs = state.get("logs", [])
        refactor_results = []
        
        if not pr or not self.ollama:
            return {"refactor_results": [], "logs": logs}
        
        # Group violations by file
        files_violations: dict[str, list[Violation]] = {}
        for v in violations:
            if v.file not in files_violations:
                files_violations[v.file] = []
            files_violations[v.file].append(v)
        
        # Attempt to refactor files with multiple violations
        from src.agents.refactor_agent import RefactorAgent
        refactor_agent = RefactorAgent(
            github_client=self.github,
            ollama_client=self.ollama,
        )
        
        for filename, file_violations in files_violations.items():
            if len(file_violations) < 2:
                continue
            
            try:
                result = refactor_agent.refactor_file(
                    pr_number=state["pr_number"],
                    filename=filename,
                    violations=file_violations,
                )
                
                if result and result.success:
                    refactor_results.append(result)
                    logs.append(add_log(
                        state,
                        "refactor_applied",
                        file=filename,
                        violations_fixed=len(result.violations_fixed),
                    ))
                    
            except Exception as e:
                logger.warning(f"Refactoring failed for {filename}: {e}")
                logs.append(add_log(
                    state,
                    "refactor_failed",
                    file=filename,
                    error=str(e),
                ))
        
        return {
            "refactor_results": refactor_results,
            "logs": logs,
        }
    
    def _post_comments(self, state: ReviewState) -> dict:
        """Post review comments to GitHub."""
        pr_number = state["pr_number"]
        comments = state.get("comments", [])
        summary = state.get("summary")
        refactor_results = state.get("refactor_results", [])
        logs = state.get("logs", [])
        
        if not summary:
            return {"logs": logs}
        
        # Update summary with refactoring info
        if refactor_results:
            summary.refactoring_performed = True
            summary.refactoring_files = [r.file for r in refactor_results]
        
        try:
            self.review_agent.post_review(pr_number, comments, summary)
            
            logs.append(add_log(
                state,
                "comments_posted",
                count=len(comments),
            ))
            
        except Exception as e:
            logger.error(f"Failed to post comments: {e}")
            logs.append(add_log(state, "error", message=str(e)))
        
        return {"logs": logs}
    
    def _generate_summary(self, state: ReviewState) -> dict:
        """Generate final summary and save logs."""
        logs = state.get("logs", [])
        violations = state.get("violations", [])
        comments = state.get("comments", [])
        
        logs.append(add_log(
            state,
            "review_completed",
            total_violations=len(violations),
            comments_posted=len(comments),
        ))
        
        # Save logs to file
        if self.log_path:
            self._save_logs(logs)
        
        return {"logs": logs}
    
    def _save_logs(self, logs: list[dict]) -> None:
        """Save logs to JSONL file."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                for log in logs:
                    f.write(json.dumps(log) + "\n")
            logger.info(f"Saved {len(logs)} log entries to {self.log_path}")
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")
    
    def run(self, pr_number: int) -> ReviewState:
        """
        Run the complete review workflow.
        
        Args:
            pr_number: Pull request number.
            
        Returns:
            Final state with all results.
        """
        initial_state: ReviewState = {
            "pr_number": pr_number,
            "pr": None,
            "files": [],
            "violations": [],
            "comments": [],
            "needs_refactor": False,
            "refactor_results": [],
            "summary": None,
            "logs": [],
            "error": None,
        }
        
        logger.info(f"Starting review workflow for PR #{pr_number}")
        
        # Run the graph
        final_state = self.app.invoke(initial_state)
        
        logger.info("Review workflow completed")
        return final_state
    
    def get_violations_json(self, state: ReviewState) -> list[dict]:
        """Export violations as JSON-serializable list."""
        return [v.to_dict() for v in state.get("violations", [])]
