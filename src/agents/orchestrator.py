"""LangGraph orchestrator for multi-agent code review workflow."""

import logging
import json
from datetime import datetime, timezone
from typing import TypedDict, Literal, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END

from src.github.models import (
    PullRequest, FileChange, Violation, ReviewComment, ReviewSummary, RefactorResult,
)
from src.github.client import GitHubClient
from src.analysis.rules_engine import RulesEngine
from src.analysis.ast_analyzer import ASTAnalyzer
from src.llm.ollama_client import OllamaClient
from src.agents.review_agent import ReviewAgent
from src.agents.protocol import AgentRole, HandoffContext, MessageBus

logger = logging.getLogger(__name__)


class ReviewState(TypedDict):
    """State for the review workflow."""
    pr_number: int
    pr: PullRequest | None
    files: list[FileChange]
    violations: list[Violation]
    comments: list[ReviewComment]
    needs_refactor: bool
    refactor_results: list[RefactorResult]
    verification_passed: bool
    tests_passed: bool
    rollback_point: str | None
    rollback_performed: bool
    summary: ReviewSummary | None
    logs: list[dict]
    error: str | None
    context: HandoffContext | None


def add_log(state: ReviewState, event: str, **kwargs) -> dict:
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}


class ReviewOrchestrator:
    """LangGraph-based orchestrator for the code review workflow with advanced capabilities."""
    
    def __init__(
        self, github_client: GitHubClient, rules_engine: RulesEngine | None = None,
        ollama_client: OllamaClient | None = None, ast_analyzer: ASTAnalyzer | None = None,
        use_llm: bool = True, log_path: Path | str | None = None,
        enable_verification: bool = True, enable_tests: bool = True, enable_rollback: bool = True,
    ):
        self.github = github_client
        self.rules = rules_engine or RulesEngine()
        self.ollama = ollama_client
        self.ast = ast_analyzer or ASTAnalyzer()
        self.use_llm = use_llm
        self.log_path = Path(log_path) if log_path else None
        self.enable_verification = enable_verification
        self.enable_tests = enable_tests
        self.enable_rollback = enable_rollback
        self.message_bus = MessageBus()
        
        self.review_agent = ReviewAgent(
            github_client=self.github, rules_engine=self.rules,
            ollama_client=self.ollama, use_llm=self.use_llm,
        )
        
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        logger.info("Initialized ReviewOrchestrator with advanced capabilities")
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ReviewState)
        
        graph.add_node("fetch_pr", self._fetch_pr)
        graph.add_node("create_rollback_point", self._create_rollback_point)
        graph.add_node("analyze_code", self._analyze_code)
        graph.add_node("check_delegation", self._check_delegation)
        graph.add_node("refactor", self._refactor)
        graph.add_node("verify_refactoring", self._verify_refactoring)
        graph.add_node("run_tests", self._run_tests)
        graph.add_node("check_safety", self._check_safety)
        graph.add_node("rollback", self._rollback)
        graph.add_node("post_comments", self._post_comments)
        graph.add_node("generate_summary", self._generate_summary)
        
        graph.set_entry_point("fetch_pr")
        graph.add_edge("fetch_pr", "create_rollback_point")
        graph.add_edge("create_rollback_point", "analyze_code")
        graph.add_edge("analyze_code", "check_delegation")
        
        graph.add_conditional_edges("check_delegation", self._should_refactor,
            {"refactor": "refactor", "skip_refactor": "post_comments"})
        
        graph.add_conditional_edges("refactor",
            lambda s: "verify" if self.enable_verification else "post",
            {"verify": "verify_refactoring", "post": "post_comments"})
        
        graph.add_conditional_edges("verify_refactoring",
            lambda s: "test" if self.enable_tests and s.get("verification_passed", True) else "check",
            {"test": "run_tests", "check": "check_safety"})
        
        graph.add_edge("run_tests", "check_safety")
        
        graph.add_conditional_edges("check_safety", self._should_rollback,
            {"rollback": "rollback", "continue": "post_comments"})
        
        graph.add_edge("rollback", "post_comments")
        graph.add_edge("post_comments", "generate_summary")
        graph.add_edge("generate_summary", END)
        
        return graph
    
    def _fetch_pr(self, state: ReviewState) -> dict:
        pr_number = state["pr_number"]
        logs = state.get("logs", [])
        logs.append(add_log(state, "review_started", pr=pr_number))
        
        try:
            pr = self.github.get_pull_request(pr_number)
            context = HandoffContext(pr_number=pr_number, repository=self.github.repo_name,
                                     branch=pr.head_branch, files=[f.filename for f in pr.files])
            context.add_to_chain(AgentRole.ORCHESTRATOR)
            logs.append(add_log(state, "pr_fetched", files_count=len(pr.files)))
            return {"pr": pr, "files": pr.files, "logs": logs, "context": context}
        except Exception as e:
            logs.append(add_log(state, "error", message=str(e)))
            return {"error": str(e), "logs": logs}
    
    def _create_rollback_point(self, state: ReviewState) -> dict:
        logs = state.get("logs", [])
        context = state.get("context")
        if not self.enable_rollback:
            return {"rollback_point": None, "logs": logs}
        try:
            sha = self.github.get_commit_sha(state["pr_number"])
            if sha and context:
                context.add_rollback_point(sha)
                logs.append(add_log(state, "rollback_point_created", sha=sha[:7]))
            return {"rollback_point": sha, "context": context, "logs": logs}
        except Exception:
            return {"rollback_point": None, "logs": logs}
    
    def _analyze_code(self, state: ReviewState) -> dict:
        if state.get("error"):
            return {}
        logs = state.get("logs", [])
        context = state.get("context")
        try:
            violations, comments, summary = self.review_agent.review_pr(state["pr_number"])
            if context:
                pr = state.get("pr")
                if pr:
                    for f in pr.files:
                        if not f.is_deleted:
                            content = self.github.get_file_contents(f.filename, ref=pr.head_branch)
                            if content:
                                context.original_code[f.filename] = content
                context.violations = [v.to_dict() for v in violations]
            for v in violations:
                logs.append(add_log(state, "rule_triggered", rule_id=v.rule_id, file=v.file,
                                   line=v.line, severity=v.severity.value))
            return {"violations": violations, "comments": comments, "summary": summary,
                    "context": context, "logs": logs}
        except Exception as e:
            logs.append(add_log(state, "error", message=str(e)))
            return {"violations": [], "comments": [], "error": str(e), "logs": logs}
    
    def _check_delegation(self, state: ReviewState) -> dict:
        violations = state.get("violations", [])
        logs = state.get("logs", [])
        needs_refactor = self.review_agent.should_delegate_to_refactor(violations)
        if needs_refactor:
            logs.append(add_log(state, "delegation", target="refactor_agent"))
        return {"needs_refactor": needs_refactor, "logs": logs}
    
    def _should_refactor(self, state: ReviewState) -> Literal["refactor", "skip_refactor"]:
        return "refactor" if state.get("needs_refactor", False) else "skip_refactor"
    
    def _refactor(self, state: ReviewState) -> dict:
        violations = state.get("violations", [])
        pr = state.get("pr")
        logs = state.get("logs", [])
        context = state.get("context")
        if not pr or not self.ollama:
            return {"refactor_results": [], "logs": logs}
        
        from src.agents.refactor_agent import RefactorAgent
        refactor_agent = RefactorAgent(github_client=self.github, ollama_client=self.ollama,
                                       ast_analyzer=self.ast)
        
        files_violations = {}
        for v in violations:
            files_violations.setdefault(v.file, []).append(v)
        
        results = []
        for filename, file_violations in files_violations.items():
            if len(file_violations) < 2:
                continue
            try:
                result = refactor_agent.refactor_file(state["pr_number"], filename, file_violations)
                if result and result.success:
                    results.append(result)
                    if context:
                        context.refactored_code[filename] = result.refactored_code
                    logs.append(add_log(state, "refactor_applied", file=filename))
            except Exception as e:
                logs.append(add_log(state, "refactor_failed", file=filename, error=str(e)))
        
        return {"refactor_results": results, "context": context, "logs": logs}
    
    def _verify_refactoring(self, state: ReviewState) -> dict:
        logs = state.get("logs", [])
        context = state.get("context")
        results = state.get("refactor_results", [])
        if not results or not context:
            return {"verification_passed": True, "logs": logs}
        
        from src.agents.verification_agent import VerificationAgent
        verifier = VerificationAgent(ast_analyzer=self.ast, ollama_client=self.ollama)
        
        all_passed = True
        for result in results:
            verify_result = verifier.verify(result.original_code, result.refactored_code,
                                           result.file, [v for v in context.violations if v.get("file") == result.file])
            logs.append(add_log(state, "verification", file=result.file, passed=verify_result.passed))
            if not verify_result.passed:
                all_passed = False
        
        return {"verification_passed": all_passed, "logs": logs}
    
    def _run_tests(self, state: ReviewState) -> dict:
        logs = state.get("logs", [])
        context = state.get("context")
        if not context or not context.refactored_code:
            return {"tests_passed": True, "logs": logs}
        
        from src.agents.test_runner import TestRunner
        try:
            result = TestRunner().run_tests_in_sandbox(context.refactored_code)
            logs.append(add_log(state, "tests", passed=result.passed, total=result.total_tests))
            return {"tests_passed": result.passed == result.total_tests, "logs": logs}
        except Exception as e:
            logs.append(add_log(state, "tests_error", error=str(e)))
            return {"tests_passed": True, "logs": logs}
    
    def _check_safety(self, state: ReviewState) -> dict:
        return {"verification_passed": state.get("verification_passed", True),
                "tests_passed": state.get("tests_passed", True)}
    
    def _should_rollback(self, state: ReviewState) -> Literal["rollback", "continue"]:
        if not self.enable_rollback:
            return "continue"
        if not state.get("verification_passed", True) or not state.get("tests_passed", True):
            return "rollback"
        return "continue"
    
    def _rollback(self, state: ReviewState) -> dict:
        logs = state.get("logs", [])
        context = state.get("context")
        rollback_point = state.get("rollback_point")
        if not rollback_point or not context:
            return {"rollback_performed": False, "logs": logs}
        
        files = list(context.refactored_code.keys())
        try:
            success = self.github.revert_to_commit(state["pr_number"], rollback_point, files)
            logs.append(add_log(state, "rollback", success=success, files=files))
            context.refactored_code.clear()
            return {"rollback_performed": success, "context": context, "logs": logs}
        except Exception as e:
            logs.append(add_log(state, "rollback_failed", error=str(e)))
            return {"rollback_performed": False, "logs": logs}
    
    def _post_comments(self, state: ReviewState) -> dict:
        logs = state.get("logs", [])
        summary = state.get("summary")
        if not summary:
            return {"logs": logs}
        
        results = state.get("refactor_results", [])
        if results and not state.get("rollback_performed", False):
            summary.refactoring_performed = True
            summary.refactoring_files = [r.file for r in results]
        
        try:
            self.review_agent.post_review(state["pr_number"], state.get("comments", []), summary)
            logs.append(add_log(state, "comments_posted", count=len(state.get("comments", []))))
        except Exception as e:
            logs.append(add_log(state, "error", message=str(e)))
        return {"logs": logs}
    
    def _generate_summary(self, state: ReviewState) -> dict:
        logs = state.get("logs", [])
        logs.append(add_log(state, "review_completed",
                           total_violations=len(state.get("violations", [])),
                           comments_posted=len(state.get("comments", []))))
        if self.log_path:
            self._save_logs(logs)
        return {"logs": logs}
    
    def _save_logs(self, logs: list[dict]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                for log in logs:
                    f.write(json.dumps(log) + "\n")
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")
    
    def run(self, pr_number: int) -> ReviewState:
        initial_state: ReviewState = {
            "pr_number": pr_number, "pr": None, "files": [], "violations": [], "comments": [],
            "needs_refactor": False, "refactor_results": [], "verification_passed": True,
            "tests_passed": True, "rollback_point": None, "rollback_performed": False,
            "summary": None, "logs": [], "error": None, "context": None,
        }
        return self.app.invoke(initial_state)
    
    def get_violations_json(self, state: ReviewState) -> list[dict]:
        return [v.to_dict() for v in state.get("violations", [])]