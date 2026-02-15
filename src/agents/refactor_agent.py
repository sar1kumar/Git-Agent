"""Refactoring agent for applying automated fixes."""

import logging
import re
from typing import Optional

from src.github.models import Violation, RefactorResult, Severity
from src.github.client import GitHubClient
from src.llm.ollama_client import OllamaClient
from src.analysis.ast_analyzer import ASTAnalyzer
logger = logging.getLogger(__name__)


class RefactorAgent:
    """Agent for applying automated code refactoring."""

    def __init__(
        self,
        github_client: GitHubClient,
        ollama_client: Optional[OllamaClient] = None,
        ast_analyzer: Optional[ASTAnalyzer] = None,
        auto_commit: bool = False,
    ):
        self.github = github_client
        self.llm = ollama_client
        self.ast = ast_analyzer or ASTAnalyzer()
        self.auto_commit = auto_commit
        logger.info(f"Initialized RefactorAgent (auto_commit={auto_commit})")

    def refactor_file(
        self,
        pr_number: int,
        filename: str,
        violations: list[Violation],
    ) -> Optional[RefactorResult]:
        """Refactor a file to fix violations."""
        logger.info(f"Refactoring {filename} ({len(violations)} violations)")

        pr = self.github.get_pull_request(pr_number)
        original_code = self.github.get_file_contents(filename, ref=pr.head_branch)

        if original_code is None:
            return RefactorResult(
                file=filename, original_code="", refactored_code="",
                changes_description="Failed to fetch file",
                success=False, error_message="Could not fetch file contents",
            )

        refactored_code = original_code
        violations_fixed = []
        changes = []

        # Sort violations by line (descending) to avoid line number shifts
        for violation in sorted(violations, key=lambda v: v.line, reverse=True):
            fixed_code, fixed, change_desc = self._apply_fix(refactored_code, violation)
            if fixed:
                refactored_code = fixed_code
                violations_fixed.append(violation.rule_id)
                changes.append(change_desc)

        # Fall back to LLM if no automatic fixes worked
        if not violations_fixed and self.llm:
            try:
                llm_result = self._llm_refactor(original_code, violations)
                if llm_result:
                    refactored_code = llm_result["refactored_code"]
                    changes.append(llm_result.get("explanation", "LLM refactoring applied"))
                    violations_fixed = [v.rule_id for v in violations]
            except Exception as e:
                logger.warning(f"LLM refactoring failed: {e}")

        if refactored_code == original_code:
            return RefactorResult(
                file=filename, original_code=original_code, refactored_code=original_code,
                changes_description="No automatic fixes available", success=False,
            )

        # Ensure any new imports are present
        refactored_code = self._ensure_imports(refactored_code, changes)

        if self.auto_commit:
            msg = f"Auto-fix: {', '.join(violations_fixed[:3])}"
            self.github.commit_file(pr_number, filename, refactored_code, msg)

        return RefactorResult(
            file=filename, original_code=original_code, refactored_code=refactored_code,
            changes_description="\n".join(changes), violations_fixed=violations_fixed,
            success=True,
        )

    def _apply_fix(self, code: str, violation: Violation) -> tuple[str, bool, str]:
        """Apply an automatic fix for a violation."""
        lines = code.split("\n")
        line_idx = violation.line - 1

        if line_idx < 0 or line_idx >= len(lines):
            return code, False, ""

        if violation.rule_id == "S002":
            return self._fix_naming(lines, line_idx, violation)
        elif violation.rule_id == "BP001":
            return self._fix_bare_except(lines, line_idx, violation)
        elif violation.rule_id == "SEC001":
            return self._fix_hardcoded_secret(lines, line_idx, violation)

        return code, False, ""

    def _fix_naming(self, lines: list[str], line_idx: int, violation: Violation) -> tuple[str, bool, str]:
        """Fix naming convention violations by renaming the symbol."""
        suggestion = violation.suggestion
        if not suggestion or "Rename to" not in suggestion:
            return "\n".join(lines), False, ""

        match = re.search(r"Rename to '(\w+)'", suggestion)
        if not match:
            return "\n".join(lines), False, ""

        new_name = match.group(1)
        line = lines[line_idx]

        func_match = re.match(r"(\s*(?:async\s+)?def\s+)(\w+)(\s*\()", line)
        class_match = re.match(r"(\s*class\s+)(\w+)(\s*[:\(])", line)

        old_name = None
        if func_match:
            old_name = func_match.group(2)
        elif class_match:
            old_name = class_match.group(2)

        if old_name:
            code = "\n".join(lines)
            code = re.sub(rf"\b{re.escape(old_name)}\b", new_name, code)
            return code, True, f"Renamed '{old_name}' to '{new_name}'"

        return "\n".join(lines), False, ""

    def _fix_bare_except(self, lines: list[str], line_idx: int, violation: Violation) -> tuple[str, bool, str]:
        """Fix bare except clauses."""
        line = lines[line_idx]
        if re.match(r"^\s*except\s*:", line):
            indent = len(line) - len(line.lstrip())
            lines[line_idx] = " " * indent + "except Exception as e:"
            return "\n".join(lines), True, "Changed bare 'except:' to 'except Exception as e:'"
        return "\n".join(lines), False, ""

    def _fix_hardcoded_secret(self, lines: list[str], line_idx: int, violation: Violation) -> tuple[str, bool, str]:
        """Replace hardcoded secrets with environment variable lookups."""
        line = lines[line_idx]
        patterns = [
            (r"(\w*password\w*)\s*=\s*['\"][^'\"]+['\"]", "PASSWORD"),
            (r"(\w*api_key\w*)\s*=\s*['\"][^'\"]+['\"]", "API_KEY"),
            (r"(\w*secret\w*)\s*=\s*['\"][^'\"]+['\"]", "SECRET"),
            (r"(\w*token\w*)\s*=\s*['\"][^'\"]+['\"]", "TOKEN"),
        ]

        for pattern, suffix in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                var_name = match.group(1)
                env_var = var_name.upper()
                if not env_var.endswith(suffix):
                    env_var = f"{var_name.upper()}_{suffix}"
                indent = len(line) - len(line.lstrip())
                lines[line_idx] = f"{' ' * indent}{var_name} = os.environ.get('{env_var}')"
                return "\n".join(lines), True, f"Replaced hardcoded {var_name} with os.environ.get('{env_var}')"

        return "\n".join(lines), False, ""

    def _ensure_imports(self, code: str, changes: list[str]) -> str:
        """Ensure required imports are present after refactoring."""
        if "os.environ" in code and "import os" not in code:
            lines = code.split("\n")
            # Find the right place to insert (after existing imports)
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_idx = i + 1
                elif line.strip() and not line.startswith("#"):
                    break
            lines.insert(insert_idx, "import os")
            changes.append("Added: import os")
            return "\n".join(lines)
        return code

    def _llm_refactor(self, code: str, violations: list[Violation]) -> Optional[dict]:
        """Use LLM to refactor code."""
        if not self.llm:
            return None
        issues = [f"Line {v.line}: {v.message}" for v in violations]
        result = self.llm.suggest_refactoring(code, issues)
        if result and result.get("refactored_code"):
            return result
        return None
