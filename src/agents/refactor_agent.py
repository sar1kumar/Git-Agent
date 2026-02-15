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
        """
        Initialize the refactor agent.
        
        Args:
            github_client: GitHub API client.
            ollama_client: Ollama client for LLM-assisted refactoring.
            auto_commit: Whether to automatically commit changes.
        """
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
        """
        Refactor a file to fix violations.
        
        Args:
            pr_number: Pull request number.
            filename: File to refactor.
            violations: Violations to fix.
            
        Returns:
            RefactorResult or None if refactoring failed.
        """
        logger.info(f"Refactoring {filename} ({len(violations)} violations)")
        
        # Get PR details for branch
        pr = self.github.get_pull_request(pr_number)
        
        # Get current file contents
        original_code = self.github.get_file_contents(filename, ref=pr.head_branch)
        if original_code is None:
            logger.error(f"Could not fetch file contents: {filename}")
            return RefactorResult(
                file=filename,
                original_code="",
                refactored_code="",
                changes_description="Failed to fetch file",
                success=False,
                error_message="Could not fetch file contents",
            )
        
        # Apply automatic fixes
        refactored_code = original_code
        violations_fixed = []
        changes = []
        
        # Sort violations by line (descending) to avoid line number shifts
        sorted_violations = sorted(violations, key=lambda v: v.line, reverse=True)
        
        for violation in sorted_violations:
            fixed_code, fixed, change_desc = self._apply_fix(
                refactored_code,
                violation,
            )
            
            if fixed:
                refactored_code = fixed_code
                violations_fixed.append(violation.rule_id)
                changes.append(change_desc)
        
        # If no automatic fixes worked, try LLM
        if not violations_fixed and self.llm:
            try:
                llm_result = self._llm_refactor(original_code, violations)
                if llm_result:
                    refactored_code = llm_result["refactored_code"]
                    changes.append(llm_result.get("explanation", "LLM refactoring applied"))
                    violations_fixed = [v.rule_id for v in violations]
            except Exception as e:
                logger.warning(f"LLM refactoring failed: {e}")
        
        # Check if anything changed
        if refactored_code == original_code:
            logger.info(f"No changes applied to {filename}")
            return RefactorResult(
                file=filename,
                original_code=original_code,
                refactored_code=original_code,
                changes_description="No automatic fixes available",
                success=False,
            )
        
        # Commit changes if enabled
        if self.auto_commit:
            commit_message = f"🤖 Auto-fix: {', '.join(violations_fixed[:3])}"
            if len(violations_fixed) > 3:
                commit_message += f" (+{len(violations_fixed) - 3} more)"
            
            success = self.github.commit_file(
                pr_number,
                filename,
                refactored_code,
                commit_message,
            )
            
            if not success:
                logger.warning(f"Failed to commit changes to {filename}")
        
        return RefactorResult(
            file=filename,
            original_code=original_code,
            refactored_code=refactored_code,
            changes_description="\n".join(changes),
            violations_fixed=violations_fixed,
            success=True,
        )
    
    def _apply_fix(
        self,
        code: str,
        violation: Violation,
    ) -> tuple[str, bool, str]:
        """
        Apply an automatic fix for a violation.
        
        Args:
            code: Current code.
            violation: Violation to fix.
            
        Returns:
            Tuple of (new_code, was_fixed, change_description).
        """
        lines = code.split("\n")
        line_idx = violation.line - 1
        
        if line_idx < 0 or line_idx >= len(lines):
            return code, False, ""
        
        line = lines[line_idx]
        
        # Apply rule-specific fixes
        if violation.rule_id == "S001":  # Line length
            return self._fix_line_length(lines, line_idx, violation)
        
        elif violation.rule_id == "S002":  # Naming conventions
            return self._fix_naming(lines, line_idx, violation)
        
        elif violation.rule_id == "BP001":  # Bare except
            return self._fix_bare_except(lines, line_idx, violation)
        
        elif violation.rule_id == "SEC001":  # Hardcoded secrets
            return self._fix_hardcoded_secret(lines, line_idx, violation)
        
        return code, False, ""
    
    def _fix_line_length(
        self,
        lines: list[str],
        line_idx: int,
        violation: Violation,
    ) -> tuple[str, bool, str]:
        """Attempt to fix long line by breaking it."""
        line = lines[line_idx]
        
        # Try breaking at common points
        # For now, just add a comment suggesting a fix
        # More sophisticated breaking would need AST analysis
        
        return "\n".join(lines), False, ""
    
    def _fix_naming(
        self,
        lines: list[str],
        line_idx: int,
        violation: Violation,
    ) -> tuple[str, bool, str]:
        """Fix naming convention violations."""
        line = lines[line_idx]
        
        # Extract the suggested name from the violation
        suggestion = violation.suggestion
        if not suggestion or "Rename to" not in suggestion:
            return "\n".join(lines), False, ""
        
        # Parse suggestion
        match = re.search(r"Rename to '(\w+)'", suggestion)
        if not match:
            return "\n".join(lines), False, ""
        
        new_name = match.group(1)
        
        # Extract current name from line
        func_match = re.match(r"(\s*(?:async\s+)?def\s+)(\w+)(\s*\()", line)
        class_match = re.match(r"(\s*class\s+)(\w+)(\s*[:\(])", line)
        
        if func_match:
            old_name = func_match.group(2)
            new_line = f"{func_match.group(1)}{new_name}{func_match.group(3)}"
            lines[line_idx] = new_line + line[len(func_match.group(0)):]
            
            # Also rename all references in the file
            code = "\n".join(lines)
            code = re.sub(rf"\b{old_name}\b", new_name, code)
            
            return code, True, f"Renamed function '{old_name}' to '{new_name}'"
        
        elif class_match:
            old_name = class_match.group(2)
            new_line = f"{class_match.group(1)}{new_name}{class_match.group(3)}"
            lines[line_idx] = new_line + line[len(class_match.group(0)):]
            
            # Also rename all references
            code = "\n".join(lines)
            code = re.sub(rf"\b{old_name}\b", new_name, code)
            
            return code, True, f"Renamed class '{old_name}' to '{new_name}'"
        
        return "\n".join(lines), False, ""
    
    def _fix_bare_except(
        self,
        lines: list[str],
        line_idx: int,
        violation: Violation,
    ) -> tuple[str, bool, str]:
        """Fix bare except clauses."""
        line = lines[line_idx]
        
        # Replace 'except:' with 'except Exception as e:'
        if re.match(r"^\s*except\s*:", line):
            indent = len(line) - len(line.lstrip())
            lines[line_idx] = " " * indent + "except Exception as e:"
            return "\n".join(lines), True, "Changed bare 'except:' to 'except Exception as e:'"
        
        return "\n".join(lines), False, ""
    
    def _fix_hardcoded_secret(
        self,
        lines: list[str],
        line_idx: int,
        violation: Violation,
    ) -> tuple[str, bool, str]:
        """Replace hardcoded secrets with environment variable lookups."""
        line = lines[line_idx]
        
        # Pattern to match secret assignments
        patterns = [
            (r"(\w*password\w*)\s*=\s*['\"][^'\"]+['\"]", "PASSWORD"),
            (r"(\w*api_key\w*)\s*=\s*['\"][^'\"]+['\"]", "API_KEY"),
            (r"(\w*secret\w*)\s*=\s*['\"][^'\"]+['\"]", "SECRET"),
            (r"(\w*token\w*)\s*=\s*['\"][^'\"]+['\"]", "TOKEN"),
        ]
        
        for pattern, env_suffix in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                var_name = match.group(1)
                env_var = var_name.upper()
                if not env_var.endswith(env_suffix):
                    env_var = f"{var_name.upper()}_{env_suffix}"
                
                # Replace with os.environ.get
                indent = len(line) - len(line.lstrip())
                new_line = f"{' ' * indent}{var_name} = os.environ.get('{env_var}')"
                lines[line_idx] = new_line
                
                # Check if os import exists, if not we'd need to add it
                # For now, just make the change
                
                return (
                    "\n".join(lines),
                    True,
                    f"Replaced hardcoded {var_name} with os.environ.get('{env_var}')",
                )
        
        return "\n".join(lines), False, ""
    
    def _llm_refactor(
        self,
        code: str,
        violations: list[Violation],
    ) -> Optional[dict]:
        """Use LLM to refactor code."""
        if not self.llm:
            return None
        
        issues = [f"Line {v.line}: {v.message}" for v in violations]
        
        result = self.llm.suggest_refactoring(code, issues)
        
        if result and result.get("refactored_code"):
            return result
        
        return None
    
    def extract_method(
        self,
        code: str,
        start_line: int,
        end_line: int,
        method_name: str,
    ) -> tuple[str, bool]:
        """
        Extract a block of code into a separate method.
        
        Args:
            code: Full file code.
            start_line: Starting line of block to extract.
            end_line: Ending line of block to extract.
            method_name: Name for the new method.
            
        Returns:
            Tuple of (new_code, success).
        """
        if not self.llm:
            return code, False
        
        lines = code.split("\n")
        
        if start_line < 1 or end_line > len(lines):
            return code, False
        
        block = "\n".join(lines[start_line - 1:end_line])
        
        prompt = f"""Extract this code block into a new method called '{method_name}'.
Determine the parameters needed and return value.

Code block:
```
{block}
```

Provide the new method definition and the call to replace the original block.
Format as JSON with 'new_method' and 'call_replacement' keys."""

        try:
            response = self.llm.generate(prompt)
            result = self.llm._parse_json_response(response)
            
            if result.get("new_method") and result.get("call_replacement"):
                # Insert new method before the extracted block
                # and replace block with call
                # This is simplified - real implementation would need proper indentation
                return code, False  # Simplified for now
                
        except Exception as e:
            logger.warning(f"Extract method failed: {e}")
        
        return code, False
    
    def simplify_conditional(
        self,
        code: str,
        line_number: int,
    ) -> tuple[str, bool]:
        """
        Simplify a complex conditional expression.
        
        Args:
            code: Full file code.
            line_number: Line with the conditional.
            
        Returns:
            Tuple of (new_code, success).
        """
        if not self.llm:
            return code, False
        
        lines = code.split("\n")
        
        if line_number < 1 or line_number > len(lines):
            return code, False
        
        # Get context around the line
        start = max(0, line_number - 5)
        end = min(len(lines), line_number + 10)
        context = "\n".join(lines[start:end])
        
        prompt = f"""Simplify the conditional on line {line_number - start + 1} of this code:

```python
{context}
```

Make it more readable. Provide only the simplified code block."""

        try:
            response = self.llm.generate(prompt, max_tokens=1024)
            # Would need to parse and apply the simplified code
            return code, False  # Simplified for now
            
        except Exception as e:
            logger.warning(f"Simplify conditional failed: {e}")
        
        return code, False
