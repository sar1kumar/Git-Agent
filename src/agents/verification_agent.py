"""Verification agent for safety checks before committing refactored code."""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from src.analysis.ast_analyzer import ASTAnalyzer
from src.llm.ollama_client import OllamaClient
from src.agents.protocol import (
    AgentMessage,
    AgentRole,
    MessageType,
    HandoffContext,
)

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Status of verification checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class VerificationCheck:
    """Result of a single verification check."""
    name: str
    status: VerificationStatus
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Complete verification result."""
    passed: bool
    checks: list[VerificationCheck] = field(default_factory=list)
    overall_message: str = ""
    safe_to_commit: bool = False
    requires_human_review: bool = False
    
    def add_check(self, check: VerificationCheck) -> None:
        """Add a check result."""
        self.checks.append(check)
        if check.status == VerificationStatus.FAILED:
            self.passed = False
            self.safe_to_commit = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "overall_message": self.overall_message,
            "safe_to_commit": self.safe_to_commit,
            "requires_human_review": self.requires_human_review,
        }


class VerificationAgent:
    """
    Agent that verifies refactored code before it gets committed.
    
    Performs multiple safety checks:
    1. Syntax validation
    2. AST comparison (structure preserved)
    3. Semantic equivalence check (via LLM)
    4. No new violations introduced
    5. Import integrity
    """
    
    def __init__(
        self,
        ast_analyzer: Optional[ASTAnalyzer] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize the verification agent.
        
        Args:
            ast_analyzer: AST analyzer for code parsing.
            ollama_client: Ollama client for semantic analysis.
        """
        self.ast = ast_analyzer or ASTAnalyzer()
        self.llm = ollama_client
        logger.info("Initialized VerificationAgent")
    
    def verify(
        self,
        original_code: str,
        refactored_code: str,
        filename: str,
        violations_to_fix: list[dict],
    ) -> VerificationResult:
        """
        Verify that refactored code is safe to commit.
        
        Args:
            original_code: Original source code.
            refactored_code: Refactored source code.
            filename: Name of the file.
            violations_to_fix: Violations that the refactoring should fix.
            
        Returns:
            VerificationResult with all check results.
        """
        result = VerificationResult(passed=True)
        
        # Check 1: Syntax validation
        syntax_check = self._check_syntax(refactored_code, filename)
        result.add_check(syntax_check)
        
        if syntax_check.status == VerificationStatus.FAILED:
            result.overall_message = "Refactored code has syntax errors"
            return result
        
        # Check 2: AST structure comparison
        ast_check = self._check_ast_structure(original_code, refactored_code, filename)
        result.add_check(ast_check)
        
        # Check 3: Import integrity
        import_check = self._check_imports(original_code, refactored_code, filename)
        result.add_check(import_check)
        
        # Check 4: No critical changes
        critical_check = self._check_no_critical_changes(original_code, refactored_code)
        result.add_check(critical_check)
        
        # Check 5: Semantic equivalence (if LLM available)
        if self.llm:
            semantic_check = self._check_semantic_equivalence(
                original_code, refactored_code, violations_to_fix
            )
            result.add_check(semantic_check)
            
            if semantic_check.status == VerificationStatus.WARNING:
                result.requires_human_review = True
        
        # Determine overall result
        failed_checks = [c for c in result.checks if c.status == VerificationStatus.FAILED]
        warning_checks = [c for c in result.checks if c.status == VerificationStatus.WARNING]
        
        if failed_checks:
            result.passed = False
            result.safe_to_commit = False
            result.overall_message = f"Verification failed: {len(failed_checks)} checks failed"
        elif warning_checks:
            result.passed = True
            result.safe_to_commit = True
            result.requires_human_review = True
            result.overall_message = f"Verification passed with {len(warning_checks)} warnings"
        else:
            result.passed = True
            result.safe_to_commit = True
            result.overall_message = "All verification checks passed"
        
        logger.info(f"Verification complete: {result.overall_message}")
        return result
    
    def _check_syntax(self, code: str, filename: str) -> VerificationCheck:
        """Check that the code has valid syntax."""
        try:
            compile(code, filename, 'exec')
            return VerificationCheck(
                name="syntax_validation",
                status=VerificationStatus.PASSED,
                message="Code has valid Python syntax",
            )
        except SyntaxError as e:
            return VerificationCheck(
                name="syntax_validation",
                status=VerificationStatus.FAILED,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                details={"line": e.lineno, "offset": e.offset, "text": e.text},
            )
    
    def _check_ast_structure(
        self,
        original: str,
        refactored: str,
        filename: str,
    ) -> VerificationCheck:
        """Check that AST structure is reasonably preserved."""
        orig_ast = self.ast.analyze(original, filename)
        refact_ast = self.ast.analyze(refactored, filename)
        
        issues = []
        
        # Check functions
        orig_funcs = {f.name for f in orig_ast.functions}
        refact_funcs = {f.name for f in refact_ast.functions}
        
        removed_funcs = orig_funcs - refact_funcs
        if removed_funcs:
            # Functions can be renamed, so this is a warning
            issues.append(f"Functions removed/renamed: {removed_funcs}")
        
        # Check classes
        orig_classes = {c.name for c in orig_ast.classes}
        refact_classes = {c.name for c in refact_ast.classes}
        
        removed_classes = orig_classes - refact_classes
        if removed_classes:
            issues.append(f"Classes removed/renamed: {removed_classes}")
        
        # Check that we didn't significantly increase complexity
        if refact_ast.complexity_score > orig_ast.complexity_score * 1.5:
            issues.append(f"Complexity increased significantly: {orig_ast.complexity_score} -> {refact_ast.complexity_score}")
        
        if issues:
            return VerificationCheck(
                name="ast_structure",
                status=VerificationStatus.WARNING,
                message="AST structure changed significantly",
                details={"issues": issues},
            )
        
        return VerificationCheck(
            name="ast_structure",
            status=VerificationStatus.PASSED,
            message="AST structure is reasonably preserved",
            details={
                "original_functions": len(orig_funcs),
                "refactored_functions": len(refact_funcs),
                "original_complexity": orig_ast.complexity_score,
                "refactored_complexity": refact_ast.complexity_score,
            },
        )
    
    def _check_imports(
        self,
        original: str,
        refactored: str,
        filename: str,
    ) -> VerificationCheck:
        """Check that imports are preserved or appropriately modified."""
        orig_ast = self.ast.analyze(original, filename)
        refact_ast = self.ast.analyze(refactored, filename)
        
        orig_imports = {imp.module for imp in orig_ast.imports}
        refact_imports = {imp.module for imp in refact_ast.imports}
        
        removed_imports = orig_imports - refact_imports
        added_imports = refact_imports - orig_imports
        
        # Removing imports is usually fine (unused imports)
        # Adding imports is also usually fine (new dependencies for refactored code)
        
        if removed_imports:
            return VerificationCheck(
                name="import_integrity",
                status=VerificationStatus.WARNING,
                message=f"Some imports were removed: {removed_imports}",
                details={"removed": list(removed_imports), "added": list(added_imports)},
            )
        
        return VerificationCheck(
            name="import_integrity",
            status=VerificationStatus.PASSED,
            message="Import integrity maintained",
            details={"added_imports": list(added_imports)},
        )
    
    def _check_no_critical_changes(
        self,
        original: str,
        refactored: str,
    ) -> VerificationCheck:
        """Check for critical changes that should never happen automatically."""
        issues = []
        
        # Check for deletion of security-related code
        security_patterns = [
            (r'\bauth\w*\b', "authentication"),
            (r'\bverify\w*\b', "verification"),
            (r'\bencrypt\w*\b', "encryption"),
            (r'\bdecrypt\w*\b', "decryption"),
            (r'\bhash\w*\b', "hashing"),
            (r'\bsanitize\w*\b', "sanitization"),
            (r'\bvalidate\w*\b', "validation"),
        ]
        
        for pattern, name in security_patterns:
            orig_matches = len(re.findall(pattern, original, re.IGNORECASE))
            refact_matches = len(re.findall(pattern, refactored, re.IGNORECASE))
            
            if orig_matches > 0 and refact_matches == 0:
                issues.append(f"All {name} code was removed")
        
        # Check for removal of error handling
        orig_try = original.count('try:')
        refact_try = refactored.count('try:')
        
        if orig_try > 0 and refact_try == 0:
            issues.append("All try/except blocks were removed")
        
        # Check for removal of logging
        orig_log = len(re.findall(r'\blogger\.\w+\(', original))
        refact_log = len(re.findall(r'\blogger\.\w+\(', refactored))
        
        if orig_log > 0 and refact_log == 0:
            issues.append("All logging statements were removed")
        
        if issues:
            return VerificationCheck(
                name="critical_changes",
                status=VerificationStatus.FAILED,
                message="Critical code may have been inappropriately removed",
                details={"issues": issues},
            )
        
        return VerificationCheck(
            name="critical_changes",
            status=VerificationStatus.PASSED,
            message="No critical changes detected",
        )
    
    def _check_semantic_equivalence(
        self,
        original: str,
        refactored: str,
        violations: list[dict],
    ) -> VerificationCheck:
        """Use LLM to check semantic equivalence."""
        if not self.llm:
            return VerificationCheck(
                name="semantic_equivalence",
                status=VerificationStatus.SKIPPED,
                message="LLM not available for semantic check",
            )
        
        violation_descriptions = [v.get("message", "") for v in violations[:5]]
        
        prompt = f"""Analyze these two code versions and determine if they are semantically equivalent
(produce the same behavior), accounting for the fact that the refactored version was modified to fix these issues:
{', '.join(violation_descriptions)}

ORIGINAL CODE:
```python
{original[:2000]}
```

REFACTORED CODE:
```python
{refactored[:2000]}
```

Respond with JSON:
{{
    "equivalent": true/false,
    "confidence": 0.0-1.0,
    "differences": ["list of behavioral differences if any"],
    "assessment": "brief explanation"
}}"""

        try:
            response = self.llm.generate(prompt, max_tokens=1024)
            result = self.llm._parse_json_response(response)
            
            is_equivalent = result.get("equivalent", False)
            confidence = result.get("confidence", 0.5)
            differences = result.get("differences", [])
            
            if is_equivalent and confidence >= 0.8:
                return VerificationCheck(
                    name="semantic_equivalence",
                    status=VerificationStatus.PASSED,
                    message="Code appears semantically equivalent",
                    details={"confidence": confidence, "assessment": result.get("assessment", "")},
                )
            elif is_equivalent and confidence >= 0.5:
                return VerificationCheck(
                    name="semantic_equivalence",
                    status=VerificationStatus.WARNING,
                    message="Code may be semantically equivalent (low confidence)",
                    details={"confidence": confidence, "differences": differences},
                )
            else:
                return VerificationCheck(
                    name="semantic_equivalence",
                    status=VerificationStatus.FAILED,
                    message="Code may not be semantically equivalent",
                    details={"confidence": confidence, "differences": differences},
                )
                
        except Exception as e:
            logger.warning(f"Semantic equivalence check failed: {e}")
            return VerificationCheck(
                name="semantic_equivalence",
                status=VerificationStatus.WARNING,
                message=f"Could not verify semantic equivalence: {e}",
            )
    
    def handle_message(self, message: AgentMessage) -> AgentMessage:
        """Handle a verification request message."""
        if message.message_type != MessageType.VERIFY_REQUEST:
            return message.create_error_response(
                AgentRole.VERIFIER,
                f"Unexpected message type: {message.message_type}",
            )
        
        payload = message.payload
        context = HandoffContext.from_dict(payload.get("context", {}))
        refactored_files = payload.get("refactored_files", {})
        
        context.add_to_chain(AgentRole.VERIFIER)
        
        all_results = {}
        all_passed = True
        requires_review = False
        
        for filename, refactored_code in refactored_files.items():
            original_code = context.original_code.get(filename, "")
            violations = [v for v in context.violations if v.get("file") == filename]
            
            result = self.verify(
                original_code=original_code,
                refactored_code=refactored_code,
                filename=filename,
                violations_to_fix=violations,
            )
            
            all_results[filename] = result.to_dict()
            
            if not result.passed:
                all_passed = False
            if result.requires_human_review:
                requires_review = True
        
        return message.create_response(
            message_type=MessageType.VERIFY_RESULT,
            sender=AgentRole.VERIFIER,
            payload={
                "context": context.to_dict(),
                "passed": all_passed,
                "requires_human_review": requires_review,
                "results": all_results,
            },
        )
