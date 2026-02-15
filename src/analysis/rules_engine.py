"""Rules engine for applying coding standards."""

import re
import logging
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import yaml

from src.github.models import (
    FileChange,
    ParsedDiff,
    Violation,
    Severity,
    ViolationCategory,
)

logger = logging.getLogger(__name__)


class Rule(ABC):
    """Base class for coding standard rules."""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        category: ViolationCategory,
        severity: Severity,
        config: dict,
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        self.config = config
    
    @abstractmethod
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        """
        Check the code against this rule.
        
        Args:
            filename: Name of the file.
            content: Full file content.
            parsed_diff: Optional parsed diff for the file.
            
        Returns:
            List of violations found.
        """
        pass
    
    def _create_violation(
        self,
        file: str,
        line: int,
        message: str,
        suggestion: str = "",
        code_snippet: str = "",
        confidence: float = 1.0,
        column: int = 0,
    ) -> Violation:
        """Helper to create a Violation object."""
        return Violation(
            rule_id=self.rule_id,
            rule_name=self.name,
            category=self.category,
            severity=self.severity,
            file=file,
            line=line,
            column=column,
            message=message,
            suggestion=suggestion,
            code_snippet=code_snippet,
            confidence=confidence,
        )


class LineLengthRule(Rule):
    """S001: Check for lines exceeding maximum length."""
    
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        violations = []
        max_length = self.config.get("max_length", 120)
        ignore_urls = self.config.get("ignore_urls", True)
        ignore_imports = self.config.get("ignore_imports", True)
        
        lines = content.split("\n")
        
        # Get changed line numbers if diff provided
        changed_lines = set()
        if parsed_diff:
            for hunk in parsed_diff.hunks:
                for line in hunk.lines:
                    if line.change_type == "added":
                        changed_lines.add(line.line_number)
        
        for i, line in enumerate(lines, start=1):
            # Skip if not in changed lines (when diff available)
            if changed_lines and i not in changed_lines:
                continue
            
            if len(line) <= max_length:
                continue
            
            # Skip URLs
            if ignore_urls and ("http://" in line or "https://" in line):
                continue
            
            # Skip imports
            if ignore_imports and (
                line.strip().startswith("import ") or 
                line.strip().startswith("from ")
            ):
                continue
            
            violations.append(self._create_violation(
                file=filename,
                line=i,
                message=f"Line exceeds {max_length} characters (length: {len(line)})",
                suggestion=f"Break this line into multiple lines or refactor for readability",
                code_snippet=line[:100] + "..." if len(line) > 100 else line,
            ))
        
        return violations


class NamingConventionRule(Rule):
    """S002: Check naming conventions."""
    
    # Patterns for detecting definitions
    FUNCTION_PATTERN = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
    CLASS_PATTERN = re.compile(r"^\s*class\s+(\w+)\s*[:\(]")
    CONSTANT_PATTERN = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=")
    VARIABLE_PATTERN = re.compile(r"^\s*(\w+)\s*=(?!=)")
    
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        violations = []
        patterns = self.config.get("patterns", {})
        
        lines = content.split("\n")
        
        # Get changed line numbers if diff provided
        changed_lines = set()
        if parsed_diff:
            for hunk in parsed_diff.hunks:
                for line in hunk.lines:
                    if line.change_type == "added":
                        changed_lines.add(line.line_number)
        
        for i, line in enumerate(lines, start=1):
            if changed_lines and i not in changed_lines:
                continue
            
            # Check functions
            func_match = self.FUNCTION_PATTERN.match(line)
            if func_match:
                name = func_match.group(1)
                expected = patterns.get("functions", "snake_case")
                if not self._matches_convention(name, expected):
                    if not name.startswith("_"):  # Skip private/magic methods
                        violations.append(self._create_violation(
                            file=filename,
                            line=i,
                            message=f"Function '{name}' should use {expected}",
                            suggestion=f"Rename to '{self._convert_to_convention(name, expected)}'",
                            code_snippet=line.strip(),
                        ))
            
            # Check classes
            class_match = self.CLASS_PATTERN.match(line)
            if class_match:
                name = class_match.group(1)
                expected = patterns.get("classes", "PascalCase")
                if not self._matches_convention(name, expected):
                    violations.append(self._create_violation(
                        file=filename,
                        line=i,
                        message=f"Class '{name}' should use {expected}",
                        suggestion=f"Rename to '{self._convert_to_convention(name, expected)}'",
                        code_snippet=line.strip(),
                    ))
        
        return violations
    
    def _matches_convention(self, name: str, convention: str) -> bool:
        """Check if name matches the convention."""
        if convention == "snake_case":
            return bool(re.match(r"^[a-z][a-z0-9_]*$", name))
        elif convention == "PascalCase":
            return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))
        elif convention == "UPPER_SNAKE_CASE":
            return bool(re.match(r"^[A-Z][A-Z0-9_]*$", name))
        elif convention == "camelCase":
            return bool(re.match(r"^[a-z][a-zA-Z0-9]*$", name))
        return True
    
    def _convert_to_convention(self, name: str, convention: str) -> str:
        """Convert name to the target convention."""
        # Split by common separators
        words = re.split(r"[_\s]+|(?<=[a-z])(?=[A-Z])", name)
        words = [w.lower() for w in words if w]
        
        if convention == "snake_case":
            return "_".join(words)
        elif convention == "PascalCase":
            return "".join(w.capitalize() for w in words)
        elif convention == "UPPER_SNAKE_CASE":
            return "_".join(w.upper() for w in words)
        elif convention == "camelCase":
            return words[0] + "".join(w.capitalize() for w in words[1:])
        return name


class FunctionComplexityRule(Rule):
    """Q001: Check cyclomatic complexity of functions."""
    
    # Patterns that increase complexity
    COMPLEXITY_PATTERNS = [
        r"\bif\b",
        r"\belif\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\band\b",
        r"\bor\b",
        r"\bexcept\b",
        r"\bcase\b",
        r"\?\s*.*\s*:",  # Ternary operator
    ]
    
    FUNCTION_START = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")
    
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        violations = []
        max_complexity = self.config.get("max_cyclomatic_complexity", 10)
        
        lines = content.split("\n")
        functions = self._find_functions(lines)
        
        for func_name, start_line, end_line, indent in functions:
            func_lines = lines[start_line - 1:end_line]
            complexity = self._calculate_complexity(func_lines)
            
            if complexity > max_complexity:
                violations.append(self._create_violation(
                    file=filename,
                    line=start_line,
                    message=f"Function '{func_name}' has cyclomatic complexity of {complexity} (max: {max_complexity})",
                    suggestion="Consider breaking this function into smaller, focused functions",
                    confidence=0.9,
                ))
        
        return violations
    
    def _find_functions(self, lines: list[str]) -> list[tuple]:
        """Find all functions with their line ranges."""
        functions = []
        
        for i, line in enumerate(lines):
            match = self.FUNCTION_START.match(line)
            if match:
                indent = len(match.group(1))
                func_name = match.group(2)
                start_line = i + 1
                
                # Find end of function
                end_line = start_line
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip() == "":
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= indent and next_line.strip():
                        break
                    end_line = j + 1
                
                functions.append((func_name, start_line, end_line, indent))
        
        return functions
    
    def _calculate_complexity(self, lines: list[str]) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for line in lines:
            for pattern in self.COMPLEXITY_PATTERNS:
                complexity += len(re.findall(pattern, line))
        
        return complexity


class FunctionLengthRule(Rule):
    """Q002: Check function length."""
    
    FUNCTION_START = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")
    
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        violations = []
        max_lines = self.config.get("max_lines", 50)
        exclude_comments = self.config.get("exclude_comments", True)
        exclude_blank = self.config.get("exclude_blank_lines", True)
        
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            match = self.FUNCTION_START.match(line)
            if match:
                indent = len(match.group(1))
                func_name = match.group(2)
                start_line = i + 1
                
                # Count function lines
                func_line_count = 0
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    
                    # Check if still in function
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent:
                            break
                    
                    # Count line based on config
                    stripped = next_line.strip()
                    if exclude_blank and not stripped:
                        continue
                    if exclude_comments and stripped.startswith("#"):
                        continue
                    
                    func_line_count += 1
                
                if func_line_count > max_lines:
                    violations.append(self._create_violation(
                        file=filename,
                        line=start_line,
                        message=f"Function '{func_name}' is {func_line_count} lines long (max: {max_lines})",
                        suggestion="Consider extracting parts of this function into smaller helper functions",
                    ))
        
        return violations


class HardcodedSecretsRule(Rule):
    """SEC001: Detect hardcoded secrets."""
    
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        violations = []
        patterns = self.config.get("patterns", [])
        exclude_patterns = self.config.get("exclude_patterns", [])
        
        lines = content.split("\n")
        
        # Get changed line numbers if diff provided
        changed_lines = set()
        if parsed_diff:
            for hunk in parsed_diff.hunks:
                for line in hunk.lines:
                    if line.change_type == "added":
                        changed_lines.add(line.line_number)
        
        for i, line in enumerate(lines, start=1):
            if changed_lines and i not in changed_lines:
                continue
            
            # Check exclusions first
            excluded = False
            for exc_pattern in exclude_patterns:
                if re.search(exc_pattern, line, re.IGNORECASE):
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Check for secrets
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(self._create_violation(
                        file=filename,
                        line=i,
                        message="Potential hardcoded secret or credential detected",
                        suggestion="Use environment variables or a secrets manager instead",
                        code_snippet=self._redact_line(line),
                        confidence=0.85,
                    ))
                    break  # One violation per line
        
        return violations
    
    def _redact_line(self, line: str) -> str:
        """Redact potential secret values in the line."""
        # Replace quoted strings after = with [REDACTED]
        return re.sub(r"(['\"])([^'\"]+)\1", r"\1[REDACTED]\1", line)


class SQLInjectionRule(Rule):
    """SEC002: Detect potential SQL injection vulnerabilities."""
    
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        violations = []
        patterns = self.config.get("patterns", [])
        
        lines = content.split("\n")
        
        # Get changed line numbers if diff provided
        changed_lines = set()
        if parsed_diff:
            for hunk in parsed_diff.hunks:
                for line in hunk.lines:
                    if line.change_type == "added":
                        changed_lines.add(line.line_number)
        
        for i, line in enumerate(lines, start=1):
            if changed_lines and i not in changed_lines:
                continue
            
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(self._create_violation(
                        file=filename,
                        line=i,
                        message="Potential SQL injection vulnerability - using string formatting in SQL query",
                        suggestion="Use parameterized queries instead: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                        code_snippet=line.strip(),
                    ))
                    break
        
        return violations


class ErrorHandlingRule(Rule):
    """BP001: Check for proper error handling."""
    
    BARE_EXCEPT = re.compile(r"^\s*except\s*:")
    BROAD_EXCEPT = re.compile(r"^\s*except\s+Exception\s*:")
    
    def check(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        violations = []
        checks = self.config.get("checks", {})
        
        lines = content.split("\n")
        
        # Get changed line numbers if diff provided
        changed_lines = set()
        if parsed_diff:
            for hunk in parsed_diff.hunks:
                for line in hunk.lines:
                    if line.change_type == "added":
                        changed_lines.add(line.line_number)
        
        for i, line in enumerate(lines, start=1):
            if changed_lines and i not in changed_lines:
                continue
            
            # Check bare except
            if checks.get("bare_except", True) and self.BARE_EXCEPT.match(line):
                violations.append(self._create_violation(
                    file=filename,
                    line=i,
                    message="Bare 'except:' clause catches all exceptions including SystemExit and KeyboardInterrupt",
                    suggestion="Catch specific exceptions: 'except (ValueError, TypeError) as e:'",
                    code_snippet=line.strip(),
                ))
            
            # Check broad except
            elif checks.get("broad_except", True) and self.BROAD_EXCEPT.match(line):
                # Check if next line is just 'pass' or empty
                if i < len(lines):
                    next_line = lines[i].strip()
                    if next_line == "pass" and checks.get("empty_except", True):
                        violations.append(self._create_violation(
                            file=filename,
                            line=i,
                            message="Empty exception handler - exceptions are silently ignored",
                            suggestion="Log the exception or handle it appropriately",
                            code_snippet=line.strip(),
                        ))
        
        return violations


class RulesEngine:
    """Engine for loading and applying coding standard rules."""
    
    RULE_CLASSES = {
        "line_length": LineLengthRule,
        "naming_conventions": NamingConventionRule,
        "function_complexity": FunctionComplexityRule,
        "function_length": FunctionLengthRule,
        "hardcoded_secrets": HardcodedSecretsRule,
        "sql_injection": SQLInjectionRule,
        "error_handling": ErrorHandlingRule,
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the rules engine.
        
        Args:
            config_path: Path to coding_standards.yaml. Defaults to config/coding_standards.yaml.
        """
        self.rules: list[Rule] = []
        self.delegation_config: dict = {}
        
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "coding_standards.yaml"
        
        self._load_config(config_path)
    
    def _load_config(self, config_path: str | Path) -> None:
        """Load rules from YAML configuration."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            self._load_defaults()
            return
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Load delegation config
        self.delegation_config = config.get("delegation", {})
        
        # Load rules
        for rule_config in config.get("rules", []):
            if not rule_config.get("enabled", True):
                continue
            
            rule_name = rule_config["name"]
            rule_class = self.RULE_CLASSES.get(rule_name)
            
            if rule_class is None:
                logger.warning(f"Unknown rule: {rule_name}")
                continue
            
            category = ViolationCategory(rule_config["category"])
            severity = Severity(rule_config["severity"])
            
            rule = rule_class(
                rule_id=rule_config["id"],
                name=rule_name,
                description=rule_config.get("description", ""),
                category=category,
                severity=severity,
                config=rule_config.get("config", {}),
            )
            
            self.rules.append(rule)
            logger.debug(f"Loaded rule: {rule.rule_id} - {rule.name}")
        
        logger.info(f"Loaded {len(self.rules)} rules from {config_path}")
    
    def _load_defaults(self) -> None:
        """Load default rules when no config is available."""
        defaults = [
            (LineLengthRule, "S001", "line_length", ViolationCategory.STYLE, Severity.WARNING, {"max_length": 120}),
            (NamingConventionRule, "S002", "naming_conventions", ViolationCategory.STYLE, Severity.WARNING, 
             {"patterns": {"functions": "snake_case", "classes": "PascalCase"}}),
            (FunctionComplexityRule, "Q001", "function_complexity", ViolationCategory.QUALITY, Severity.ERROR,
             {"max_cyclomatic_complexity": 10}),
            (FunctionLengthRule, "Q002", "function_length", ViolationCategory.QUALITY, Severity.WARNING,
             {"max_lines": 50}),
            (HardcodedSecretsRule, "SEC001", "hardcoded_secrets", ViolationCategory.SECURITY, Severity.CRITICAL,
             {"patterns": [r"password\s*=", r"api_key\s*=", r"secret\s*="]}),
            (SQLInjectionRule, "SEC002", "sql_injection", ViolationCategory.SECURITY, Severity.CRITICAL,
             {"patterns": [r"f['\"]SELECT", r"\.format\(.*\).*SELECT"]}),
            (ErrorHandlingRule, "BP001", "error_handling", ViolationCategory.BEST_PRACTICES, Severity.WARNING,
             {"checks": {"bare_except": True, "broad_except": True}}),
        ]
        
        for rule_class, rule_id, name, category, severity, config in defaults:
            rule = rule_class(
                rule_id=rule_id,
                name=name,
                description="",
                category=category,
                severity=severity,
                config=config,
            )
            self.rules.append(rule)
    
    def analyze(
        self,
        filename: str,
        content: str,
        parsed_diff: Optional[ParsedDiff] = None,
    ) -> list[Violation]:
        """
        Analyze code against all rules.
        
        Args:
            filename: Name of the file.
            content: Full file content.
            parsed_diff: Optional parsed diff.
            
        Returns:
            List of violations found.
        """
        all_violations = []
        
        for rule in self.rules:
            try:
                violations = rule.check(filename, content, parsed_diff)
                all_violations.extend(violations)
                logger.debug(f"Rule {rule.rule_id}: found {len(violations)} violations")
            except Exception as e:
                logger.error(f"Error in rule {rule.rule_id}: {e}")
        
        return all_violations
    
    def should_delegate(self, violations: list[Violation]) -> bool:
        """
        Check if violations warrant delegation to refactoring agent.
        
        Args:
            violations: List of violations found.
            
        Returns:
            True if delegation should occur.
        """
        if not self.delegation_config.get("enabled", True):
            return False
        
        criteria = self.delegation_config.get("criteria", {})
        
        # Check for critical violations
        if criteria.get("critical_violation_auto_delegate", True):
            if any(v.severity == Severity.CRITICAL for v in violations):
                return True
        
        # Check violations per file
        violations_threshold = criteria.get("violations_per_file_threshold", 3)
        files_violations = {}
        for v in violations:
            files_violations[v.file] = files_violations.get(v.file, 0) + 1
        
        if any(count >= violations_threshold for count in files_violations.values()):
            return True
        
        # Check complexity violations
        complexity_threshold = criteria.get("cyclomatic_complexity_threshold", 10)
        for v in violations:
            if v.rule_id == "Q001":  # Function complexity rule
                return True
        
        return False
