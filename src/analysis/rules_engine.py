"""Rules engine for applying coding standards."""

import re
import logging
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import yaml

from src.github.models import ParsedDiff, Violation, Severity, ViolationCategory

logger = logging.getLogger(__name__)


class Rule(ABC):
    """Base class for coding standard rules."""

    def __init__(self, rule_id: str, name: str, description: str,
                 category: ViolationCategory, severity: Severity, config: dict):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity
        self.config = config

    @abstractmethod
    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        pass

    def _create_violation(self, file: str, line: int, message: str,
                          suggestion: str = "", code_snippet: str = "",
                          confidence: float = 1.0) -> Violation:
        return Violation(
            rule_id=self.rule_id, rule_name=self.name, category=self.category,
            severity=self.severity, file=file, line=line, message=message,
            suggestion=suggestion, code_snippet=code_snippet, confidence=confidence,
        )

    @staticmethod
    def _get_changed_lines(parsed_diff: Optional[ParsedDiff]) -> set[int]:
        """Extract changed line numbers from a diff. Shared by all line-level rules."""
        if not parsed_diff:
            return set()
        return {
            line.line_number
            for hunk in parsed_diff.hunks
            for line in hunk.lines
            if line.change_type == "added"
        }


class LineLengthRule(Rule):
    """S001: Check for lines exceeding maximum length."""

    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        violations = []
        max_length = self.config.get("max_length", 120)
        ignore_urls = self.config.get("ignore_urls", True)
        ignore_imports = self.config.get("ignore_imports", True)
        changed_lines = self._get_changed_lines(parsed_diff)

        for i, line in enumerate(content.split("\n"), start=1):
            if changed_lines and i not in changed_lines:
                continue
            if len(line) <= max_length:
                continue
            if ignore_urls and ("http://" in line or "https://" in line):
                continue
            if ignore_imports and (line.strip().startswith("import ") or line.strip().startswith("from ")):
                continue

            violations.append(self._create_violation(
                file=filename, line=i,
                message=f"Line exceeds {max_length} characters (length: {len(line)})",
                suggestion="Break this line into multiple lines or refactor for readability",
                code_snippet=line[:100] + "..." if len(line) > 100 else line,
            ))
        return violations


class NamingConventionRule(Rule):
    """S002: Check naming conventions."""

    FUNCTION_PATTERN = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
    CLASS_PATTERN = re.compile(r"^\s*class\s+(\w+)\s*[:\(]")

    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        violations = []
        patterns = self.config.get("patterns", {})
        changed_lines = self._get_changed_lines(parsed_diff)

        for i, line in enumerate(content.split("\n"), start=1):
            if changed_lines and i not in changed_lines:
                continue

            func_match = self.FUNCTION_PATTERN.match(line)
            if func_match:
                name = func_match.group(1)
                expected = patterns.get("functions", "snake_case")
                if not name.startswith("_") and not self._matches(name, expected):
                    violations.append(self._create_violation(
                        file=filename, line=i,
                        message=f"Function '{name}' should use {expected}",
                        suggestion=f"Rename to '{self._convert(name, expected)}'",
                        code_snippet=line.strip(),
                    ))

            class_match = self.CLASS_PATTERN.match(line)
            if class_match:
                name = class_match.group(1)
                expected = patterns.get("classes", "PascalCase")
                if not self._matches(name, expected):
                    violations.append(self._create_violation(
                        file=filename, line=i,
                        message=f"Class '{name}' should use {expected}",
                        suggestion=f"Rename to '{self._convert(name, expected)}'",
                        code_snippet=line.strip(),
                    ))
        return violations

    @staticmethod
    def _matches(name: str, convention: str) -> bool:
        rules = {
            "snake_case": r"^[a-z][a-z0-9_]*$",
            "PascalCase": r"^[A-Z][a-zA-Z0-9]*$",
            "UPPER_SNAKE_CASE": r"^[A-Z][A-Z0-9_]*$",
            "camelCase": r"^[a-z][a-zA-Z0-9]*$",
        }
        return bool(re.match(rules.get(convention, r".*"), name))

    @staticmethod
    def _convert(name: str, convention: str) -> str:
        words = [w.lower() for w in re.split(r"[_\s]+|(?<=[a-z])(?=[A-Z])", name) if w]
        if convention == "snake_case":
            return "_".join(words)
        elif convention == "PascalCase":
            return "".join(w.capitalize() for w in words)
        elif convention == "UPPER_SNAKE_CASE":
            return "_".join(w.upper() for w in words)
        return name


class FunctionComplexityRule(Rule):
    """Q001: Check cyclomatic complexity of functions."""

    COMPLEXITY_PATTERNS = [
        r"\bif\b", r"\belif\b", r"\bfor\b", r"\bwhile\b",
        r"\band\b", r"\bor\b", r"\bexcept\b",
    ]
    FUNCTION_START = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")

    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        violations = []
        max_complexity = self.config.get("max_cyclomatic_complexity", 10)
        lines = content.split("\n")

        for func_name, start, end, _ in self._find_functions(lines):
            complexity = self._calculate_complexity(lines[start - 1:end])
            if complexity > max_complexity:
                violations.append(self._create_violation(
                    file=filename, line=start,
                    message=f"Function '{func_name}' has cyclomatic complexity of {complexity} (max: {max_complexity})",
                    suggestion="Consider breaking this function into smaller, focused functions",
                    confidence=0.9,
                ))
        return violations

    def _find_functions(self, lines: list[str]) -> list[tuple]:
        functions = []
        for i, line in enumerate(lines):
            match = self.FUNCTION_START.match(line)
            if match:
                indent = len(match.group(1))
                start_line = i + 1
                end_line = start_line
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == "":
                        continue
                    if len(lines[j]) - len(lines[j].lstrip()) <= indent and lines[j].strip():
                        break
                    end_line = j + 1
                functions.append((match.group(2), start_line, end_line, indent))
        return functions

    def _calculate_complexity(self, lines: list[str]) -> int:
        complexity = 1
        for line in lines:
            for pattern in self.COMPLEXITY_PATTERNS:
                complexity += len(re.findall(pattern, line))
        return complexity


class FunctionLengthRule(Rule):
    """Q002: Check function length."""

    FUNCTION_START = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")

    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        violations = []
        max_lines = self.config.get("max_lines", 50)
        exclude_comments = self.config.get("exclude_comments", True)
        exclude_blank = self.config.get("exclude_blank_lines", True)
        lines = content.split("\n")

        for i, line in enumerate(lines):
            match = self.FUNCTION_START.match(line)
            if not match:
                continue

            indent = len(match.group(1))
            func_name = match.group(2)
            func_line_count = 0

            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= indent:
                    break
                stripped = next_line.strip()
                if exclude_blank and not stripped:
                    continue
                if exclude_comments and stripped.startswith("#"):
                    continue
                func_line_count += 1

            if func_line_count > max_lines:
                violations.append(self._create_violation(
                    file=filename, line=i + 1,
                    message=f"Function '{func_name}' is {func_line_count} lines long (max: {max_lines})",
                    suggestion="Consider extracting parts into smaller helper functions",
                ))
        return violations


class HardcodedSecretsRule(Rule):
    """SEC001: Detect hardcoded secrets."""

    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        violations = []
        patterns = self.config.get("patterns", [])
        exclude_patterns = self.config.get("exclude_patterns", [])
        changed_lines = self._get_changed_lines(parsed_diff)

        for i, line in enumerate(content.split("\n"), start=1):
            if changed_lines and i not in changed_lines:
                continue
            if any(re.search(ep, line, re.IGNORECASE) for ep in exclude_patterns):
                continue
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    redacted = re.sub(r"(['\"])([^'\"]+)\1", r"\1[REDACTED]\1", line)
                    violations.append(self._create_violation(
                        file=filename, line=i,
                        message="Potential hardcoded secret or credential detected",
                        suggestion="Use environment variables or a secrets manager instead",
                        code_snippet=redacted, confidence=0.85,
                    ))
                    break
        return violations


class SQLInjectionRule(Rule):
    """SEC002: Detect potential SQL injection vulnerabilities."""

    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        violations = []
        patterns = self.config.get("patterns", [])
        changed_lines = self._get_changed_lines(parsed_diff)

        for i, line in enumerate(content.split("\n"), start=1):
            if changed_lines and i not in changed_lines:
                continue
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(self._create_violation(
                        file=filename, line=i,
                        message="Potential SQL injection vulnerability - using string formatting in SQL query",
                        suggestion="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                        code_snippet=line.strip(),
                    ))
                    break
        return violations


class ErrorHandlingRule(Rule):
    """BP001: Check for proper error handling."""

    BARE_EXCEPT = re.compile(r"^\s*except\s*:")
    BROAD_EXCEPT = re.compile(r"^\s*except\s+Exception\s*:")

    def check(self, filename: str, content: str,
              parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        violations = []
        checks = self.config.get("checks", {})
        changed_lines = self._get_changed_lines(parsed_diff)
        lines = content.split("\n")

        for i, line in enumerate(lines, start=1):
            if changed_lines and i not in changed_lines:
                continue

            if checks.get("bare_except", True) and self.BARE_EXCEPT.match(line):
                violations.append(self._create_violation(
                    file=filename, line=i,
                    message="Bare 'except:' clause catches all exceptions including SystemExit and KeyboardInterrupt",
                    suggestion="Catch specific exceptions: 'except (ValueError, TypeError) as e:'",
                    code_snippet=line.strip(),
                ))
            elif checks.get("broad_except", True) and self.BROAD_EXCEPT.match(line):
                if i < len(lines) and lines[i].strip() == "pass" and checks.get("empty_except", True):
                    violations.append(self._create_violation(
                        file=filename, line=i,
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
        self.rules: list[Rule] = []
        self.delegation_config: dict = {}

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "coding_standards.yaml"
        self._load_config(config_path)

    def _load_config(self, config_path: str | Path) -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            self._load_defaults()
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.delegation_config = config.get("delegation", {})

        for rule_config in config.get("rules", []):
            if not rule_config.get("enabled", True):
                continue
            rule_class = self.RULE_CLASSES.get(rule_config["name"])
            if rule_class is None:
                logger.warning(f"Unknown rule: {rule_config['name']}")
                continue

            rule = rule_class(
                rule_id=rule_config["id"], name=rule_config["name"],
                description=rule_config.get("description", ""),
                category=ViolationCategory(rule_config["category"]),
                severity=Severity(rule_config["severity"]),
                config=rule_config.get("config", {}),
            )
            self.rules.append(rule)

        logger.info(f"Loaded {len(self.rules)} rules from {config_path}")

    def _load_defaults(self) -> None:
        defaults = [
            (LineLengthRule, "S001", "line_length", ViolationCategory.STYLE, Severity.WARNING,
             {"max_length": 120}),
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
        for cls, rid, name, cat, sev, cfg in defaults:
            self.rules.append(cls(rule_id=rid, name=name, description="",
                                  category=cat, severity=sev, config=cfg))

    def analyze(self, filename: str, content: str,
                parsed_diff: Optional[ParsedDiff] = None) -> list[Violation]:
        """Analyze code against all enabled rules."""
        all_violations = []
        for rule in self.rules:
            try:
                all_violations.extend(rule.check(filename, content, parsed_diff))
            except Exception as e:
                logger.error(f"Error in rule {rule.rule_id}: {e}")
        return all_violations

    def should_delegate(self, violations: list[Violation]) -> bool:
        """Check if violations warrant delegation to refactoring agent."""
        if not self.delegation_config.get("enabled", True):
            return False

        criteria = self.delegation_config.get("criteria", {})

        # Auto-delegate on critical violations
        if criteria.get("critical_violation_auto_delegate", True):
            if any(v.severity == Severity.CRITICAL for v in violations):
                return True

        # Delegate if any file has too many violations
        threshold = criteria.get("violations_per_file_threshold", 3)
        file_counts: dict[str, int] = {}
        for v in violations:
            file_counts[v.file] = file_counts.get(v.file, 0) + 1
        if any(count >= threshold for count in file_counts.values()):
            return True

        # Delegate on complexity violations
        if any(v.rule_id == "Q001" for v in violations):
            return True

        return False
