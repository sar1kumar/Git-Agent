"""Tests for the review agent and related components."""

import pytest
from unittest.mock import Mock, MagicMock

from src.github.models import (
    FileChange,
    Violation,
    Severity,
    ViolationCategory,
    ReviewComment,
    ReviewSummary,
    ParsedDiff,
    DiffHunk,
    HunkLine,
)
from src.analysis.diff_parser import DiffParser
from src.analysis.rules_engine import RulesEngine, LineLengthRule, HardcodedSecretsRule


class TestDiffParser:
    """Tests for the DiffParser."""
    
    def test_parse_simple_patch(self):
        """Test parsing a simple diff patch."""
        parser = DiffParser()
        
        patch = """@@ -1,3 +1,4 @@
 import os
+import sys
 
 def main():"""
        
        result = parser.parse_patch("test.py", patch)
        
        assert result.filename == "test.py"
        assert len(result.hunks) == 1
        assert result.hunks[0].new_start == 1
        assert result.hunks[0].new_count == 4
    
    def test_get_added_lines(self):
        """Test extracting added lines from a patch."""
        parser = DiffParser()
        
        patch = """@@ -1,2 +1,4 @@
 line1
+added1
+added2
 line2"""
        
        result = parser.parse_patch("test.py", patch)
        added = result.get_added_lines()
        
        assert len(added) == 2
        assert added[0].content == "added1"
        assert added[1].content == "added2"
    
    def test_get_changed_line_numbers(self):
        """Test getting line numbers of changes."""
        parser = DiffParser()
        
        patch = """@@ -1,2 +1,4 @@
 line1
+added1
+added2
 line2"""
        
        result = parser.parse_patch("test.py", patch)
        line_numbers = parser.get_changed_line_numbers(result)
        
        assert 2 in line_numbers
        assert 3 in line_numbers


class TestLineLengthRule:
    """Tests for the line length rule."""
    
    def test_detects_long_lines(self):
        """Test detection of lines exceeding max length."""
        rule = LineLengthRule(
            rule_id="S001",
            name="line_length",
            description="Test rule",
            category=ViolationCategory.STYLE,
            severity=Severity.WARNING,
            config={"max_length": 80},
        )
        
        code = "x = 1\n" + "a" * 100 + "\ny = 2"
        
        violations = rule.check("test.py", code)
        
        assert len(violations) == 1
        assert violations[0].line == 2
        assert violations[0].severity == Severity.WARNING
    
    def test_ignores_urls(self):
        """Test that URLs are ignored."""
        rule = LineLengthRule(
            rule_id="S001",
            name="line_length",
            description="Test rule",
            category=ViolationCategory.STYLE,
            severity=Severity.WARNING,
            config={"max_length": 80, "ignore_urls": True},
        )
        
        code = "# See: https://example.com/very/long/url/that/exceeds/the/max/length/limit"
        
        violations = rule.check("test.py", code)
        
        assert len(violations) == 0


class TestHardcodedSecretsRule:
    """Tests for the hardcoded secrets rule."""
    
    def test_detects_hardcoded_password(self):
        """Test detection of hardcoded passwords."""
        rule = HardcodedSecretsRule(
            rule_id="SEC001",
            name="hardcoded_secrets",
            description="Test rule",
            category=ViolationCategory.SECURITY,
            severity=Severity.CRITICAL,
            config={
                "patterns": [r"password\s*=\s*['\"][^'\"]+['\"]"],
                "exclude_patterns": [],
            },
        )
        
        code = 'password = "secret123"'
        
        violations = rule.check("test.py", code)
        
        assert len(violations) == 1
        assert violations[0].severity == Severity.CRITICAL
    
    def test_ignores_env_var_lookup(self):
        """Test that environment variable lookups are ignored."""
        rule = HardcodedSecretsRule(
            rule_id="SEC001",
            name="hardcoded_secrets",
            description="Test rule",
            category=ViolationCategory.SECURITY,
            severity=Severity.CRITICAL,
            config={
                "patterns": [r"password\s*="],
                "exclude_patterns": [r"os\.environ", r"getenv"],
            },
        )
        
        code = 'password = os.environ.get("PASSWORD")'
        
        violations = rule.check("test.py", code)
        
        assert len(violations) == 0


class TestRulesEngine:
    """Tests for the rules engine."""
    
    def test_loads_default_rules(self):
        """Test that default rules are loaded when no config provided."""
        engine = RulesEngine(config_path="/nonexistent/path.yaml")
        
        assert len(engine.rules) > 0
        rule_ids = [r.rule_id for r in engine.rules]
        assert "S001" in rule_ids
        assert "SEC001" in rule_ids
    
    def test_should_delegate_on_critical(self):
        """Test delegation on critical violations."""
        engine = RulesEngine(config_path="/nonexistent/path.yaml")
        engine.delegation_config = {
            "enabled": True,
            "criteria": {
                "critical_violation_auto_delegate": True,
            },
        }
        
        violations = [
            Violation(
                rule_id="SEC001",
                rule_name="test",
                category=ViolationCategory.SECURITY,
                severity=Severity.CRITICAL,
                file="test.py",
                line=1,
                message="Test violation",
            )
        ]
        
        assert engine.should_delegate(violations) is True
    
    def test_should_not_delegate_on_warnings(self):
        """Test no delegation on warnings only."""
        engine = RulesEngine(config_path="/nonexistent/path.yaml")
        engine.delegation_config = {
            "enabled": True,
            "criteria": {
                "critical_violation_auto_delegate": True,
                "violations_per_file_threshold": 5,
            },
        }
        
        violations = [
            Violation(
                rule_id="S001",
                rule_name="test",
                category=ViolationCategory.STYLE,
                severity=Severity.WARNING,
                file="test.py",
                line=1,
                message="Test violation",
            )
        ]
        
        assert engine.should_delegate(violations) is False


class TestReviewSummary:
    """Tests for the review summary."""
    
    def test_to_markdown(self):
        """Test markdown generation."""
        summary = ReviewSummary(
            total_files=5,
            total_violations=3,
            critical_count=1,
            error_count=1,
            warning_count=1,
            info_count=0,
            files_with_issues=["test.py", "main.py"],
        )
        
        markdown = summary.to_markdown()
        
        assert "AI Code Review Summary" in markdown
        assert "5" in markdown  # total files
        assert "3" in markdown  # total violations
        assert "test.py" in markdown
        assert "main.py" in markdown


class TestReviewComment:
    """Tests for review comments."""
    
    def test_format_body_with_severity(self):
        """Test comment formatting with severity badge."""
        comment = ReviewComment(
            file="test.py",
            line=10,
            body="This is a test issue",
            severity=Severity.ERROR,
            rule_id="Q001",
        )
        
        formatted = comment.format_body()
        
        assert "ERROR" in formatted
        assert "Q001" in formatted
        assert "This is a test issue" in formatted
    
    def test_format_body_with_suggestion(self):
        """Test comment formatting with suggestion code."""
        comment = ReviewComment(
            file="test.py",
            line=10,
            body="Use snake_case",
            severity=Severity.WARNING,
            suggestion_code="def my_function():",
        )
        
        formatted = comment.format_body()
        
        assert "```suggestion" in formatted
        assert "def my_function():" in formatted
