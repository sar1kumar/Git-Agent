"""Test runner agent for validating refactored code."""

import os
import subprocess
import logging
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum

from src.agents.protocol import (
    AgentMessage,
    AgentRole,
    MessageType,
    HandoffContext,
)

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Status of test execution."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    NO_TESTS = "no_tests"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    stdout: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Result of running a test suite."""
    status: TestStatus
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_ms: float = 0.0
    tests: list[TestResult] = field(default_factory=list)
    coverage_percent: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 1.0
        return self.passed / self.total_tests
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_ms": self.duration_ms,
            "success_rate": self.success_rate,
            "coverage_percent": self.coverage_percent,
            "tests": [
                {
                    "name": t.name,
                    "status": t.status.value,
                    "duration_ms": t.duration_ms,
                    "error_message": t.error_message,
                }
                for t in self.tests
            ],
        }


class TestRunner:
    """
    Agent that runs tests to validate refactored code.
    
    Supports:
    - pytest for Python
    - Detecting test files related to changed files
    - Running with coverage
    - Sandboxed execution
    """
    
    # Test framework detection patterns
    PYTEST_MARKERS = ["pytest", "conftest.py", "pytest.ini", "pyproject.toml"]
    UNITTEST_MARKERS = ["unittest"]
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 300,
        use_coverage: bool = True,
    ):
        """
        Initialize the test runner.
        
        Args:
            working_dir: Working directory for test execution.
            timeout_seconds: Maximum time for test execution.
            use_coverage: Whether to collect coverage data.
        """
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout_seconds
        self.use_coverage = use_coverage
        logger.info(f"Initialized TestRunner in {self.working_dir}")
    
    def run_tests(
        self,
        test_files: Optional[list[str]] = None,
        changed_files: Optional[list[str]] = None,
    ) -> TestSuiteResult:
        """
        Run tests and return results.
        
        Args:
            test_files: Specific test files to run.
            changed_files: Changed source files to find related tests for.
            
        Returns:
            TestSuiteResult with test execution results.
        """
        # Determine which tests to run
        if test_files is None and changed_files:
            test_files = self._find_related_tests(changed_files)
        
        if not test_files:
            # Run all tests if no specific files
            test_files = self._discover_tests()
        
        if not test_files:
            return TestSuiteResult(
                status=TestStatus.NO_TESTS,
                stdout="No test files found",
            )
        
        # Detect test framework
        framework = self._detect_framework()
        
        if framework == "pytest":
            return self._run_pytest(test_files)
        else:
            return self._run_unittest(test_files)
    
    def run_tests_in_sandbox(
        self,
        code_changes: dict[str, str],
        test_files: Optional[list[str]] = None,
    ) -> TestSuiteResult:
        """
        Run tests with code changes in an isolated sandbox.
        
        Args:
            code_changes: Dict of filename -> new code content.
            test_files: Specific test files to run.
            
        Returns:
            TestSuiteResult with test execution results.
        """
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the project
            shutil.copytree(
                self.working_dir,
                os.path.join(tmpdir, "project"),
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", ".venv", "venv", "node_modules"
                ),
            )
            
            sandbox_dir = os.path.join(tmpdir, "project")
            
            # Apply code changes
            for filename, code in code_changes.items():
                filepath = os.path.join(sandbox_dir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write(code)
            
            # Run tests in sandbox
            original_dir = self.working_dir
            self.working_dir = sandbox_dir
            
            try:
                result = self.run_tests(test_files, list(code_changes.keys()))
            finally:
                self.working_dir = original_dir
            
            return result
    
    def _find_related_tests(self, source_files: list[str]) -> list[str]:
        """Find test files related to changed source files."""
        related_tests = []
        
        for source_file in source_files:
            # Get base name without extension
            base_name = Path(source_file).stem
            dir_name = Path(source_file).parent
            
            # Common test file patterns
            test_patterns = [
                f"test_{base_name}.py",
                f"{base_name}_test.py",
                f"tests/test_{base_name}.py",
                f"tests/{dir_name}/test_{base_name}.py",
            ]
            
            for pattern in test_patterns:
                test_path = os.path.join(self.working_dir, pattern)
                if os.path.exists(test_path):
                    related_tests.append(pattern)
        
        # Also check for tests/ directory
        tests_dir = os.path.join(self.working_dir, "tests")
        if os.path.isdir(tests_dir):
            for test_file in Path(tests_dir).rglob("test_*.py"):
                rel_path = str(test_file.relative_to(self.working_dir))
                if rel_path not in related_tests:
                    related_tests.append(rel_path)
        
        return list(set(related_tests))
    
    def _discover_tests(self) -> list[str]:
        """Discover all test files in the project."""
        test_files = []
        
        for pattern in ["test_*.py", "*_test.py"]:
            for test_file in Path(self.working_dir).rglob(pattern):
                # Skip venv and common excluded directories
                path_str = str(test_file)
                if any(x in path_str for x in [".venv", "venv", "node_modules", ".git"]):
                    continue
                test_files.append(str(test_file.relative_to(self.working_dir)))
        
        return test_files
    
    def _detect_framework(self) -> str:
        """Detect which test framework is being used."""
        # Check for pytest markers
        for marker in self.PYTEST_MARKERS:
            if os.path.exists(os.path.join(self.working_dir, marker)):
                return "pytest"
        
        # Check requirements.txt
        req_file = os.path.join(self.working_dir, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file) as f:
                content = f.read().lower()
                if "pytest" in content:
                    return "pytest"
        
        # Default to pytest
        return "pytest"
    
    def _run_pytest(self, test_files: list[str]) -> TestSuiteResult:
        """Run tests using pytest."""
        cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        
        if self.use_coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        cmd.extend(test_files)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            return self._parse_pytest_output(result)
            
        except subprocess.TimeoutExpired:
            return TestSuiteResult(
                status=TestStatus.ERROR,
                stderr=f"Test execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return TestSuiteResult(
                status=TestStatus.ERROR,
                stderr=str(e),
            )
    
    def _parse_pytest_output(self, result: subprocess.CompletedProcess) -> TestSuiteResult:
        """Parse pytest output into TestSuiteResult."""
        suite = TestSuiteResult(
            status=TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        
        # Parse summary line like "5 passed, 2 failed, 1 error in 0.5s"
        import re
        
        summary_match = re.search(
            r"(\d+) passed.*?(\d+) failed.*?(\d+) error",
            result.stdout,
            re.IGNORECASE,
        )
        
        if summary_match:
            suite.passed = int(summary_match.group(1))
            suite.failed = int(summary_match.group(2))
            suite.errors = int(summary_match.group(3))
            suite.total_tests = suite.passed + suite.failed + suite.errors
        else:
            # Try simpler patterns
            passed_match = re.search(r"(\d+) passed", result.stdout)
            failed_match = re.search(r"(\d+) failed", result.stdout)
            
            if passed_match:
                suite.passed = int(passed_match.group(1))
            if failed_match:
                suite.failed = int(failed_match.group(1))
            
            suite.total_tests = suite.passed + suite.failed
        
        # Parse duration
        duration_match = re.search(r"in ([\d.]+)s", result.stdout)
        if duration_match:
            suite.duration_ms = float(duration_match.group(1)) * 1000
        
        # Parse coverage if available
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
        if coverage_match:
            suite.coverage_percent = float(coverage_match.group(1))
        
        # Parse individual test results
        test_pattern = re.compile(r"([\w/\.]+::[\w]+)\s+(PASSED|FAILED|ERROR|SKIPPED)")
        for match in test_pattern.finditer(result.stdout):
            test_name = match.group(1)
            status_str = match.group(2)
            
            status_map = {
                "PASSED": TestStatus.PASSED,
                "FAILED": TestStatus.FAILED,
                "ERROR": TestStatus.ERROR,
                "SKIPPED": TestStatus.SKIPPED,
            }
            
            suite.tests.append(TestResult(
                name=test_name,
                status=status_map.get(status_str, TestStatus.ERROR),
            ))
        
        return suite
    
    def _run_unittest(self, test_files: list[str]) -> TestSuiteResult:
        """Run tests using unittest."""
        cmd = ["python", "-m", "unittest", "-v"] + test_files
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            suite = TestSuiteResult(
                status=TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            
            # Parse unittest output
            import re
            
            # Count results from stderr (unittest outputs there)
            ok_match = re.search(r"OK \(tests=(\d+)\)", result.stderr)
            fail_match = re.search(r"FAILED \(.*?failures=(\d+)", result.stderr)
            
            if ok_match:
                suite.passed = int(ok_match.group(1))
                suite.total_tests = suite.passed
            
            if fail_match:
                suite.failed = int(fail_match.group(1))
                suite.total_tests = suite.passed + suite.failed
            
            return suite
            
        except subprocess.TimeoutExpired:
            return TestSuiteResult(
                status=TestStatus.ERROR,
                stderr=f"Test execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return TestSuiteResult(
                status=TestStatus.ERROR,
                stderr=str(e),
            )
    
    def handle_message(self, message: AgentMessage) -> AgentMessage:
        """Handle a test request message."""
        if message.message_type != MessageType.TEST_REQUEST:
            return message.create_error_response(
                AgentRole.TEST_RUNNER,
                f"Unexpected message type: {message.message_type}",
            )
        
        payload = message.payload
        context = HandoffContext.from_dict(payload.get("context", {}))
        test_files = payload.get("test_files")
        
        context.add_to_chain(AgentRole.TEST_RUNNER)
        
        # Run tests with refactored code in sandbox
        if context.refactored_code:
            result = self.run_tests_in_sandbox(
                context.refactored_code,
                test_files,
            )
        else:
            result = self.run_tests(
                test_files,
                context.files,
            )
        
        return message.create_response(
            message_type=MessageType.TEST_RESULT,
            sender=AgentRole.TEST_RUNNER,
            payload={
                "context": context.to_dict(),
                "result": result.to_dict(),
                "passed": result.status == TestStatus.PASSED,
            },
        )
