"""Main entry point for the AI Code Review Agent."""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the code review agent."""
    parser = argparse.ArgumentParser(
        description="AI-powered GitHub PR Code Review Agent"
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Pull request number to review",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository in owner/repo format",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based analysis (static rules only)",
    )
    parser.add_argument(
        "--no-post",
        action="store_true",
        help="Don't post comments to GitHub (dry run)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSONL format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to coding standards config file",
    )
    
    args = parser.parse_args()
    
    # Get PR number from args or environment
    pr_number = args.pr or int(os.environ.get("PR_NUMBER", 0))
    if not pr_number:
        logger.error("PR number is required. Set --pr or PR_NUMBER env var.")
        sys.exit(1)
    
    # Get repository from args or environment
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        logger.error("Repository is required. Set --repo or GITHUB_REPOSITORY env var.")
        sys.exit(1)
    
    # Check for GitHub token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GitHub token is required. Set GITHUB_TOKEN env var.")
        sys.exit(1)
    
    logger.info(f"Starting review of PR #{pr_number} in {repo}")
    
    # Import here to avoid issues with missing dependencies
    from src.github.client import GitHubClient
    from src.analysis.rules_engine import RulesEngine
    from src.llm.ollama_client import OllamaClient
    from src.agents.orchestrator import ReviewOrchestrator
    
    # Initialize components
    github_client = GitHubClient(token=token, repo_name=repo)
    
    config_path = args.config
    rules_engine = RulesEngine(config_path=config_path)
    
    ollama_client = None
    use_llm = not args.no_llm
    
    if use_llm:
        ollama_client = OllamaClient()
        if not ollama_client.is_available():
            logger.warning("Ollama not available, falling back to static analysis only")
            use_llm = False
            ollama_client = None
    
    # Set up log path
    log_path = args.output
    if not log_path:
        log_dir = Path("examples")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"review_pr{pr_number}_{timestamp}.jsonl"
    
    # Create orchestrator and run review
    orchestrator = ReviewOrchestrator(
        github_client=github_client,
        rules_engine=rules_engine,
        ollama_client=ollama_client,
        use_llm=use_llm,
        log_path=log_path,
    )
    
    # Run the review
    final_state = orchestrator.run(pr_number)
    
    # Output results
    violations = final_state.get("violations", [])
    summary = final_state.get("summary")
    
    logger.info(f"Review completed: {len(violations)} violations found")
    
    if summary:
        print("\n" + "=" * 60)
        print(summary.to_markdown())
        print("=" * 60 + "\n")
    
    # Export violations to JSON
    if violations:
        violations_json = orchestrator.get_violations_json(final_state)
        print("\nViolations (JSON):")
        print(json.dumps(violations_json, indent=2))
    
    # Return exit code based on severity
    if any(v.severity.value in ("critical", "error") for v in violations):
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
