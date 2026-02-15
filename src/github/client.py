"""GitHub API client for PR operations."""

import os
import logging
from typing import Optional

from github import Github, GithubException
from github.PullRequest import PullRequest as GHPullRequest

from .models import PullRequest, FileChange, ReviewComment, Severity

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for interacting with GitHub API."""

    def __init__(self, token: Optional[str] = None, repo_name: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable.")

        self.repo_name = repo_name or os.environ.get("GITHUB_REPOSITORY")
        if not self.repo_name:
            raise ValueError("Repository name is required. Set GITHUB_REPOSITORY environment variable.")

        self._github = Github(self.token)
        self._repo = self._github.get_repo(self.repo_name)
        logger.info(f"Initialized GitHub client for {self.repo_name}")

    def get_pull_request(self, pr_number: int) -> PullRequest:
        """Fetch pull request details including changed files."""
        logger.info(f"Fetching PR #{pr_number}")

        try:
            gh_pr: GHPullRequest = self._repo.get_pull(pr_number)
        except GithubException as e:
            logger.error(f"Failed to fetch PR #{pr_number}: {e}")
            raise

        files = [
            FileChange(
                filename=f.filename, status=f.status,
                additions=f.additions, deletions=f.deletions, patch=f.patch,
            )
            for f in gh_pr.get_files()
        ]

        pr = PullRequest(
            number=gh_pr.number, title=gh_pr.title, body=gh_pr.body or "",
            base_branch=gh_pr.base.ref, head_branch=gh_pr.head.ref,
            author=gh_pr.user.login, files=files, commits_count=gh_pr.commits,
        )

        logger.info(f"Fetched PR #{pr_number}: {len(files)} files changed")
        return pr

    def get_file_contents(self, path: str, ref: str) -> Optional[str]:
        """Get contents of a file at a specific git ref."""
        try:
            content = self._repo.get_contents(path, ref=ref)
            if isinstance(content, list):
                return None
            return content.decoded_content.decode("utf-8")
        except GithubException as e:
            if e.status == 404:
                logger.debug(f"File not found: {path} at {ref}")
                return None
            raise

    def post_review(
        self, pr_number: int, comments: list[ReviewComment], summary: str,
    ) -> bool:
        """Post a complete review with inline comments and a summary."""
        try:
            gh_pr = self._repo.get_pull(pr_number)
            commit = gh_pr.get_commits().reversed[0]

            review_comments = [
                {"path": c.file, "line": c.line, "body": c.format_body()}
                for c in comments
            ]

            has_critical = any(c.severity == Severity.CRITICAL for c in comments)
            has_errors = any(c.severity == Severity.ERROR for c in comments)

            if has_critical or has_errors:
                event = "REQUEST_CHANGES"
            elif comments:
                event = "COMMENT"
            else:
                event = "APPROVE"

            gh_pr.create_review(
                commit=commit, body=summary, event=event,
                comments=review_comments if review_comments else None,
            )
            logger.info(f"Posted review with {len(comments)} comments, event={event}")
            return True

        except GithubException as e:
            logger.error(f"Failed to post review: {e}")
            return self._post_fallback(pr_number, comments, summary)

    def _post_fallback(
        self, pr_number: int, comments: list[ReviewComment], summary: str,
    ) -> bool:
        """Fallback: post summary as issue comment + individual line comments."""
        try:
            gh_pr = self._repo.get_pull(pr_number)
            gh_pr.create_issue_comment(summary)

            commit = gh_pr.get_commits().reversed[0]
            for comment in comments:
                try:
                    gh_pr.create_review_comment(
                        body=comment.format_body(), commit=commit,
                        path=comment.file, line=comment.line,
                    )
                except GithubException as e:
                    logger.warning(f"Could not post line comment: {e}")
            return True
        except GithubException as e:
            logger.error(f"Failed to post fallback comments: {e}")
            return False

    def post_summary_comment(self, pr_number: int, summary: str) -> bool:
        """Post a summary comment on the PR."""
        try:
            gh_pr = self._repo.get_pull(pr_number)
            gh_pr.create_issue_comment(summary)
            logger.info(f"Posted summary comment on PR #{pr_number}")
            return True
        except GithubException as e:
            logger.error(f"Failed to post summary: {e}")
            return False

    def commit_file(
        self, pr_number: int, file_path: str, content: str, message: str,
    ) -> bool:
        """Commit a file change to the PR branch."""
        try:
            gh_pr = self._repo.get_pull(pr_number)
            branch = gh_pr.head.ref

            try:
                current_file = self._repo.get_contents(file_path, ref=branch)
                sha = current_file.sha
            except GithubException:
                sha = None

            if sha:
                self._repo.update_file(
                    path=file_path, message=message,
                    content=content, sha=sha, branch=branch,
                )
            else:
                self._repo.create_file(
                    path=file_path, message=message, content=content, branch=branch,
                )

            logger.info(f"Committed changes to {file_path}")
            return True
        except GithubException as e:
            logger.error(f"Failed to commit file: {e}")
            return False

    def get_commit_sha(self, pr_number: int) -> Optional[str]:
        """Get the latest commit SHA from a PR."""
        try:
            gh_pr = self._repo.get_pull(pr_number)
            commits = list(gh_pr.get_commits())
            return commits[-1].sha if commits else None
        except GithubException as e:
            logger.error(f"Failed to get commit SHA: {e}")
            return None

    def revert_to_commit(
        self, pr_number: int, commit_sha: str, files: list[str],
    ) -> bool:
        """Revert specific files to their state at a given commit."""
        try:
            gh_pr = self._repo.get_pull(pr_number)
            branch = gh_pr.head.ref

            for file_path in files:
                try:
                    old_content = self._repo.get_contents(file_path, ref=commit_sha)
                    if isinstance(old_content, list):
                        continue
                    content = old_content.decoded_content.decode("utf-8")
                    current_file = self._repo.get_contents(file_path, ref=branch)

                    self._repo.update_file(
                        path=file_path,
                        message=f"Rollback: Revert {file_path} to {commit_sha[:7]}",
                        content=content, sha=current_file.sha, branch=branch,
                    )
                    logger.info(f"Reverted {file_path} to {commit_sha[:7]}")
                except GithubException as e:
                    logger.warning(f"Could not revert {file_path}: {e}")

            return True
        except GithubException as e:
            logger.error(f"Failed to revert: {e}")
            return False
