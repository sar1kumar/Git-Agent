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
    
    def __init__(
        self,
        token: Optional[str] = None,
        repo_name: Optional[str] = None,
    ):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token. Falls back to GITHUB_TOKEN env var.
            repo_name: Repository in 'owner/repo' format. Falls back to GITHUB_REPOSITORY env var.
        """
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
        """
        Fetch pull request details including changed files.
        
        Args:
            pr_number: Pull request number.
            
        Returns:
            PullRequest object with file changes.
        """
        logger.info(f"Fetching PR #{pr_number}")
        
        try:
            gh_pr: GHPullRequest = self._repo.get_pull(pr_number)
        except GithubException as e:
            logger.error(f"Failed to fetch PR #{pr_number}: {e}")
            raise
        
        # Get file changes
        files = []
        for file in gh_pr.get_files():
            file_change = FileChange(
                filename=file.filename,
                status=file.status,
                additions=file.additions,
                deletions=file.deletions,
                patch=file.patch,
            )
            files.append(file_change)
        
        pr = PullRequest(
            number=gh_pr.number,
            title=gh_pr.title,
            body=gh_pr.body or "",
            base_branch=gh_pr.base.ref,
            head_branch=gh_pr.head.ref,
            author=gh_pr.user.login,
            files=files,
            commits_count=gh_pr.commits,
        )
        
        logger.info(f"Fetched PR #{pr_number}: {len(files)} files changed")
        return pr
    
    def get_file_contents(self, path: str, ref: str) -> Optional[str]:
        """
        Get contents of a file at a specific ref.
        
        Args:
            path: File path in the repository.
            ref: Git ref (branch, tag, or commit SHA).
            
        Returns:
            File contents as string, or None if not found.
        """
        try:
            content = self._repo.get_contents(path, ref=ref)
            if isinstance(content, list):
                # Directory, not a file
                return None
            return content.decoded_content.decode("utf-8")
        except GithubException as e:
            if e.status == 404:
                logger.debug(f"File not found: {path} at {ref}")
                return None
            raise
    
    def post_review_comment(
        self,
        pr_number: int,
        comment: ReviewComment,
        commit_sha: str,
    ) -> bool:
        """
        Post a review comment on a specific line.
        
        Args:
            pr_number: Pull request number.
            comment: ReviewComment to post.
            commit_sha: Commit SHA to comment on.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            gh_pr = self._repo.get_pull(pr_number)
            gh_pr.create_review_comment(
                body=comment.format_body(),
                commit_id=self._repo.get_commit(commit_sha),
                path=comment.file,
                line=comment.line,
            )
            logger.info(f"Posted comment on {comment.file}:{comment.line}")
            return True
        except GithubException as e:
            logger.error(f"Failed to post comment: {e}")
            return False
    
    def post_review(
        self,
        pr_number: int,
        comments: list[ReviewComment],
        summary: str,
    ) -> bool:
        """
        Post a complete review with multiple comments.
        
        Args:
            pr_number: Pull request number.
            comments: List of ReviewComment objects.
            summary: Overall review summary.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            gh_pr = self._repo.get_pull(pr_number)
            commit = gh_pr.get_commits().reversed[0]
            
            # Build review comments
            review_comments = []
            for comment in comments:
                review_comments.append({
                    "path": comment.file,
                    "line": comment.line,
                    "body": comment.format_body(),
                })
            
            # Determine review event based on severity
            has_critical = any(c.severity == Severity.CRITICAL for c in comments)
            has_errors = any(c.severity == Severity.ERROR for c in comments)
            
            if has_critical or has_errors:
                event = "REQUEST_CHANGES"
            elif comments:
                event = "COMMENT"
            else:
                event = "APPROVE"
            
            gh_pr.create_review(
                commit=commit,
                body=summary,
                event=event,
                comments=review_comments if review_comments else None,
            )
            
            logger.info(f"Posted review with {len(comments)} comments, event={event}")
            return True
            
        except GithubException as e:
            logger.error(f"Failed to post review: {e}")
            # Fall back to posting individual comments
            return self._post_comments_individually(pr_number, comments, summary)
    
    def _post_comments_individually(
        self,
        pr_number: int,
        comments: list[ReviewComment],
        summary: str,
    ) -> bool:
        """Fallback method to post comments individually."""
        try:
            gh_pr = self._repo.get_pull(pr_number)
            
            # Post summary as issue comment
            gh_pr.create_issue_comment(summary)
            
            # Post individual line comments
            commit = gh_pr.get_commits().reversed[0]
            for comment in comments:
                try:
                    gh_pr.create_review_comment(
                        body=comment.format_body(),
                        commit=commit,
                        path=comment.file,
                        line=comment.line,
                    )
                except GithubException as e:
                    logger.warning(f"Could not post line comment: {e}")
            
            return True
        except GithubException as e:
            logger.error(f"Failed to post individual comments: {e}")
            return False
    
    def post_summary_comment(self, pr_number: int, summary: str) -> bool:
        """
        Post a summary comment on the PR.
        
        Args:
            pr_number: Pull request number.
            summary: Summary markdown content.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            gh_pr = self._repo.get_pull(pr_number)
            gh_pr.create_issue_comment(summary)
            logger.info(f"Posted summary comment on PR #{pr_number}")
            return True
        except GithubException as e:
            logger.error(f"Failed to post summary: {e}")
            return False
    
    def commit_file(
        self,
        pr_number: int,
        file_path: str,
        content: str,
        message: str,
    ) -> bool:
        """
        Commit a file change to the PR branch.
        
        Args:
            pr_number: Pull request number.
            file_path: Path to the file.
            content: New file content.
            message: Commit message.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            gh_pr = self._repo.get_pull(pr_number)
            branch = gh_pr.head.ref
            
            # Get current file to get its SHA
            try:
                current_file = self._repo.get_contents(file_path, ref=branch)
                sha = current_file.sha
            except GithubException:
                sha = None
            
            if sha:
                self._repo.update_file(
                    path=file_path,
                    message=message,
                    content=content,
                    sha=sha,
                    branch=branch,
                )
            else:
                self._repo.create_file(
                    path=file_path,
                    message=message,
                    content=content,
                    branch=branch,
                )
            
            logger.info(f"Committed changes to {file_path}")
            return True
            
        except GithubException as e:
            logger.error(f"Failed to commit file: {e}")
            return False
