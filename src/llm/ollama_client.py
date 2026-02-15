"""Ollama LLM client for code analysis."""

import os
import re
import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    DEFAULT_MODEL = "codellama:7b"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        logger.info(f"Initialized Ollama client with model={self.model}, host={self.host}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from the LLM."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        try:
            response = self._client.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            generated_text = response.json().get("response", "")
            logger.debug(f"Generated {len(generated_text)} chars")
            return generated_text.strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Ollama request error: {e}")
            raise

    def analyze_code(
        self, code: str, context: str = "", focus_areas: Optional[list[str]] = None,
    ) -> dict:
        """Analyze code for issues and improvements. Returns dict with 'issues' and 'summary'."""
        focus = ", ".join(focus_areas) if focus_areas else "code quality, security, and best practices"

        system_prompt = (
            "You are an expert code reviewer. Analyze code for issues and provide actionable feedback.\n"
            'Always respond in valid JSON format with: {"issues": [{"severity": "...", "line": null, '
            '"description": "...", "suggestion": "..."}], "summary": "..."}'
        )
        prompt = f"Analyze this code focusing on {focus}:\n\n```\n{code}\n```\n"
        if context:
            prompt += f"\nContext: {context}\n"
        prompt += "\nProvide your analysis as JSON."

        try:
            response = self.generate(prompt, system=system_prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {"issues": [], "summary": f"Analysis failed: {e}"}

    def generate_review_comment(self, code: str, violation_type: str, violation_message: str) -> str:
        """Generate a helpful review comment for a violation."""
        system_prompt = (
            "You are a helpful code reviewer. Generate clear, actionable review comments. "
            "Be concise but thorough."
        )
        prompt = (
            f"Generate a review comment for this code issue:\n\n"
            f"Violation type: {violation_type}\nMessage: {violation_message}\n\n"
            f"Code:\n```\n{code}\n```\n\nWrite a helpful, professional comment."
        )
        try:
            return self.generate(prompt, system=system_prompt, max_tokens=512)
        except Exception as e:
            logger.error(f"Failed to generate comment: {e}")
            return violation_message

    def suggest_refactoring(self, code: str, issues: list[str]) -> dict:
        """Suggest refactored code to fix issues."""
        system_prompt = (
            "You are an expert code refactorer. Fix code issues while preserving functionality.\n"
            'Always respond in valid JSON: {"refactored_code": "...", "explanation": "..."}'
        )
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        prompt = f"Refactor this code to fix:\n\n{issues_text}\n\nOriginal code:\n```\n{code}\n```"

        try:
            response = self.generate(prompt, system=system_prompt, max_tokens=4096)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Refactoring suggestion failed: {e}")
            return {"refactored_code": code, "explanation": f"Refactoring failed: {e}"}

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(response[json_start:json_end])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from LLM response")
        return {"issues": [], "summary": response[:500]}

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self._client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
