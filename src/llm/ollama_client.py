"""Ollama LLM client for code analysis."""

import os
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
        """
        Initialize Ollama client.
        
        Args:
            model: Model name. Defaults to OLLAMA_MODEL env var or codellama:7b.
            host: Ollama server URL. Defaults to OLLAMA_HOST env var or localhost:11434.
            timeout: Request timeout in seconds.
        """
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
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt.
            system: System prompt for context.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in response.
            
        Returns:
            Generated text response.
        """
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            logger.debug(f"Generated {len(generated_text)} chars")
            return generated_text.strip()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Ollama request error: {e}")
            raise
    
    def analyze_code(
        self,
        code: str,
        context: str = "",
        focus_areas: Optional[list[str]] = None,
    ) -> dict:
        """
        Analyze code for issues and improvements.
        
        Args:
            code: Code snippet to analyze.
            context: Additional context about the code.
            focus_areas: Specific areas to focus on (e.g., ["security", "performance"]).
            
        Returns:
            Dict with 'issues', 'suggestions', and 'summary' keys.
        """
        focus = ", ".join(focus_areas) if focus_areas else "code quality, security, and best practices"
        
        system_prompt = """You are an expert code reviewer. Analyze code for issues and provide actionable feedback.
Always respond in valid JSON format with the following structure:
{
    "issues": [
        {
            "severity": "critical|error|warning|info",
            "line": <line_number or null>,
            "description": "...",
            "suggestion": "..."
        }
    ],
    "summary": "Brief overall assessment"
}"""

        prompt = f"""Analyze this code focusing on {focus}:

```
{code}
```

{f"Context: {context}" if context else ""}

Provide your analysis as JSON. Be specific about line numbers when possible."""

        try:
            response = self.generate(prompt, system=system_prompt)
            
            # Try to parse JSON from response
            result = self._parse_json_response(response)
            return result
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {
                "issues": [],
                "summary": f"Analysis failed: {str(e)}",
            }
    
    def generate_review_comment(
        self,
        code: str,
        violation_type: str,
        violation_message: str,
    ) -> str:
        """
        Generate a helpful review comment for a violation.
        
        Args:
            code: Code snippet with the violation.
            violation_type: Type of violation (e.g., "hardcoded_secret").
            violation_message: Original violation message.
            
        Returns:
            Formatted review comment.
        """
        system_prompt = """You are a helpful code reviewer. Generate clear, actionable review comments.
Be concise but thorough. Include a brief explanation of why this is an issue and how to fix it."""

        prompt = f"""Generate a review comment for this code issue:

Violation type: {violation_type}
Message: {violation_message}

Code:
```
{code}
```

Write a helpful, professional comment explaining the issue and suggesting a fix."""

        try:
            comment = self.generate(prompt, system=system_prompt, max_tokens=512)
            return comment
        except Exception as e:
            logger.error(f"Failed to generate comment: {e}")
            return violation_message
    
    def suggest_refactoring(
        self,
        code: str,
        issues: list[str],
    ) -> dict:
        """
        Suggest refactored code to fix issues.
        
        Args:
            code: Original code.
            issues: List of issues to fix.
            
        Returns:
            Dict with 'refactored_code' and 'explanation' keys.
        """
        system_prompt = """You are an expert code refactorer. Fix code issues while preserving functionality.
Always respond in valid JSON format with:
{
    "refactored_code": "...",
    "explanation": "Brief explanation of changes made"
}"""

        issues_text = "\n".join(f"- {issue}" for issue in issues)
        
        prompt = f"""Refactor this code to fix the following issues:

{issues_text}

Original code:
```
{code}
```

Provide the refactored code and explain your changes as JSON."""

        try:
            response = self.generate(prompt, system=system_prompt, max_tokens=4096)
            result = self._parse_json_response(response)
            return result
        except Exception as e:
            logger.error(f"Refactoring suggestion failed: {e}")
            return {
                "refactored_code": code,
                "explanation": f"Refactoring failed: {str(e)}",
            }
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        import re
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
        
        # Return empty result
        logger.warning("Could not parse JSON from LLM response")
        return {"issues": [], "summary": response[:500]}
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self._client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def ensure_model(self) -> bool:
        """Ensure the model is available, pulling if necessary."""
        try:
            # Check if model exists
            response = self._client.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                return False
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if self.model in model_names or any(self.model in n for n in model_names):
                logger.info(f"Model {self.model} is available")
                return True
            
            # Pull model
            logger.info(f"Pulling model {self.model}...")
            pull_response = self._client.post(
                f"{self.host}/api/pull",
                json={"name": self.model},
                timeout=600.0,  # 10 minutes for model download
            )
            return pull_response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to ensure model: {e}")
            return False
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
