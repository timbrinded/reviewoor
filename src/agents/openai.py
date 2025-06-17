"""OpenAI GPT agent implementation."""

import os
import logging
from typing import Optional

from .base import BaseCodeReviewAgent

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None
    logger.warning("openai package not installed")


class OpenAIReviewAgent(BaseCodeReviewAgent):
    """Code review agent using OpenAI's GPT"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        super().__init__(f"openai/{model}")
        if not openai:
            raise ImportError("openai package not installed")
        
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def _call_model(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content