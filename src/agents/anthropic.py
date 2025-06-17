"""Anthropic Claude agent implementation."""

import os
import logging
from typing import Optional

from .base import BaseCodeReviewAgent

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None
    logger.warning("anthropic package not installed")


class AnthropicReviewAgent(BaseCodeReviewAgent):
    """Code review agent using Anthropic's Claude"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(f"anthropic/{model}")
        if not anthropic:
            raise ImportError("anthropic package not installed")
        
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def _call_model(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract text from the response - content blocks can have different types
        content = response.content[0]
        # For Anthropic API, we need to handle the content properly
        # The content is typically a TextBlock with a text attribute
        return getattr(content, 'text', str(content))