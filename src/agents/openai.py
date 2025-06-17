"""OpenAI GPT agent implementation."""

import os
import logging
from typing import Optional, Any, Dict

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
        # Special handling for o3-pro models which use the responses endpoint
        if "o3-pro" in self.model:
            # The responses endpoint might be available as client.responses
            # or we might need to make a direct API call
            try:
                # Try the responses endpoint with correct parameters
                response: Any = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=2000
                )
                # The response structure might be different for responses endpoint
                # Try to extract content from the appropriate field
                if hasattr(response, 'choices') and response.choices:
                    choice: Any = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content = choice.message.content
                    elif hasattr(choice, 'text'):
                        content = choice.text
                    else:
                        content = str(choice)
                else:
                    content = str(response)
                return content if content else ""
            except (AttributeError, Exception) as e:
                # If responses endpoint is not available in the SDK yet,
                # we might need to use the chat completions with special params
                logger.warning(f"Responses endpoint error: {e}. Falling back to chat completions")
                fallback_params: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 2000
                }
                response: Any = self.client.chat.completions.create(**fallback_params)
                content: Optional[str] = response.choices[0].message.content
                return content if content else ""
        
        # Regular chat completions for other models
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Use max_completion_tokens for thinking models, max_tokens/temp for others
        if self.model.startswith("o"):
            params["max_completion_tokens"] = 2000
        else:
            params["temperature"] = 0.3
            params["max_tokens"] = 2000

        response: Any = self.client.chat.completions.create(**params)
        content: Optional[str] = response.choices[0].message.content
        if content is None:
            return ""
        return content
