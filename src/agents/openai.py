"""OpenAI GPT agent implementation."""

import os
import logging
import json
from typing import Optional, Any, Dict, List, Tuple

from .base import BaseCodeReviewAgent

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None
    logger.warning("openai package not installed")


class OpenAIReviewAgent(BaseCodeReviewAgent):
    """Code review agent using OpenAI's GPT"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", enable_tools: bool = True):
        super().__init__(f"openai/{model}", enable_tools=enable_tools)
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
    
    def _call_model_with_tools(self, prompt: str, tools: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Call OpenAI model with tool support"""
        tool_calls_made = []
        
        # o3-pro models don't support tools yet
        if "o3-pro" in self.model:
            return self._call_model(prompt), []
        
        messages = [{"role": "user", "content": prompt}]
        
        # Initial call with tools
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }
        
        # Use appropriate token parameter
        if self.model.startswith("o"):
            params["max_completion_tokens"] = 2000
        else:
            params["temperature"] = 0.3
            params["max_tokens"] = 2000
        
        max_iterations = 3  # Prevent infinite loops, reduced to avoid excessive tool calls
        iteration = 0
        
        while iteration < max_iterations:
            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message
            
            # Add assistant's message to conversation
            messages.append(message.model_dump())
            
            # Check if the model wants to use tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    tool_result = self.execute_tool(function_name, function_args)
                    
                    # Track the tool call
                    tool_calls_made.append({
                        "tool": function_name,
                        "arguments": function_args,
                        "result": tool_result
                    })
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                
                # Continue the conversation
                params["messages"] = messages
                iteration += 1
            else:
                # Model provided final answer
                return message.content or "", tool_calls_made
        
        # Max iterations reached - return a default response
        logger.warning(f"Max tool iterations reached for {self.model}")
        default_response = json.dumps({
            "issues": [],
            "summary": "Analysis completed with tool assistance, but reached iteration limit.",
            "metrics": {
                "complexity_score": 5,
                "maintainability_score": 5,
                "security_score": 5
            }
        })
        return default_response, tool_calls_made
