"""Anthropic Claude agent implementation."""

import os
import logging
import json
from typing import Optional, List, Dict, Any, Tuple

import agentops
from agentops import track_agent, operation as track_function

from .base import BaseCodeReviewAgent

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None
    logger.warning("anthropic package not installed")


@track_agent(name="AnthropicReviewAgent")
class AnthropicReviewAgent(BaseCodeReviewAgent):
    """Code review agent using Anthropic's Claude"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", enable_tools: bool = True):
        super().__init__(f"anthropic/{model}", enable_tools=enable_tools)
        if not anthropic:
            raise ImportError("anthropic package not installed")
        
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    @track_function(name="anthropic_call_model")
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
    
    @track_function(name="anthropic_call_model_with_tools")
    def _call_model_with_tools(self, prompt: str, tools: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Call Anthropic model with tool support"""
        tool_calls_made = []
        
        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"]
                })
        
        messages = [{"role": "user", "content": prompt}]
        max_iterations = 3  # Reduced to avoid excessive tool calls
        iteration = 0
        
        while iteration < max_iterations:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=messages,
                tools=anthropic_tools
            )
            
            # Check if the model wants to use tools
            tool_use_found = False
            for content in response.content:
                if hasattr(content, 'type') and content.type == 'tool_use':
                    tool_use_found = True
                    
                    # Execute the tool
                    tool_result = self.execute_tool(content.name, content.input)
                    
                    # Track the tool call
                    tool_calls_made.append({
                        "tool": content.name,
                        "arguments": content.input,
                        "result": tool_result
                    })
                    
                    # Add assistant's message
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    # Add tool result
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(tool_result)
                        }]
                    })
                    
            if not tool_use_found:
                # Model provided final answer
                for content in response.content:
                    if hasattr(content, 'type') and content.type == 'text':
                        return content.text, tool_calls_made
                
                # Fallback if no text found
                return str(response.content), tool_calls_made
            
            iteration += 1
        
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