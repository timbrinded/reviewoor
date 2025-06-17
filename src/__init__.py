"""
AgentOps - Python Code Review Agent
A full-featured AI agent that reviews Python code for bugs, style issues, and improvements.
Supports both Anthropic Claude and OpenAI GPT models.
"""

import logging

from .models import Severity, CodeIssue, ReviewResult
from .analyzers import CodeAnalyzer
from .agents import BaseCodeReviewAgent, AnthropicReviewAgent, OpenAIReviewAgent
from .core import CodeReviewOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)

__version__ = "0.1.0"

__all__ = [
    # Models
    "Severity",
    "CodeIssue", 
    "ReviewResult",
    
    # Analyzers
    "CodeAnalyzer",
    
    # Agents
    "BaseCodeReviewAgent",
    "AnthropicReviewAgent",
    "OpenAIReviewAgent",
    
    # Core
    "CodeReviewOrchestrator",
]