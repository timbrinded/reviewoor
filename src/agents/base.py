"""Base class for code review agents."""

import json
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import agentops
from agentops import track_agent, operation as track_function

from ..models import CodeIssue, ReviewResult, Severity
from ..analyzers import CodeAnalyzer
from ..tools import (
    check_imports, analyze_complexity, find_security_patterns,
    check_type_consistency, detect_code_smells, get_function_metrics,
    TOOL_DEFINITIONS
)

logger = logging.getLogger(__name__)


@track_agent(name="BaseCodeReviewAgent")
class BaseCodeReviewAgent(ABC):
    """Abstract base class for code review agents"""
    
    def __init__(self, model_name: str, enable_tools: bool = True):
        self.model_name = model_name
        self.static_analyzer = CodeAnalyzer()
        self.enable_tools = enable_tools
        self.tool_calls = []  # Track tool usage
        self.available_tools = {
            "check_imports": check_imports,
            "analyze_complexity": analyze_complexity,
            "find_security_patterns": find_security_patterns,
            "check_type_consistency": check_type_consistency,
            "detect_code_smells": detect_code_smells,
            "get_function_metrics": get_function_metrics
        }
        self.review_prompt_template = """You are an expert Python code reviewer. Analyze the following code and provide a detailed review.

Code to review:
```python
{code}
```

Static analysis already found these issues:
{static_issues}

Please provide:
1. Additional bugs, logic errors, or potential crashes not caught by static analysis
2. Security vulnerabilities
3. Performance issues
4. Design problems and code smells
5. Suggestions for improvement

Format your response as a JSON object with this structure:
{{
    "issues": [
        {{
            "severity": "critical|high|medium|low|info",
            "category": "Bug|Security|Performance|Design|Style|Documentation",
            "message": "Clear description of the issue",
            "line_number": null or integer,
            "suggestion": "How to fix it"
        }}
    ],
    "summary": "Brief overall assessment of the code quality",
    "metrics": {{
        "complexity_score": 1-10,
        "maintainability_score": 1-10,
        "security_score": 1-10
    }}
}}

Focus on significant issues that would actually impact code quality or functionality."""
        
        self.review_prompt_with_tools = """You are an expert Python code reviewer with access to specialized analysis tools.

Code to review:
```python
{code}
```

Static analysis already found these issues:
{static_issues}

You have access to the following tools to help analyze the code:
- check_imports: Analyze imports for unused imports and import issues
- analyze_complexity: Check cyclomatic complexity of functions
- find_security_patterns: Scan for security vulnerabilities
- check_type_consistency: Check for type consistency issues
- detect_code_smells: Find code smells like long lines, TODOs, magic numbers
- get_function_metrics: Get metrics like lines of code, parameters, documentation

Use these tools as needed to gather information, but aim to complete your analysis within 2-3 tool calls. 

IMPORTANT: After using tools, you MUST provide your final review as a JSON response. Do not continue calling tools indefinitely.

Format your final response as a JSON object with this structure:
{{
    "issues": [
        {{
            "severity": "critical|high|medium|low|info",
            "category": "Bug|Security|Performance|Design|Style|Documentation",
            "message": "Clear description of the issue",
            "line_number": null or integer,
            "suggestion": "How to fix it"
        }}
    ],
    "summary": "Brief overall assessment of the code quality",
    "metrics": {{
        "complexity_score": 1-10,
        "maintainability_score": 1-10,
        "security_score": 1-10
    }}
}}"""
    
    @abstractmethod
    def _call_model(self, prompt: str) -> str:
        """Call the specific AI model - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _call_model_with_tools(self, prompt: str, tools: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Call the specific AI model with tools - to be implemented by subclasses"""
        pass
    
    @track_function(name="review_code")
    def review_code(self, code: str, file_path: str = "unknown.py") -> ReviewResult:
        """Review Python code and return detailed results"""
        start_time = time.time()
        self.tool_calls = []  # Reset tool calls for this review
        
        # First run static analysis
        static_issues = self.static_analyzer.analyze(code)
        
        # Prepare static issues summary for the AI
        static_summary = self._format_static_issues(static_issues)
        
        # Get AI review
        if self.enable_tools:
            prompt = self.review_prompt_with_tools.format(
                code=code,
                static_issues=static_summary
            )
            
            try:
                ai_response, tool_calls = self._call_model_with_tools(prompt, TOOL_DEFINITIONS)
                self.tool_calls = tool_calls
                ai_result = self._parse_ai_response(ai_response)
            except Exception as e:
                logger.error(f"AI model with tools error: {e}")
                # Fallback to regular review
                prompt = self.review_prompt_template.format(
                    code=code,
                    static_issues=static_summary
                )
                try:
                    ai_response = self._call_model(prompt)
                    ai_result = self._parse_ai_response(ai_response)
                except Exception as e2:
                    logger.error(f"AI model error: {e2}")
                    ai_result = {
                        "issues": [],
                        "summary": f"AI review failed: {str(e2)}",
                        "metrics": {}
                    }
        else:
            prompt = self.review_prompt_template.format(
                code=code,
                static_issues=static_summary
            )
            
            try:
                ai_response = self._call_model(prompt)
                ai_result = self._parse_ai_response(ai_response)
            except Exception as e:
                logger.error(f"AI model error: {e}")
                ai_result = {
                    "issues": [],
                    "summary": f"AI review failed: {str(e)}",
                    "metrics": {}
                }
        
        # Combine results
        all_issues = static_issues + [
            CodeIssue(
                severity=Severity(issue["severity"]),
                category=issue["category"],
                message=issue["message"],
                line_number=issue.get("line_number"),
                suggestion=issue.get("suggestion"),
                confidence=0.8  # AI issues have slightly lower confidence
            )
            for issue in ai_result.get("issues", [])
        ]
        
        # Remove duplicates
        all_issues = self._deduplicate_issues(all_issues)
        
        review_time = time.time() - start_time
        
        return ReviewResult(
            file_path=file_path,
            issues=all_issues,
            summary=ai_result.get("summary", ""),
            metrics=ai_result.get("metrics", {}),
            review_time=review_time,
            model_used=self.model_name,
            tool_calls=self.tool_calls  # Include tool usage info
        )
    
    def _format_static_issues(self, issues: List[CodeIssue]) -> str:
        """Format static issues for the AI prompt"""
        if not issues:
            return "No issues found by static analysis."
        
        formatted = []
        for issue in issues[:10]:  # Limit to avoid token overflow
            line_info = f"Line {issue.line_number}: " if issue.line_number else ""
            formatted.append(f"- {line_info}{issue.category} - {issue.message}")
        
        if len(issues) > 10:
            formatted.append(f"... and {len(issues) - 10} more issues")
        
        return "\n".join(formatted)
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response, handling potential JSON errors"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON")
            return {
                "issues": [],
                "summary": "Failed to parse AI response",
                "metrics": {}
            }
    
    @track_function(name="deduplicate_issues")
    def _deduplicate_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Remove duplicate issues based on message and line number"""
        seen = set()
        unique_issues = []
        
        for issue in issues:
            # Convert line_number to a hashable type
            line_num = issue.line_number
            if isinstance(line_num, list):
                line_num = tuple(line_num) if line_num else None
            
            key = (issue.message, line_num, issue.category)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        return unique_issues
    
    @track_function(name="execute_tool")
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with the given arguments"""
        if tool_name not in self.available_tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        try:
            tool_func = self.available_tools[tool_name]
            result = tool_func(**arguments)
            return result
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return {"success": False, "error": str(e)}