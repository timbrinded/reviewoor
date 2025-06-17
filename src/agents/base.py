"""Base class for code review agents."""

import json
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..models import CodeIssue, ReviewResult, Severity
from ..analyzers import CodeAnalyzer

logger = logging.getLogger(__name__)


class BaseCodeReviewAgent(ABC):
    """Abstract base class for code review agents"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.static_analyzer = CodeAnalyzer()
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
{
    "issues": [
        {
            "severity": "critical|high|medium|low|info",
            "category": "Bug|Security|Performance|Design|Style|Documentation",
            "message": "Clear description of the issue",
            "line_number": null or integer,
            "suggestion": "How to fix it"
        }
    ],
    "summary": "Brief overall assessment of the code quality",
    "metrics": {
        "complexity_score": 1-10,
        "maintainability_score": 1-10,
        "security_score": 1-10
    }
}

Focus on significant issues that would actually impact code quality or functionality."""
    
    @abstractmethod
    def _call_model(self, prompt: str) -> str:
        """Call the specific AI model - to be implemented by subclasses"""
        pass
    
    def review_code(self, code: str, file_path: str = "unknown.py") -> ReviewResult:
        """Review Python code and return detailed results"""
        start_time = time.time()
        
        # First run static analysis
        static_issues = self.static_analyzer.analyze(code)
        
        # Prepare static issues summary for the AI
        static_summary = self._format_static_issues(static_issues)
        
        # Get AI review
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
            model_used=self.model_name
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
    
    def _deduplicate_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Remove duplicate issues based on message and line number"""
        seen = set()
        unique_issues = []
        
        for issue in issues:
            key = (issue.message, issue.line_number, issue.category)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        return unique_issues