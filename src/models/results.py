"""Review result model."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from .issues import CodeIssue


@dataclass
class ReviewResult:
    """Complete review result for a code file"""
    file_path: str
    issues: List[CodeIssue] = field(default_factory=list)
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    review_time: float = 0.0
    model_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "message": issue.message,
                    "line_number": issue.line_number,
                    "column": issue.column,
                    "suggestion": issue.suggestion,
                    "confidence": issue.confidence
                }
                for issue in self.issues
            ],
            "summary": self.summary,
            "metrics": self.metrics,
            "review_time": self.review_time,
            "model_used": self.model_used
        }