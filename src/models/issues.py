"""Code issue model."""

from dataclasses import dataclass
from typing import Optional

from .severity import Severity


@dataclass
class CodeIssue:
    """Represents a single code issue found during review"""
    severity: Severity
    category: str
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    confidence: float = 1.0  # 0-1 confidence score