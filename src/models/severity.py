"""Severity levels for code issues."""

from enum import Enum


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"  # Security vulnerabilities, crashes
    HIGH = "high"         # Bugs, logic errors
    MEDIUM = "medium"     # Style issues, performance
    LOW = "low"          # Suggestions, minor improvements
    INFO = "info"        # Informational notes