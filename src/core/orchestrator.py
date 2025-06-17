"""Code review orchestrator for managing reviews across files."""

import os
import json
import glob
import logging
from typing import List, Dict, Any, Optional

from ..models import ReviewResult, Severity
from ..agents import AnthropicReviewAgent, OpenAIReviewAgent

logger = logging.getLogger(__name__)


class CodeReviewOrchestrator:
    """Orchestrates code reviews across multiple files and providers"""
    
    # Default models for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-4",
        "anthropic": "claude-3-5-sonnet-20241022"
    }
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None, enable_tools: bool = True, **kwargs):
        """Initialize with specified provider
        
        Args:
            provider: "anthropic" or "openai" (default: "openai")
            model: Model name to use, or None for provider default
            enable_tools: Whether to enable tool usage (default: True)
            **kwargs: Additional arguments passed to the agent (api_key, etc.)
        """
        # Default to OpenAI if not specified
        provider = provider.lower()
        
        # Use default model if not specified
        if model is None:
            model = self.DEFAULT_MODELS.get(provider)
            if model is None:
                raise ValueError(f"No default model for provider: {provider}")
        
        # Pass model and enable_tools to kwargs
        kwargs['model'] = model
        kwargs['enable_tools'] = enable_tools
        
        if provider == "anthropic":
            self.agent = AnthropicReviewAgent(**kwargs)
        elif provider == "openai":
            self.agent = OpenAIReviewAgent(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: 'openai', 'anthropic'")
        
        self.provider = provider
        self.model = model
        self.enable_tools = enable_tools
        self.results_history = []
    
    def review_file(self, file_path: str) -> ReviewResult:
        """Review a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        result = self.agent.review_code(code, file_path)
        self.results_history.append(result)
        return result
    
    def review_code_string(self, code: str, identifier: str = "snippet") -> ReviewResult:
        """Review a code string directly"""
        result = self.agent.review_code(code, identifier)
        self.results_history.append(result)
        return result
    
    def review_directory(self, directory: str, pattern: str = "*.py") -> List[ReviewResult]:
        """Review all Python files in a directory"""
        results = []
        py_files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
        
        for file_path in py_files:
            try:
                result = self.review_file(file_path)
                results.append(result)
                logger.info(f"Reviewed {file_path}: {len(result.issues)} issues found")
            except Exception as e:
                logger.error(f"Failed to review {file_path}: {e}")
        
        return results
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all reviews"""
        if not self.results_history:
            return {"error": "No reviews conducted yet"}
        
        total_issues = sum(len(r.issues) for r in self.results_history)
        severity_counts = {s.value: 0 for s in Severity}
        category_counts = {}
        
        for result in self.results_history:
            for issue in result.issues:
                severity_counts[issue.severity.value] += 1
                category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        avg_metrics = {}
        metric_keys = ["complexity_score", "maintainability_score", "security_score"]
        for key in metric_keys:
            values = [r.metrics.get(key, 0) for r in self.results_history if key in r.metrics]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        return {
            "total_files_reviewed": len(self.results_history),
            "total_issues": total_issues,
            "issues_by_severity": severity_counts,
            "issues_by_category": category_counts,
            "average_metrics": avg_metrics,
            "total_review_time": sum(r.review_time for r in self.results_history),
            "models_used": list(set(r.model_used for r in self.results_history))
        }
    
    def export_results(self, output_file: str = "review_results.json"):
        """Export all results to JSON file"""
        data = {
            "summary": self.get_summary_report(),
            "reviews": [result.to_dict() for result in self.results_history]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to {output_file}")