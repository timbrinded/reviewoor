#!/usr/bin/env python3
"""
Example 10: DeepEval Benchmarking
This example uses DeepEval framework to evaluate code review models.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# DeepEval imports
import deepeval
from deepeval import evaluate
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval.g_eval import GEval

# Suppress INFO logs from httpx
logging.getLogger("httpx").setLevel(logging.WARN)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Disable browser opening when running DeepEval
os.environ["CI"] = "true"

# Configure DeepEval with Confident AI API key
confident_api_key = os.getenv("CONFIDENT_API_KEY")
if confident_api_key:
    deepeval.login_with_confident_api_key(api_key=confident_api_key)
    print("✓ Logged in to DeepEval with Confident AI")
else:
    print("⚠️  No CONFIDENT_API_KEY found - results won't be synced to Confident AI dashboard")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

# Test cases with known issues for evaluation
TEST_CASES = {
    "security_vulnerabilities": {
        "code": '''
import os
import pickle
import subprocess

def execute_command(user_input):
    # Command injection vulnerability
    result = subprocess.call(user_input, shell=True)
    return result

def load_data(filename):
    # Insecure deserialization
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_user_data(user_id):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query

API_KEY = "sk-1234567890abcdef"  # Hardcoded secret
''',
        "expected_issues": [
            "subprocess.call with shell=True allows command injection",
            "pickle.load can execute arbitrary code",
            "SQL query using f-string is vulnerable to injection",
            "hardcoded API key or credential in code"
        ]
    },
    
    "performance_issues": {
        "code": '''
def fibonacci(n):
    # Inefficient recursive implementation
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def find_duplicates(items):
    # O(n²) complexity when could be O(n)
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates
''',
        "expected_issues": [
            "fibonacci has exponential time complexity without memoization",
            "find_duplicates has O(n²) complexity when O(n) is possible with a set"
        ]
    },
    
    "code_quality": {
        "code": '''
def processData(d):
    # Poor naming, no docstring, no type hints
    r = []
    for i in d:
        if i > 0:
            r.append(i * 2)
    return r

class manager:  # Class name not capitalized
    def __init__(self):
        self.data = []
    
    def AddItem(self, item):  # Method name not snake_case
        self.data.append(item)

def calculate_metrics(data=[]):  # Mutable default argument
    if data == None:  # Should use 'is None'
        data = []
    
    try:
        result = sum(data) / len(data)
    except:  # Bare except
        result = 0
    
    return result
''',
        "expected_issues": [
            "function and variable names should be descriptive (processData, r, d, i)",
            "missing docstrings on functions and classes",
            "mutable default argument (data=[]) can cause bugs",
            "bare except clause catches all exceptions including system ones",
            "use 'is None' for None comparison, not '== None'"
        ]
    }
}


class CodeReviewAccuracyMetric(GEval):
    """Custom G-Eval metric for code review accuracy"""
    
    def __init__(self, **kwargs):
        # Filter out any unexpected kwargs that DeepEval might pass
        super().__init__(
            name="Code Review Accuracy",
            criteria=(
                "Evaluate whether the code review identifies the core issues present in the code. "
                "Focus on whether the fundamental problems are caught, not exact wording. "
                "The review should identify security risks, performance problems, and quality issues."
            ),
            evaluation_steps=[
                "1. Check if the actual output identifies the same core issues as the expected output",
                "2. Give credit for identifying issues even if described differently",
                "3. Critical security issues (command injection, SQL injection, etc.) must be caught",
                "4. Performance bottlenecks should be identified even if solutions differ",
                "5. Code quality issues can be described in various ways - focus on the problem identified",
                "6. Don't penalize for additional valid issues found beyond the expected ones"
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            strict_mode=False,
            threshold=0.7  # 70% score is passing, allowing for some variation
        )


class CodeReviewCompletenessMetric(GEval):
    """Custom G-Eval metric for review completeness"""
    
    def __init__(self, **kwargs):
        # Filter out any unexpected kwargs that DeepEval might pass
        super().__init__(
            name="Code Review Completeness",
            criteria=(
                "Evaluate the completeness and quality of the code review. "
                "A good review should provide actionable suggestions, explain issues clearly, "
                "and prioritize them by severity."
            ),
            evaluation_steps=[
                "1. Check if the review provides clear explanations for each issue",
                "2. Verify that actionable suggestions are provided",
                "3. Check if issues are properly categorized by severity",
                "4. Ensure the review covers multiple aspects (security, performance, quality)",
                "5. Verify that line numbers or specific locations are mentioned when applicable"
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            threshold=0.6  # 60% for completeness, as this is more subjective
        )


def create_test_case(test_name: str, code: str, review_result: str, expected_issues: List[str]) -> LLMTestCase:
    """Create a DeepEval test case from code review results"""
    expected_output = (
        "The code review should identify these core issues:\n" +
        "\n".join([f"- {issue}" for issue in expected_issues]) +
        "\n\nNote: The exact wording may vary, focus on whether the core problem is identified."
    )
    
    return LLMTestCase(
        input=f"Review this Python code:\n\n{code}",
        actual_output=review_result,
        expected_output=expected_output,
        additional_metadata={
            "test_name": test_name,
            "code_length": len(code),
            "expected_issue_count": len(expected_issues)
        }
    )


def format_review_for_deepeval(result) -> str:
    """Format ReviewResult into a string for DeepEval evaluation"""
    review_text = f"Code Review Summary: {result.summary}\n\n"
    review_text += "Issues Found:\n"
    
    # Group issues by severity
    issues_by_severity = {}
    for issue in result.issues:
        severity = issue.severity.value
        if severity not in issues_by_severity:
            issues_by_severity[severity] = []
        issues_by_severity[severity].append(issue)
    
    # Format issues
    for severity in ["critical", "high", "medium", "low", "info"]:
        if severity in issues_by_severity:
            review_text += f"\n{severity.upper()} Priority:\n"
            for issue in issues_by_severity[severity]:
                line_info = f"Line {issue.line_number}: " if issue.line_number else ""
                review_text += f"- {line_info}{issue.message}"
                if issue.suggestion:
                    review_text += f" (Suggestion: {issue.suggestion})"
                review_text += "\n"
    
    # Add metrics if available
    if result.metrics:
        review_text += "\nCode Metrics:\n"
        for key, value in result.metrics.items():
            review_text += f"- {key}: {value}\n"
    
    return review_text


def benchmark_model_with_deepeval(provider: str, model: str, test_cases: Dict[str, Dict]) -> Dict[str, Any]:
    """Benchmark a model using DeepEval framework"""
    print(f"\n📊 Benchmarking {provider}/{model} with DeepEval...")
    
    # Initialize orchestrator
    orchestrator = CodeReviewOrchestrator(provider=provider, model=model, enable_tools=True)
    
    # Create metrics
    accuracy_metric = CodeReviewAccuracyMetric()
    completeness_metric = CodeReviewCompletenessMetric()
    
    # Create test cases
    deepeval_test_cases = []
    total_time = 0
    
    for test_name, test_data in test_cases.items():
        print(f"  Testing {test_name}...", end="", flush=True)
        
        start_time = time.time()
        result = orchestrator.review_code_string(test_data["code"], f"{test_name}.py")
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Format review result for DeepEval
        review_text = format_review_for_deepeval(result)
        
        # Create DeepEval test case
        test_case = create_test_case(
            test_name=test_name,
            code=test_data["code"],
            review_result=review_text,
            expected_issues=test_data["expected_issues"]
        )
        
        deepeval_test_cases.append(test_case)
        print(f" ✓ ({len(result.issues)} issues, {elapsed:.2f}s)")
    
    # Run evaluation
    print("\n  Running DeepEval metrics...")
    from deepeval.evaluate.configs import DisplayConfig
    
    evaluation_result = evaluate(
        test_cases=deepeval_test_cases,
        metrics=[accuracy_metric, completeness_metric],
        display_config=DisplayConfig(print_results=False, show_indicator=False)
    )
    
    # Calculate aggregate scores
    accuracy_scores = []
    completeness_scores = []
    
    # Process test results
    for i, test_result in enumerate(evaluation_result.test_results):
        test_name = deepeval_test_cases[i].additional_metadata["test_name"]
        print(f"\n  {test_name}:")
        
        # Get scores from metrics_data - it's a list of MetricData objects
        if test_result.metrics_data:
            for metric_data in test_result.metrics_data:
                # Access the MetricData object attributes
                metric_name = metric_data.name
                score = metric_data.score
                
                # Extract just the metric name (remove "(GEval)" suffix)
                if " (GEval)" in metric_name:
                    metric_name = metric_name.replace(" (GEval)", "")
                
                if metric_name == "Code Review Accuracy":
                    accuracy_scores.append(score)
                    print(f"    Accuracy: {score:.2f}")
                    if metric_data.reason:
                        print(f"      Reason: {metric_data.reason[:100]}...")
                elif metric_name == "Code Review Completeness":
                    completeness_scores.append(score)
                    print(f"    Completeness: {score:.2f}")
                    if metric_data.reason:
                        print(f"      Reason: {metric_data.reason[:100]}...")
    
    # Calculate summary metrics
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
    
    return {
        "model": f"{provider}/{model}",
        "avg_accuracy": avg_accuracy,
        "avg_completeness": avg_completeness,
        "total_time": total_time,
        "avg_time_per_test": total_time / len(test_cases),
        "evaluation_result": evaluation_result
    }


def main():
    print("=== DeepEval Model Benchmarking ===")
    print("Using DeepEval framework for LLM evaluation\n")
    
    # Models to benchmark
    if os.getenv("TEST_SINGLE_MODEL"):
        # For testing, just use one fast model
        models_to_test = [
            ("openai", "gpt-4o-mini-2024-07-18"),
        ]
    else:
        models_to_test = [
            ("openai", "gpt-4o-2024-08-06"),
            ("openai", "gpt-4.1-2025-04-14"),
            ("openai", "gpt-4.1-mini-2025-04-14"),
            ("openai", "gpt-4.1-nano-2025-04-14"),
            ("openai", "gpt-4o-mini-2024-07-18"),
        ]
    
    # Add Anthropic models if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.extend([
            ("anthropic", "claude-sonnet-4-20250514"),
            ("anthropic", "claude-3-7-sonnet-20250219"),
            ("anthropic", "claude-3-5-haiku-20241022 "),
        ])
    
    # Run benchmarks
    all_results = []
    
    for provider, model in models_to_test:
        try:
            results = benchmark_model_with_deepeval(provider, model, TEST_CASES)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Failed to benchmark {provider}/{model}: {e}")
            continue
    
    # Display summary
    print("\n" + "="*80)
    print("DEEPEVAL BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n{'Model':<35} {'Accuracy':<12} {'Completeness':<15} {'Avg Time':<10}")
    print("-"*72)
    
    for result in all_results:
        print(f"{result['model']:<35} {result['avg_accuracy']:<12.2f} "
              f"{result['avg_completeness']:<15.2f} {result['avg_time_per_test']:<10.2f}s")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"deepeval_results_{timestamp}.json")
    
    # Prepare serializable results
    serializable_results = []
    for result in all_results:
        serializable_result = {
            "model": result["model"],
            "avg_accuracy": result["avg_accuracy"],
            "avg_completeness": result["avg_completeness"],
            "total_time": result["total_time"],
            "avg_time_per_test": result["avg_time_per_test"]
        }
        serializable_results.append(serializable_result)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Best model by accuracy
    if all_results:
        best_model = max(all_results, key=lambda x: x['avg_accuracy'])
        print(f"\n🏆 Best model by accuracy: {best_model['model']} ({best_model['avg_accuracy']:.2f})")
    
    print("\n✨ DeepEval benchmark complete!")
    print("\n📊 Key Insights:")
    print("  - DeepEval provides LLM-based evaluation for more nuanced assessment")
    print("  - Accuracy metric evaluates if expected issues are found")
    print("  - Completeness metric evaluates review quality and actionability")
    print("  - Results show both detection capability and review quality")


if __name__ == "__main__":
    main()