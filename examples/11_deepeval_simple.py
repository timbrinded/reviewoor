#!/usr/bin/env python3
"""
Example 11: Simplified DeepEval Integration
A simpler example showing DeepEval for code review evaluation.
"""

import os
import sys
import time
from dotenv import load_dotenv
from typing import Dict, List

# DeepEval imports
import deepeval
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.evaluate.configs import DisplayConfig

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

# Simple test case
TEST_CODE = '''
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)  # Potential division by zero
'''

EXPECTED_ISSUES = [
    "division by zero when numbers list is empty",
    "no input validation",
    "missing docstring"
]


def create_code_review_metric():
    """Create a simple G-Eval metric for code review evaluation"""
    return GEval(
        name="Code Review Quality",
        criteria="Evaluate if the code review identifies key issues like bugs, edge cases, and code quality problems.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5  # Set a reasonable threshold
    )


def format_review_output(result):
    """Format the review result into a readable string"""
    output = f"Summary: {result.summary}\n\n"
    output += f"Found {len(result.issues)} issues:\n"
    
    for issue in result.issues:
        line_info = f"Line {issue.line_number}: " if issue.line_number else ""
        output += f"- [{issue.severity.value.upper()}] {line_info}{issue.message}\n"
        if issue.suggestion:
            output += f"  Suggestion: {issue.suggestion}\n"
    
    return output


def main():
    print("=== Simplified DeepEval Code Review Evaluation ===\n")
    
    # Initialize code reviewer
    orchestrator = CodeReviewOrchestrator(provider="openai", model="gpt-4o-mini-2024-07-18")
    
    print("1. Running code review...")
    start_time = time.time()
    result = orchestrator.review_code_string(TEST_CODE, "test.py")
    review_time = time.time() - start_time
    
    # Format the review
    review_output = format_review_output(result)
    print(f"   Completed in {review_time:.2f}s")
    print(f"   Found {len(result.issues)} issues")
    
    # Create DeepEval test case
    test_case = LLMTestCase(
        input=f"Review this Python code: {TEST_CODE}",
        actual_output=review_output,
        expected_output="The review should identify: " + ", ".join(EXPECTED_ISSUES)
    )
    
    # Create metric
    metric = create_code_review_metric()
    
    print("\n2. Evaluating with DeepEval...")
    
    # Run evaluation
    evaluation_result = evaluate(
        test_cases=[test_case],
        metrics=[metric],
        display_config=DisplayConfig(print_results=False, show_indicator=False)
    )
    
    # Display results
    test_result = evaluation_result.test_results[0]
    metric_data = test_result.metrics_data[0]
    
    # Access metric data properly - metrics_data is a list of MetricData objects
    score = metric_data.score
    metric_name = metric_data.name
    reason = metric_data.reason if metric_data.reason else ''
    
    print(f"   Score: {score:.2f}/1.00")
    print(f"   Success: {'✓' if test_result.success else '✗'}")
    
    # Show the review output
    print("\n3. Review Output:")
    print("-" * 50)
    print(review_output)
    print("-" * 50)
    
    # Show reasoning (if available)
    if reason:
        print("\n4. DeepEval Reasoning:")
        print(reason)
    
    print("\n✨ Evaluation complete!")
    
    # Usage tips
    print("\n💡 DeepEval Benefits:")
    print("  - LLM-based evaluation for nuanced assessment")
    print("  - Can evaluate subjective qualities like completeness")
    print("  - Provides reasoning for scores")
    print("  - Easy integration with CI/CD pipelines")


if __name__ == "__main__":
    main()