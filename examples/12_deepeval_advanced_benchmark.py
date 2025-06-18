#!/usr/bin/env python3
"""
Example 12: Advanced DeepEval Benchmarking with Dataset Integration
This example demonstrates:
- Different run types (with/without tools, different prompts)
- DeepEval dataset integration for test case management
- Comprehensive model comparison
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# DeepEval imports
import deepeval
from deepeval import evaluate
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.dataset.dataset import EvaluationDataset
from deepeval.evaluate.configs import DisplayConfig

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

# Test cases organized by category
TEST_SUITES = {
    "security": {
        "command_injection": {
            "code": '''
import subprocess

def run_command(user_input):
    # Direct command injection vulnerability
    result = subprocess.call(user_input, shell=True)
    return result

def run_safe_command(cmd_parts):
    # Safer approach without shell=True
    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    return result.stdout
''',
            "expected_issues": [
                "subprocess.call with shell=True allows command injection",
                "user input is passed directly to shell without validation"
            ],
            "category": "security",
            "severity": "critical"
        },
        
        "sql_injection": {
            "code": '''
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    
    return cursor.fetchone()

def search_users(name):
    # Another SQL injection
    query = "SELECT * FROM users WHERE name LIKE '%" + name + "%'"
    cursor.execute(query)
    return cursor.fetchall()
''',
            "expected_issues": [
                "SQL injection via f-string formatting",
                "SQL injection via string concatenation",
                "use parameterized queries instead"
            ],
            "category": "security",
            "severity": "critical"
        }
    },
    
    "performance": {
        "inefficient_algorithms": {
            "code": '''
def find_common_elements(list1, list2):
    # O(n*m) complexity
    common = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 and item1 not in common:
                common.append(item1)
    return common

def count_occurrences(items):
    # Inefficient counting
    counts = {}
    for item in items:
        count = 0
        for i in items:
            if i == item:
                count += 1
        counts[item] = count
    return counts
''',
            "expected_issues": [
                "nested loops create O(n*m) complexity",
                "could use set intersection for O(n+m)",
                "counting has O(n²) complexity instead of O(n)"
            ],
            "category": "performance",
            "severity": "high"
        }
    },
    
    "code_quality": {
        "poor_practices": {
            "code": '''
def process(d):
    r = []
    for i in d:
        if i > 0:
            r.append(i * 2)
    return r

class data_manager:  # Wrong naming convention
    def __init__(self):
        self.items = []
    
    def AddData(self, data):  # Wrong naming convention
        self.items.append(data)

def calculate(numbers=[]):  # Mutable default
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)  # No zero check
''',
            "expected_issues": [
                "poor variable naming (d, r, i)",
                "class name should be CamelCase",
                "method name should be snake_case",
                "mutable default argument",
                "potential division by zero"
            ],
            "category": "quality",
            "severity": "medium"
        }
    }
}


class CodeReviewAccuracyMetric(GEval):
    """Custom G-Eval metric for code review accuracy"""
    
    def __init__(self, run_type: str = "standard", **kwargs):
        super().__init__(
            name=f"Code Review Accuracy ({run_type})",
            criteria=(
                "Evaluate whether the code review identifies the core issues present in the code. "
                "Focus on whether the fundamental problems are caught, not exact wording. "
                "The review should identify security risks, performance problems, and quality issues."
            ),
            evaluation_steps=[
                "1. Check if the actual output identifies the same core issues as the expected output",
                "2. Give credit for identifying issues even if described differently",
                "3. Critical security issues must be caught for security test cases",
                "4. Performance bottlenecks should be identified for performance cases",
                "5. Code quality issues should be identified for quality cases",
                "6. Don't penalize for additional valid issues found"
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            strict_mode=False,
            threshold=0.7
        )


class CodeReviewCompletenessMetric(GEval):
    """Custom G-Eval metric for review completeness"""
    
    def __init__(self, run_type: str = "standard", **kwargs):
        super().__init__(
            name=f"Code Review Completeness ({run_type})",
            criteria=(
                "Evaluate the completeness and quality of the code review. "
                "A good review should provide actionable suggestions, explain issues clearly, "
                "and prioritize them by severity."
            ),
            evaluation_steps=[
                "1. Check if the review provides clear explanations",
                "2. Verify actionable suggestions are provided",
                "3. Check proper severity categorization",
                "4. Ensure comprehensive coverage",
                "5. Verify specific locations are mentioned"
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            threshold=0.6
        )


def create_test_case(
    test_name: str,
    code: str,
    review_result: str,
    expected_issues: List[str],
    category: str,
    severity: str,
    run_type: str,
    model: str
) -> LLMTestCase:
    """Create a DeepEval test case with comprehensive metadata"""
    expected_output = (
        f"The code review should identify these {category} issues:\n" +
        "\n".join([f"- {issue}" for issue in expected_issues]) +
        "\n\nNote: Focus on whether the core problem is identified, not exact wording."
    )
    
    return LLMTestCase(
        input=f"Review this Python code for {category} issues:\n\n{code}",
        actual_output=review_result,
        expected_output=expected_output,
        additional_metadata={
            "test_name": test_name,
            "category": category,
            "severity": severity,
            "run_type": run_type,
            "model": model,
            "code_length": len(code),
            "expected_issue_count": len(expected_issues),
            "timestamp": datetime.now().isoformat()
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
    
    return review_text


def benchmark_model_configuration(
    provider: str,
    model: str,
    test_suites: Dict[str, Dict],
    run_type: str,
    enable_tools: bool,
    custom_prompt: Optional[str] = None
) -> Tuple[List[LLMTestCase], Dict[str, Any]]:
    """Benchmark a specific model configuration"""
    
    config_name = f"{provider}/{model} ({run_type})"
    print(f"\n📊 Benchmarking {config_name}...")
    
    # Initialize orchestrator with configuration
    orchestrator = CodeReviewOrchestrator(
        provider=provider,
        model=model,
        enable_tools=enable_tools
    )
    
    # Set custom prompt if provided
    if custom_prompt:
        base_prompt = orchestrator.agent.review_prompt_template
        # Modify the prompt to include the custom instruction
        modified_prompt = f"{custom_prompt}\n\n{base_prompt}"
        orchestrator.agent.review_prompt_template = modified_prompt
    
    # Create test cases
    test_cases = []
    total_time = 0
    issue_counts = []
    
    for suite_name, suite_tests in test_suites.items():
        print(f"\n  Testing {suite_name} suite:")
        
        for test_name, test_data in suite_tests.items():
            print(f"    - {test_name}...", end="", flush=True)
            
            start_time = time.time()
            result = orchestrator.review_code_string(test_data["code"], f"{test_name}.py")
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Format review result
            review_text = format_review_for_deepeval(result)
            
            # Create test case
            test_case = create_test_case(
                test_name=test_name,
                code=test_data["code"],
                review_result=review_text,
                expected_issues=test_data["expected_issues"],
                category=test_data["category"],
                severity=test_data["severity"],
                run_type=run_type,
                model=f"{provider}/{model}"
            )
            
            test_cases.append(test_case)
            issue_counts.append(len(result.issues))
            print(f" ✓ ({len(result.issues)} issues, {elapsed:.2f}s)")
    
    # Return test cases and metrics
    metrics = {
        "total_time": total_time,
        "avg_time": total_time / len(test_cases),
        "total_issues": sum(issue_counts),
        "avg_issues": sum(issue_counts) / len(issue_counts),
        "config_name": config_name
    }
    
    return test_cases, metrics


def evaluate_test_cases(
    test_cases: List[LLMTestCase],
    run_type: str
) -> Dict[str, Any]:
    """Evaluate test cases using DeepEval metrics"""
    
    print(f"\n  Running DeepEval metrics for {run_type}...")
    
    # Create metrics for this run type
    accuracy_metric = CodeReviewAccuracyMetric(run_type=run_type)
    completeness_metric = CodeReviewCompletenessMetric(run_type=run_type)
    
    # Run evaluation
    evaluation_result = evaluate(
        test_cases=test_cases,
        metrics=[accuracy_metric, completeness_metric],
        display_config=DisplayConfig(print_results=False, show_indicator=False)
    )
    
    # Process results
    accuracy_scores = []
    completeness_scores = []
    results_by_category = {}
    
    for i, test_result in enumerate(evaluation_result.test_results):
        test_case = test_cases[i]
        category = test_case.additional_metadata["category"]
        
        if category not in results_by_category:
            results_by_category[category] = {
                "accuracy": [],
                "completeness": []
            }
        
        # Get scores from metrics
        for metric_data in test_result.metrics_data:
            metric_name = metric_data.name
            score = metric_data.score
            
            if "Accuracy" in metric_name:
                accuracy_scores.append(score)
                results_by_category[category]["accuracy"].append(score)
            elif "Completeness" in metric_name:
                completeness_scores.append(score)
                results_by_category[category]["completeness"].append(score)
    
    # Calculate aggregates
    results = {
        "overall_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
        "overall_completeness": sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0,
        "by_category": {}
    }
    
    for category, scores in results_by_category.items():
        results["by_category"][category] = {
            "accuracy": sum(scores["accuracy"]) / len(scores["accuracy"]) if scores["accuracy"] else 0,
            "completeness": sum(scores["completeness"]) / len(scores["completeness"]) if scores["completeness"] else 0
        }
    
    return results


def create_dataset_from_results(all_test_cases: List[LLMTestCase], dataset_name: str):
    """Create a DeepEval dataset from test cases for future use"""
    
    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)
    
    # Save dataset as JSON file for dataset manager
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_path = f"datasets/{dataset_name}_{timestamp}.json"
    
    # Convert test cases to serializable format
    test_cases_data = []
    for test_case in all_test_cases:
        test_case_dict = {
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "additional_metadata": test_case.additional_metadata if hasattr(test_case, 'additional_metadata') else {}
        }
        test_cases_data.append(test_case_dict)
    
    # Create dataset structure
    dataset_data = {
        "name": dataset_name,
        "created": datetime.now().isoformat(),
        "test_cases": test_cases_data,
        "total_test_cases": len(test_cases_data)
    }
    
    # Save to file
    with open(dataset_path, 'w') as f:
        json.dump(dataset_data, f, indent=2)
    
    print(f"\n💾 Dataset saved to: {dataset_path}")
    print(f"📦 Contains {len(all_test_cases)} test cases")
    
    # Also create DeepEval dataset object for future use
    dataset = EvaluationDataset()
    for test_case in all_test_cases:
        dataset.add_test_case(test_case)
    
    # Note: DeepEval datasets can be pushed to Confident AI for sharing
    # dataset.push(alias=dataset_name)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Advanced DeepEval benchmarking for code review models")
    parser.add_argument("--models", nargs="+", help="Specific models to test (e.g., openai:gpt-4)")
    parser.add_argument("--run-types", nargs="+", default=["standard", "no-tools"],
                       choices=["standard", "no-tools", "security-focused", "performance-focused"],
                       help="Types of runs to perform")
    parser.add_argument("--test-suites", nargs="+", default=["security", "performance", "code_quality"],
                       help="Test suites to run")
    parser.add_argument("--save-dataset", action="store_true", help="Save results as DeepEval dataset")
    parser.add_argument("--dataset-name", default="code_review_benchmark", help="Name for saved dataset")
    
    args = parser.parse_args()
    
    print("=== Advanced DeepEval Model Benchmarking ===")
    print(f"Run types: {', '.join(args.run_types)}")
    print(f"Test suites: {', '.join(args.test_suites)}\n")
    
    # Determine models to test
    if args.models:
        models_to_test = []
        for model_spec in args.models:
            if ":" in model_spec:
                provider, model = model_spec.split(":", 1)
                models_to_test.append((provider, model))
            else:
                print(f"⚠️  Invalid model format: {model_spec} (use provider:model)")
    else:
        # Default models
        models_to_test = [
            ("openai", "gpt-4o-mini-2024-07-18"),
            ("openai", "gpt-4o-2024-08-06"),
        ]
        
        if os.getenv("ANTHROPIC_API_KEY"):
            models_to_test.extend([
                ("anthropic", "claude-3-5-haiku-20241022"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
            ])
    
    # Filter test suites
    selected_suites = {
        suite: tests for suite, tests in TEST_SUITES.items()
        if suite in args.test_suites
    }
    
    # Run configurations
    run_configs = {
        "standard": {
            "enable_tools": True,
            "custom_prompt": None
        },
        "no-tools": {
            "enable_tools": False,
            "custom_prompt": None
        },
        "security-focused": {
            "enable_tools": True,
            "custom_prompt": "SECURITY FOCUS: Pay special attention to security vulnerabilities, potential exploits, and unsafe practices. Prioritize security issues with critical or high severity."
        },
        "performance-focused": {
            "enable_tools": True,
            "custom_prompt": "PERFORMANCE FOCUS: Focus primarily on performance issues, algorithmic complexity, inefficient operations, and optimization opportunities."
        }
    }
    
    # Collect all results
    all_results = []
    all_test_cases = []
    
    for provider, model in models_to_test:
        model_results = {
            "provider": provider,
            "model": model,
            "run_types": {}
        }
        
        for run_type in args.run_types:
            if run_type not in run_configs:
                continue
                
            config = run_configs[run_type]
            
            try:
                # Benchmark this configuration
                test_cases, metrics = benchmark_model_configuration(
                    provider=provider,
                    model=model,
                    test_suites=selected_suites,
                    run_type=run_type,
                    **config
                )
                
                all_test_cases.extend(test_cases)
                
                # Evaluate
                eval_results = evaluate_test_cases(test_cases, run_type)
                
                # Store results
                model_results["run_types"][run_type] = {
                    "metrics": metrics,
                    "evaluation": eval_results
                }
                
            except Exception as e:
                print(f"\n❌ Failed {run_type} run for {provider}/{model}: {e}")
                continue
        
        all_results.append(model_results)
    
    # Display comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("="*80)
    
    # Overall comparison table
    print(f"\n{'Model':<30} {'Run Type':<15} {'Accuracy':<10} {'Completeness':<12} {'Avg Time':<10}")
    print("-"*77)
    
    for result in all_results:
        model_name = f"{result['provider']}/{result['model']}"
        for run_type, run_data in result["run_types"].items():
            eval_data = run_data["evaluation"]
            metrics = run_data["metrics"]
            
            print(f"{model_name:<30} {run_type:<15} "
                  f"{eval_data['overall_accuracy']:<10.2f} "
                  f"{eval_data['overall_completeness']:<12.2f} "
                  f"{metrics['avg_time']:<10.2f}s")
    
    # Category breakdown
    print("\n" + "="*80)
    print("RESULTS BY CATEGORY")
    print("="*80)
    
    for result in all_results:
        model_name = f"{result['provider']}/{result['model']}"
        print(f"\n{model_name}:")
        
        for run_type, run_data in result["run_types"].items():
            print(f"  {run_type}:")
            
            for category, scores in run_data["evaluation"]["by_category"].items():
                print(f"    {category}: Accuracy={scores['accuracy']:.2f}, "
                      f"Completeness={scores['completeness']:.2f}")
    
    # Save detailed results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"deepeval_advanced_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Create dataset if requested
    if args.save_dataset and all_test_cases:
        create_dataset_from_results(all_test_cases, args.dataset_name)
        print(f"📦 Dataset '{args.dataset_name}' created with {len(all_test_cases)} test cases")
    
    # Find best configurations
    print("\n🏆 Best Configurations:")
    
    # Best overall
    best_overall = None
    best_score = 0
    
    for result in all_results:
        if "run_types" in result:
            for run_type, run_data in result["run_types"].items():
                if "evaluation" in run_data and "overall_accuracy" in run_data["evaluation"]:
                    score = run_data["evaluation"]["overall_accuracy"]
                    if score > best_score:
                        best_score = score
                        best_overall = (result["provider"], result["model"], run_type)
    
    if best_overall:
        print(f"  Best Overall: {best_overall[0]}/{best_overall[1]} ({best_overall[2]}) - {best_score:.2f}")
    
    # Best by category
    categories = ["security", "performance", "quality"]
    for category in categories:
        best_cat = None
        best_cat_score = 0
        
        for result in all_results:
            if "run_types" in result:
                for run_type, run_data in result["run_types"].items():
                    if ("evaluation" in run_data and 
                        "by_category" in run_data["evaluation"] and 
                        category in run_data["evaluation"]["by_category"]):
                        score = run_data["evaluation"]["by_category"][category]["accuracy"]
                        if score > best_cat_score:
                            best_cat_score = score
                            best_cat = (result["provider"], result["model"], run_type)
        
        if best_cat:
            print(f"  Best for {category}: {best_cat[0]}/{best_cat[1]} ({best_cat[2]}) - {best_cat_score:.2f}")
    
    print("\n✨ Advanced benchmarking complete!")


if __name__ == "__main__":
    main()