#!/usr/bin/env python3
"""
Example 09: Model Benchmarking
This example benchmarks different models against each other for code review tasks.
"""

import os
import sys
import time
import json
import agentops
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

@dataclass
class ModelMetrics:
    """Metrics for evaluating model performance"""
    model: str
    provider: str
    total_issues_found: int
    issues_by_severity: Dict[str, int]
    response_time: float
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    accuracy_score: Optional[float] = None
    false_positive_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Test cases with known issues for evaluation
TEST_CASES = {
    "security_vulnerabilities": '''
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
    
    "performance_issues": '''
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

def process_large_list(data):
    # Multiple iterations when one would suffice
    filtered = []
    for item in data:
        if item > 0:
            filtered.append(item)
    
    squared = []
    for item in filtered:
        squared.append(item ** 2)
    
    total = 0
    for item in squared:
        total += item
    
    return total
''',
    
    "code_quality": '''
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
'''
}

# Expected issues for accuracy calculation
EXPECTED_ISSUES = {
    "security_vulnerabilities": {
        "critical": ["command injection", "sql injection", "insecure deserialization", "hardcoded secret"],
        "high": ["subprocess with shell=True", "pickle.load", "f-string in SQL"]
    },
    "performance_issues": {
        "high": ["exponential time complexity", "inefficient recursive", "O(n²) complexity"],
        "medium": ["multiple iterations", "unnecessary loops"]
    },
    "code_quality": {
        "medium": ["mutable default argument", "bare except", "== None"],
        "low": ["missing docstring", "poor naming", "no type hints", "class name", "method name"]
    }
}

def calculate_accuracy(found_issues: List[Any], test_name: str) -> Dict[str, float]:
    """Calculate accuracy metrics based on expected issues"""
    expected = EXPECTED_ISSUES.get(test_name, {})
    
    found_critical_high = 0
    expected_critical_high = len(expected.get("critical", [])) + len(expected.get("high", []))
    
    # Count found issues by checking message content
    for issue in found_issues:
        if issue.severity.value in ["critical", "high"]:
            found_critical_high += 1
    
    # Simple accuracy: ratio of critical/high issues found
    if expected_critical_high > 0:
        accuracy = min(1.0, found_critical_high / expected_critical_high)
    else:
        accuracy = 1.0 if found_critical_high == 0 else 0.0
    
    # False positive rate: issues found beyond expected
    false_positives = max(0, found_critical_high - expected_critical_high)
    fpr = false_positives / max(1, found_critical_high) if found_critical_high > 0 else 0.0
    
    return {"accuracy": accuracy, "false_positive_rate": fpr}

def benchmark_model(provider: str, model: str, test_cases: Dict[str, str]) -> ModelMetrics:
    """Benchmark a single model across all test cases"""
    print(f"\n📊 Benchmarking {provider}/{model}...")
    
    # Initialize AgentOps for this model
    session_tags = [f"{provider}", f"{model}", "benchmark"]
    agentops.start_trace(tags=session_tags)
    
    try:
        orchestrator = CodeReviewOrchestrator(provider=provider, model=model)
        
        total_issues = 0
        total_time = 0
        all_severities = {}
        accuracy_scores = []
        fpr_scores = []
        
        for test_name, code in test_cases.items():
            print(f"  Testing {test_name}...", end="", flush=True)
            
            start_time = time.time()
            result = orchestrator.review_code_string(code, f"{test_name}.py")
            elapsed = time.time() - start_time
            
            total_time += elapsed
            total_issues += len(result.issues)
            
            # Count severities
            for issue in result.issues:
                sev = issue.severity.value
                all_severities[sev] = all_severities.get(sev, 0) + 1
            
            # Calculate accuracy
            metrics = calculate_accuracy(result.issues, test_name)
            accuracy_scores.append(metrics["accuracy"])
            fpr_scores.append(metrics["false_positive_rate"])
            
            print(f" ✓ ({len(result.issues)} issues, {elapsed:.2f}s)")
        
        # Calculate averages
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        avg_fpr = sum(fpr_scores) / len(fpr_scores) if fpr_scores else 0
        
        metrics = ModelMetrics(
            model=model,
            provider=provider,
            total_issues_found=total_issues,
            issues_by_severity=all_severities,
            response_time=total_time,
            accuracy_score=avg_accuracy,
            false_positive_rate=avg_fpr
        )
        
        # End AgentOps trace
        agentops.end_trace(end_state="Success")
        
        return metrics
        
    except Exception as e:
        print(f" ❌ Error: {e}")
        agentops.end_trace(end_state="Error")
        raise

def main():
    print("=== Model Benchmarking Suite ===")
    print("This will test multiple models on standardized code review tasks.\n")
    
    # Initialize AgentOps
    AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") or 'a0aac5f4-2b60-43c2-bb9e-f79247f5a2dc'
    agentops.init(
        api_key=AGENTOPS_API_KEY,
        default_tags=['model_benchmark']
    )
    
    # Models to benchmark
    models_to_test = [
        ("openai", "gpt-4o-2024-08-06"),
        ("openai", "gpt-4.1-2025-04-14"),
        ("openai", "gpt-4.1-mini-2025-04-14"),
        ("openai", "gpt-4.1-nano-2025-04-14"),
        ("openai", "gpt-4o-mini-2024-07-18"),
        ("openai", "o3-2025-04-16"),
        ("openai", "o4-mini-2025-04-16"),
        ("openai", "o3-mini-2025-01-31"),
    ]
    
    # Add Anthropic models if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.extend([
            ("anthropic", "claude-opus-4-20250514"),
            ("anthropic", "claude-sonnet-4-20250514"),
            ("anthropic", "claude-3-7-sonnet-20250219"),
            ("anthropic", "claude-3-5-haiku-20241022 "),
        ])
    
    # Check for o3-pro access (usually requires special access)
    if os.getenv("OPENAI_O3_PRO_ACCESS"):
        models_to_test.append(("openai", "o3-pro-2025-06-10"))
    
    results = []
    
    for provider, model in models_to_test:
        try:
            metrics = benchmark_model(provider, model, TEST_CASES)
            results.append(metrics)
        except Exception as e:
            print(f"Failed to benchmark {provider}/{model}: {e}")
    
    # Display results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Sort by accuracy
    results.sort(key=lambda x: x.accuracy_score or 0, reverse=True)
    
    print(f"\n{'Model':<30} {'Issues':<10} {'Time(s)':<10} {'Accuracy':<10} {'FPR':<10}")
    print("-"*70)
    
    for metric in results:
        model_name = f"{metric.provider}/{metric.model}"
        print(f"{model_name:<30} {metric.total_issues_found:<10} "
              f"{metric.response_time:<10.2f} {metric.accuracy_score or 0:<10.2%} "
              f"{metric.false_positive_rate or 0:<10.2%}")
    
    # Detailed breakdown
    print("\n📊 Detailed Issue Breakdown:")
    for metric in results:
        print(f"\n{metric.provider}/{metric.model}:")
        for sev, count in sorted(metric.issues_by_severity.items()):
            print(f"  {sev}: {count}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump([m.to_dict() for m in results], f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # AgentOps insights
    print("\n🔍 AgentOps Insights:")
    print("   View detailed traces at: https://app.agentops.ai")
    print("   - Token usage per model")
    print("   - Cost analysis")
    print("   - Response time distribution")
    print("   - Error rates and types")
    
    print("\n✨ Benchmark complete!")

if __name__ == "__main__":
    main()