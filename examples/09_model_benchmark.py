#!/usr/bin/env python3
"""
Example 09: Model Benchmarking
This example benchmarks different models against each other for code review tasks.
"""

import os
import sys
import time
import json
import logging
import agentops
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Suppress INFO logs from httpx
logging.getLogger("httpx").setLevel(logging.WARN)

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
    # Tool usage metrics
    tools_used: int = 0
    tool_calls: Optional[List[str]] = None
    tool_accuracy: Optional[float] = None
    tool_response_time: Optional[float] = None
    
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
    
    "complex_imports": '''
import os
import sys
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt

# Some imports are unused
def process_data(data: List[int]) -> int:
    # Only uses os and json
    filepath = os.path.join("data", "output.json")
    with open(filepath, 'w') as f:
        json.dump({"result": sum(data)}, f)
    return len(data)
''',
    
    "high_complexity": '''
def complex_business_logic(user_data, config, permissions):
    # Very high cyclomatic complexity
    result = {}
    
    if user_data.get('type') == 'premium':
        if config.get('enable_premium'):
            if permissions.get('read'):
                if permissions.get('write'):
                    if user_data.get('status') == 'active':
                        if config.get('region') == 'US':
                            result['access'] = 'full'
                        elif config.get('region') == 'EU':
                            if user_data.get('gdpr_consent'):
                                result['access'] = 'full'
                            else:
                                result['access'] = 'limited'
                        else:
                            result['access'] = 'restricted'
                    else:
                        result['access'] = 'suspended'
                else:
                    result['access'] = 'readonly'
            else:
                result['access'] = 'none'
        else:
            result['access'] = 'disabled'
    elif user_data.get('type') == 'basic':
        if config.get('enable_basic'):
            if permissions.get('read'):
                result['access'] = 'readonly'
            else:
                result['access'] = 'none'
        else:
            result['access'] = 'disabled'
    else:
        result['access'] = 'unknown'
    
    # More nested logic
    for feature in config.get('features', []):
        if feature in permissions.get('allowed_features', []):
            if user_data.get(f'{feature}_enabled'):
                if result.get('features') is None:
                    result['features'] = []
                result['features'].append(feature)
    
    return result
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
    },
    "complex_imports": {
        "medium": ["unused imports"],
        "low": ["too many imports"]
    },
    "high_complexity": {
        "high": ["cyclomatic complexity", "deeply nested"],
        "medium": ["too many branches", "complex logic"]
    }
}

# Expected tool usage for each test case
EXPECTED_TOOLS = {
    "security_vulnerabilities": ["find_security_patterns"],
    "performance_issues": ["analyze_complexity", "get_function_metrics"],
    "code_quality": ["detect_code_smells", "check_type_consistency"],
    "complex_imports": ["check_imports"],
    "high_complexity": ["analyze_complexity", "get_function_metrics"]
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

def calculate_tool_metrics(tool_calls: List[Dict[str, Any]], test_name: str) -> Dict[str, Any]:
    """Calculate tool usage metrics"""
    expected_tools = EXPECTED_TOOLS.get(test_name, [])
    
    # Extract tool names from calls
    tools_used = [call["tool"] for call in tool_calls]
    unique_tools = list(set(tools_used))
    
    # Calculate tool accuracy (how many expected tools were used)
    if expected_tools:
        tools_hit = sum(1 for tool in expected_tools if tool in unique_tools)
        tool_accuracy = tools_hit / len(expected_tools)
    else:
        tool_accuracy = 1.0 if not tools_used else 0.0
    
    return {
        "tools_used": len(tool_calls),
        "unique_tools": unique_tools,
        "tool_accuracy": tool_accuracy,
        "expected_tools_used": sum(1 for t in expected_tools if t in unique_tools)
    }

def benchmark_model(provider: str, model: str, test_cases: Dict[str, str], enable_tools: bool = True) -> ModelMetrics:
    """Benchmark a single model across all test cases"""
    print(f"\n📊 Benchmarking {provider}/{model} (tools={'enabled' if enable_tools else 'disabled'})...")
    
    # Initialize AgentOps for this model
    session_tags = [f"{provider}", f"{model}", "benchmark"]
    if enable_tools:
        session_tags.append("with_tools")
    agentops.start_trace(tags=session_tags)
    
    try:
        orchestrator = CodeReviewOrchestrator(provider=provider, model=model, enable_tools=enable_tools)
        
        total_issues = 0
        total_time = 0
        all_severities = {}
        accuracy_scores = []
        fpr_scores = []
        all_tool_calls = []
        tool_accuracy_scores = []
        
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
            
            # Track tool usage
            if hasattr(result, 'tool_calls') and result.tool_calls:
                all_tool_calls.extend(result.tool_calls)
                tool_metrics = calculate_tool_metrics(result.tool_calls, test_name)
                tool_accuracy_scores.append(tool_metrics["tool_accuracy"])
                print(f" ✓ ({len(result.issues)} issues, {elapsed:.2f}s, {len(result.tool_calls)} tool calls)")
            else:
                print(f" ✓ ({len(result.issues)} issues, {elapsed:.2f}s)")
        
        # Calculate averages
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        avg_fpr = sum(fpr_scores) / len(fpr_scores) if fpr_scores else 0
        
        # Tool metrics
        unique_tools = list(set(call["tool"] for call in all_tool_calls))
        avg_tool_accuracy = sum(tool_accuracy_scores) / len(tool_accuracy_scores) if tool_accuracy_scores else 0
        
        metrics = ModelMetrics(
            model=model,
            provider=provider,
            total_issues_found=total_issues,
            issues_by_severity=all_severities,
            response_time=total_time,
            accuracy_score=avg_accuracy,
            false_positive_rate=avg_fpr,
            tools_used=len(all_tool_calls),
            tool_calls=unique_tools,
            tool_accuracy=avg_tool_accuracy if enable_tools else None
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
        # ("openai", "o3-2025-04-16"),
        ("openai", "o4-mini-2025-04-16"),
        ("openai", "o3-mini-2025-01-31"),
    ]
    
    # Add Anthropic models if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.extend([
            # ("anthropic", "claude-opus-4-20250514"),
            ("anthropic", "claude-sonnet-4-20250514"),
            ("anthropic", "claude-3-7-sonnet-20250219"),
            ("anthropic", "claude-3-5-haiku-20241022 "),
        ])
    
    # Check for o3-pro access (usually requires special access)
    # if os.getenv("OPENAI_O3_PRO_ACCESS"):
    #     models_to_test.append(("openai", "o3-pro-2025-06-10"))
    
    results = []
    
    for provider, model in models_to_test:
        try:
            # Benchmark with tools enabled
            metrics = benchmark_model(provider, model, TEST_CASES, enable_tools=True)
            results.append(metrics)
            
            # Also benchmark without tools for comparison (for select models)
            if model in ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022"]:
                metrics_no_tools = benchmark_model(provider, model, TEST_CASES, enable_tools=False)
                metrics_no_tools.model = f"{model}-no-tools"
                results.append(metrics_no_tools)
        except Exception as e:
            print(f"Failed to benchmark {provider}/{model}: {e}")
    
    # Display results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Sort by accuracy
    results.sort(key=lambda x: x.accuracy_score or 0, reverse=True)
    
    print(f"\n{'Model':<30} {'Issues':<10} {'Time(s)':<10} {'Accuracy':<10} {'FPR':<10} {'Tools':<10} {'Tool Acc':<10}")
    print("-"*90)
    
    for metric in results:
        model_name = f"{metric.provider}/{metric.model}"
        tool_info = f"{metric.tools_used}" if metric.tools_used > 0 else "-"
        tool_acc = f"{metric.tool_accuracy:.0%}" if metric.tool_accuracy is not None else "-"
        
        print(f"{model_name:<30} {metric.total_issues_found:<10} "
              f"{metric.response_time:<10.2f} {metric.accuracy_score or 0:<10.2%} "
              f"{metric.false_positive_rate or 0:<10.2%} {tool_info:<10} {tool_acc:<10}")
    
    # Detailed breakdown
    print("\n📊 Detailed Issue Breakdown:")
    for metric in results:
        print(f"\n{metric.provider}/{metric.model}:")
        for sev, count in sorted(metric.issues_by_severity.items()):
            print(f"  {sev}: {count}")
        if metric.tool_calls:
            print(f"  Tools used: {', '.join(metric.tool_calls)}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump([m.to_dict() for m in results], f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Tool usage comparison
    print("\n🔧 Tool Usage Analysis:")
    tools_enabled = [m for m in results if m.tools_used > 0]
    if tools_enabled:
        avg_tool_accuracy = sum(m.tool_accuracy for m in tools_enabled if m.tool_accuracy) / len(tools_enabled)
        print(f"   Average tool accuracy: {avg_tool_accuracy:.0%}")
        print(f"   Models using tools: {len(tools_enabled)}")
        
        # Compare with/without tools
        for base_model in ["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022"]:
            with_tools = next((m for m in results if m.model == base_model), None)
            without_tools = next((m for m in results if m.model == f"{base_model}-no-tools"), None)
            
            if with_tools and without_tools:
                print(f"\n   {base_model} comparison:")
                print(f"     With tools: {with_tools.accuracy_score:.0%} accuracy, {with_tools.total_issues_found} issues")
                print(f"     Without tools: {without_tools.accuracy_score:.0%} accuracy, {without_tools.total_issues_found} issues")
                print(f"     Improvement: {(with_tools.accuracy_score - without_tools.accuracy_score):.0%}")
    
    # AgentOps insights
    print("\n🔍 AgentOps Insights:")
    print("   View detailed traces at: https://app.agentops.ai")
    print("   - Token usage per model")
    print("   - Cost analysis")
    print("   - Response time distribution")
    print("   - Error rates and types")
    print("   - Tool usage patterns")
    
    print("\n✨ Benchmark complete!")

if __name__ == "__main__":
    main()