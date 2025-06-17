"""
Advanced benchmarking framework for code review models with AgentOps integration.
"""

import os
import time
import json
import agentops
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from ..core import CodeReviewOrchestrator
from ..models import CodeIssue


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model"""
    model: str
    provider: str
    test_results: List['TestResult'] = field(default_factory=list)
    total_time: float = 0.0
    total_issues: int = 0
    avg_issues_per_test: float = 0.0
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    agentops_session_id: Optional[str] = None
    
    def calculate_summary(self):
        """Calculate summary statistics"""
        if self.test_results:
            self.total_time = sum(r.execution_time for r in self.test_results)
            self.total_issues = sum(r.issues_found for r in self.test_results)
            self.avg_issues_per_test = self.total_issues / len(self.test_results)
            
            # Aggregate severity distribution
            for result in self.test_results:
                for sev, count in result.severity_counts.items():
                    self.severity_distribution[sev] = self.severity_distribution.get(sev, 0) + count
            
            # Calculate accuracy metrics
            precision_scores = [r.precision for r in self.test_results if r.precision is not None]
            recall_scores = [r.recall for r in self.test_results if r.recall is not None]
            f1_scores = [r.f1_score for r in self.test_results if r.f1_score is not None]
            
            if precision_scores:
                self.accuracy_metrics['avg_precision'] = sum(precision_scores) / len(precision_scores)
            if recall_scores:
                self.accuracy_metrics['avg_recall'] = sum(recall_scores) / len(recall_scores)
            if f1_scores:
                self.accuracy_metrics['avg_f1'] = sum(f1_scores) / len(f1_scores)


@dataclass
class TestResult:
    """Result for a single test case"""
    test_name: str
    issues_found: int
    execution_time: float
    severity_counts: Dict[str, int] = field(default_factory=dict)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)


class CodeReviewBenchmark:
    """Comprehensive benchmarking system for code review models"""
    
    def __init__(self, ground_truth_path: Optional[str] = None):
        self.ground_truth = {}
        if ground_truth_path and os.path.exists(ground_truth_path):
            with open(ground_truth_path, 'r') as f:
                self.ground_truth = json.load(f)
        
        self.test_suites = {
            "security": self._get_security_tests(),
            "performance": self._get_performance_tests(),
            "quality": self._get_quality_tests(),
            "edge_cases": self._get_edge_case_tests()
        }
    
    def _get_security_tests(self) -> Dict[str, Tuple[str, List[Dict]]]:
        """Security-focused test cases with expected issues"""
        return {
            "sql_injection": (
                '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)
                ''',
                [
                    {"type": "sql_injection", "severity": "critical", "line": 2}
                ]
            ),
            "command_injection": (
                '''
import subprocess
def run_command(user_input):
    subprocess.call(user_input, shell=True)
                ''',
                [
                    {"type": "command_injection", "severity": "critical", "line": 3}
                ]
            ),
            "hardcoded_secrets": (
                '''
API_KEY = "sk-1234567890abcdef"
PASSWORD = "admin123"
                ''',
                [
                    {"type": "hardcoded_secret", "severity": "critical", "line": 1},
                    {"type": "hardcoded_secret", "severity": "critical", "line": 2}
                ]
            )
        }
    
    def _get_performance_tests(self) -> Dict[str, Tuple[str, List[Dict]]]:
        """Performance-focused test cases"""
        return {
            "inefficient_recursion": (
                '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
                ''',
                [
                    {"type": "exponential_complexity", "severity": "high", "line": 1}
                ]
            ),
            "nested_loops": (
                '''
def find_pairs(list1, list2):
    pairs = []
    for i in list1:
        for j in list2:
            if i + j == 10:
                pairs.append((i, j))
    return pairs
                ''',
                [
                    {"type": "quadratic_complexity", "severity": "medium", "line": 3}
                ]
            )
        }
    
    def _get_quality_tests(self) -> Dict[str, Tuple[str, List[Dict]]]:
        """Code quality test cases"""
        return {
            "poor_practices": (
                '''
def process(d=[]):
    if d == None:
        d = []
    try:
        return sum(d) / len(d)
    except:
        return 0
                ''',
                [
                    {"type": "mutable_default", "severity": "high", "line": 1},
                    {"type": "incorrect_none_check", "severity": "medium", "line": 2},
                    {"type": "bare_except", "severity": "medium", "line": 6}
                ]
            )
        }
    
    def _get_edge_case_tests(self) -> Dict[str, Tuple[str, List[Dict]]]:
        """Edge cases and complex scenarios"""
        return {
            "empty_function": (
                '''
def do_nothing():
    pass
                ''',
                []  # No issues expected
            ),
            "complex_logic": (
                '''
def complex_function(data):
    if not data:
        return None
    
    result = {}
    for item in data:
        if isinstance(item, dict) and 'id' in item:
            if item['id'] not in result:
                result[item['id']] = []
            result[item['id']].append(item)
    
    return result
                ''',
                []  # Complex but correct code
            )
        }
    
    def benchmark_model(self, provider: str, model: str, 
                       test_suites: Optional[List[str]] = None) -> BenchmarkResult:
        """Benchmark a single model across test suites"""
        if test_suites is None:
            test_suites = list(self.test_suites.keys())
        
        # Start AgentOps trace
        trace = agentops.start_trace(
            tags=[provider, model, "benchmark"]
        )
        
        result = BenchmarkResult(model=model, provider=provider)
        orchestrator = CodeReviewOrchestrator(provider=provider, model=model)
        
        try:
            for suite_name in test_suites:
                if suite_name not in self.test_suites:
                    continue
                
                print(f"\n  Running {suite_name} tests...")
                suite_tests = self.test_suites[suite_name]
                
                for test_name, (code, expected_issues) in suite_tests.items():
                    start_time = time.time()
                    
                    # Review the code
                    review_result = orchestrator.review_code_string(
                        code, f"{suite_name}_{test_name}.py"
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Create test result
                    test_result = TestResult(
                        test_name=f"{suite_name}/{test_name}",
                        issues_found=len(review_result.issues),
                        execution_time=execution_time
                    )
                    
                    # Count severities
                    for issue in review_result.issues:
                        sev = issue.severity.value
                        test_result.severity_counts[sev] = test_result.severity_counts.get(sev, 0) + 1
                    
                    # Calculate accuracy if we have ground truth
                    if expected_issues:
                        metrics = self._calculate_accuracy(review_result.issues, expected_issues)
                        test_result.precision = metrics['precision']
                        test_result.recall = metrics['recall']
                        test_result.f1_score = metrics['f1_score']
                        test_result.false_positives = metrics['false_positives']
                        test_result.false_negatives = metrics['false_negatives']
                    
                    result.test_results.append(test_result)
                    print(f"    ✓ {test_name}: {len(review_result.issues)} issues in {execution_time:.2f}s")
            
            # Calculate summary statistics
            result.calculate_summary()
            
            # Store AgentOps session ID if available
            if hasattr(trace, 'session_id'):
                result.agentops_session_id = trace.session_id
            
            agentops.end_trace(end_state="Success")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            agentops.end_trace(end_state="Error")
            raise
        
        return result
    
    def _calculate_accuracy(self, found_issues: List[CodeIssue], 
                           expected_issues: List[Dict]) -> Dict[str, Any]:
        """Calculate precision, recall, and F1 score"""
        # Simple matching based on line numbers and severity
        found_lines_severity = {
            (issue.line_number, issue.severity.value) 
            for issue in found_issues 
            if issue.line_number
        }
        
        expected_lines_severity = {
            (int(issue.get('line', 0)), str(issue.get('severity', ''))) 
            for issue in expected_issues
            if issue.get('line') is not None
        }
        
        true_positives = len(found_lines_severity & expected_lines_severity)
        false_positives = len(found_lines_severity - expected_lines_severity)
        false_negatives = len(expected_lines_severity - found_lines_severity)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positives': [str(fp) for fp in (found_lines_severity - expected_lines_severity)],
            'false_negatives': [str(fn) for fn in (expected_lines_severity - found_lines_severity)]
        }
    
    def run_full_benchmark(self, models: List[Tuple[str, str]]) -> List[BenchmarkResult]:
        """Run benchmark on multiple models"""
        results = []
        
        print("🏁 Starting Code Review Model Benchmark")
        print("="*60)
        
        for provider, model in models:
            print(f"\n📊 Benchmarking {provider}/{model}")
            try:
                result = self.benchmark_model(provider, model)
                results.append(result)
            except Exception as e:
                print(f"   Failed: {e}")
        
        return results
    
    def generate_report(self, results: List[BenchmarkResult], output_path: Optional[str] = None):
        """Generate comprehensive benchmark report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': len(results),
            'results': []
        }
        
        # Sort by F1 score
        results.sort(
            key=lambda r: r.accuracy_metrics.get('avg_f1', 0), 
            reverse=True
        )
        
        print("\n" + "="*80)
        print("BENCHMARK REPORT")
        print("="*80)
        
        # Summary table
        print(f"\n{'Model':<35} {'Tests':<8} {'Time(s)':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-"*80)
        
        for result in results:
            model_name = f"{result.provider}/{result.model}"
            precision = result.accuracy_metrics.get('avg_precision', 0)
            recall = result.accuracy_metrics.get('avg_recall', 0)
            f1 = result.accuracy_metrics.get('avg_f1', 0)
            
            print(f"{model_name:<35} {len(result.test_results):<8} "
                  f"{result.total_time:<10.2f} {precision:<10.2%} "
                  f"{recall:<10.2%} {f1:<10.2%}")
            
            # Add to report
            report['results'].append({
                'model': result.model,
                'provider': result.provider,
                'metrics': result.accuracy_metrics,
                'total_time': result.total_time,
                'total_issues': result.total_issues,
                'severity_distribution': result.severity_distribution,
                'agentops_session_id': result.agentops_session_id
            })
        
        # Save detailed report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Detailed report saved to: {output_path}")
        
        # AgentOps integration info
        print("\n🔍 AgentOps Analytics:")
        print("   Visit https://app.agentops.ai to view:")
        print("   - Token usage and costs per model")
        print("   - Detailed error analysis")
        print("   - Response time distributions")
        print("   - Model-specific performance patterns")
        
        return report


# Convenience function for quick benchmarking
def quick_benchmark(models: Optional[List[Tuple[str, str]]] = None):
    """Quick benchmark with default models"""
    if models is None:
        models = [
            ("openai", "gpt-4"),
            ("openai", "gpt-3.5-turbo"),
        ]
        
        # Add Anthropic if available
        if os.getenv("ANTHROPIC_API_KEY"):
            models.append(("anthropic", "claude-3-5-sonnet-20241022"))
    
    benchmark = CodeReviewBenchmark()
    results = benchmark.run_full_benchmark(models)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"benchmark_report_{timestamp}.json")
    
    benchmark.generate_report(results, output_path)
    
    return results