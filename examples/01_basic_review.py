#!/usr/bin/env python3
"""
Example 01: Basic Code Review
This example shows the simplest way to review Python code using the agent.
"""

import os
import agentops
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

# Sample code with various issues
sample_code = '''
def calculate_stats(numbers=[]):
    """Calculate basic statistics"""
    total = 0
    count = 0
    
    for num in numbers:
        total = total + num  # Could use +=
        count = count + 1
    
    average = total / count  # Bug: No check for empty list
    
    return {
        "total": total,
        "count": count,
        "average": average
    }

def process_user_input():
    user_data = input("Enter data: ")
    result = eval(user_data)  # Security issue!
    return result
'''

def main():
    print("=== Basic Code Review Example ===\n")
    
    AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") or 'a0aac5f4-2b60-43c2-bb9e-f79247f5a2dc'
    agentops.init(
    api_key=AGENTOPS_API_KEY,
    default_tags=['openai assistants']
)

    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file")
        print("   Falling back to static analysis only...\n")
        
        # Use static analyzer only
        from src import CodeAnalyzer
        analyzer = CodeAnalyzer()
        issues = analyzer.analyze(sample_code)
        
        print(f"Static analysis found {len(issues)} issues:\n")
        for issue in issues:
            print(f"  Line {issue.line_number}: [{issue.severity.value}] {issue.message}")
        return
    
    # Initialize the orchestrator (defaults to OpenAI)
    orchestrator = CodeReviewOrchestrator()
    print(f"Using {orchestrator.provider} ({orchestrator.model}) for AI-powered review\n")
    
    # Review the code
    print("Reviewing code...")
    result = orchestrator.review_code_string(sample_code, "sample.py")
    
    # Display results
    print(f"\n✅ Review completed in {result.review_time:.2f} seconds")
    print(f"📊 Found {len(result.issues)} total issues\n")
    
    # Group issues by severity
    by_severity = {}
    for issue in result.issues:
        severity = issue.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(issue)
    
    # Display issues by severity
    for severity in ["critical", "high", "medium", "low", "info"]:
        if severity in by_severity:
            print(f"\n{severity.upper()} ({len(by_severity[severity])} issues):")
            for issue in by_severity[severity]:
                print(f"  Line {issue.line_number}: {issue.message}")
                if issue.suggestion:
                    print(f"    → {issue.suggestion}")
    
    # Display summary and metrics
    print(f"\n📝 Summary: {result.summary}")
    if result.metrics:
        print("\n📈 Code Metrics:")
        for metric, score in result.metrics.items():
            print(f"  - {metric}: {score}/10")

if __name__ == "__main__":
    main()