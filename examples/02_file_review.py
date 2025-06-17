#!/usr/bin/env python3
"""
Example 02: File Review
This example shows how to review existing Python files.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

def main():
    print("=== File Review Example ===\n")
    
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file")
        return
    
    # Determine which provider to use
    provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    
    # Initialize the orchestrator
    orchestrator = CodeReviewOrchestrator(provider=provider)
    
    # Review a specific file (let's review the example.py file)
    file_to_review = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example.py")
    
    if not os.path.exists(file_to_review):
        print(f"File not found: {file_to_review}")
        return
    
    print(f"Reviewing file: {file_to_review}")
    print("This may take a moment...\n")
    
    # Perform the review
    result = orchestrator.review_file(file_to_review)
    
    # Display results
    print(f"✅ Review completed in {result.review_time:.2f} seconds")
    print(f"📊 Found {len(result.issues)} issues in {result.file_path}\n")
    
    # Show top 10 issues
    print("Top issues found:")
    for i, issue in enumerate(result.issues[:10], 1):
        print(f"\n{i}. [{issue.severity.value.upper()}] {issue.category}")
        print(f"   Line {issue.line_number}: {issue.message}")
        if issue.suggestion:
            print(f"   Suggestion: {issue.suggestion}")
    
    if len(result.issues) > 10:
        print(f"\n... and {len(result.issues) - 10} more issues")
    
    # Export results to JSON
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "file_review_results.json")
    orchestrator.export_results(output_file)
    print(f"\n💾 Full results exported to: {output_file}")

if __name__ == "__main__":
    main()