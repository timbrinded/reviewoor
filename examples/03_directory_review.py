#!/usr/bin/env python3
"""
Example 03: Directory Review
This example shows how to review all Python files in a directory.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

def main():
    print("=== Directory Review Example ===\n")
    
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file")
        return
    
    # Determine which provider to use
    provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    
    # Initialize the orchestrator
    orchestrator = CodeReviewOrchestrator(provider=provider)
    
    # Review the src directory
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    
    print(f"Reviewing all Python files in: {src_dir}")
    print("This may take several minutes...\n")
    
    # Perform the review
    results = orchestrator.review_directory(src_dir, pattern="*.py")
    
    print(f"\n✅ Reviewed {len(results)} files")
    
    # Display summary for each file
    total_issues = 0
    critical_files = []
    
    print("\nFile Summary:")
    print("-" * 60)
    
    for result in results:
        issue_count = len(result.issues)
        total_issues += issue_count
        
        # Count issues by severity
        severity_counts = {}
        for issue in result.issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Check if file has critical issues
        if severity_counts.get("critical", 0) > 0:
            critical_files.append((result.file_path, severity_counts["critical"]))
        
        # Display file info
        file_name = os.path.basename(result.file_path)
        print(f"{file_name:<30} {issue_count:>3} issues", end="")
        
        if severity_counts:
            severity_str = ", ".join([f"{count} {sev}" for sev, count in severity_counts.items()])
            print(f" ({severity_str})")
        else:
            print()
    
    print("-" * 60)
    print(f"Total: {total_issues} issues across {len(results)} files\n")
    
    # Highlight critical files
    if critical_files:
        print("⚠️  Files with CRITICAL issues:")
        for file_path, count in critical_files:
            print(f"  - {os.path.basename(file_path)} ({count} critical issues)")
    
    # Generate and display summary report
    report = orchestrator.get_summary_report()
    
    print("\n📊 Overall Statistics:")
    print(f"  - Files reviewed: {report['total_files_reviewed']}")
    print(f"  - Total issues: {report['total_issues']}")
    print(f"  - Review time: {report['total_review_time']:.2f} seconds")
    
    print("\n📈 Issues by Severity:")
    for severity, count in report['issues_by_severity'].items():
        print(f"  - {severity}: {count}")
    
    print("\n📂 Issues by Category:")
    for category, count in sorted(report['issues_by_category'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {category}: {count}")
    
    if report['average_metrics']:
        print("\n⭐ Average Code Metrics:")
        for metric, score in report['average_metrics'].items():
            print(f"  - {metric}: {score:.1f}/10")
    
    # Export results
    output_file = "directory_review_results.json"
    orchestrator.export_results(output_file)
    print(f"\n💾 Full results exported to: {output_file}")

if __name__ == "__main__":
    main()