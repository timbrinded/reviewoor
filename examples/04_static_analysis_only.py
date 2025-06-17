#!/usr/bin/env python3
"""
Example 04: Static Analysis Only
This example shows how to use just the static analyzer without AI.
Perfect for quick checks or when you don't have API keys.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeAnalyzer

# Sample code with various static analysis issues
problematic_code = '''
import os
import sys

def process_config(config_data = {}):
    """Process configuration data"""
    password = "admin123"  # Hardcoded password
    api_key = "sk-1234567890abcdef"  # Hardcoded API key
    
    if config_data == None:  # Should use 'is None'
        config_data = {}
    
    # Very long line that exceeds the recommended character limit and makes the code harder to read and maintain in any editor
    
    return config_data

class DataProcessor:
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        # Missing docstring
        self.items.append(item)
    
    def process_all(self):
        try:
            for i in range(len(self.items)):
                item = self.items[i]
                processed = eval(f"item.{item.operation}()")  # Dangerous eval!
                print(processed)
        except:  # Bare except clause
            pass
    
    def clear_items(self):
        """Clear all items"""
        self.items = []   # Trailing whitespace

def risky_function(user_input):
    """Function with security issues"""
    # SQL injection risk
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    
    # Command injection risk
    os.system(f"echo {user_input}")
    
    # Using exec
    exec(f"result = {user_input}")
    
    return query

# Function with too many lines
def very_long_function(data):
    """This function is way too long and should be refactored"""
    result = []
    
    # Imagine 50+ lines of code here...
    for i in range(100):
        if i % 2 == 0:
            result.append(i)
        else:
            result.append(i * 2)
    
    # More processing...
    filtered = []
    for item in result:
        if item > 50:
            filtered.append(item)
    
    # Even more processing...
    final = []
    for item in filtered:
        final.append(item ** 2)
    
    return final
'''

def main():
    print("=== Static Analysis Only Example ===\n")
    print("This example runs without API keys, using only static analysis.\n")
    
    # Initialize the static analyzer
    analyzer = CodeAnalyzer()
    
    # Analyze the code
    print("Analyzing code for issues...")
    issues = analyzer.analyze(problematic_code)
    
    print(f"\n✅ Analysis complete! Found {len(issues)} issues:\n")
    
    # Group issues by category
    by_category = {}
    for issue in issues:
        category = issue.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(issue)
    
    # Display issues by category
    for category, category_issues in by_category.items():
        print(f"\n{category} ({len(category_issues)} issues):")
        print("-" * 50)
        
        for issue in category_issues:
            print(f"Line {issue.line_number}: [{issue.severity.value}] {issue.message}")
            if issue.suggestion:
                print(f"  → Suggestion: {issue.suggestion}")
    
    # Summary statistics
    severity_counts = {}
    for issue in issues:
        sev = issue.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    print("\n📊 Summary by Severity:")
    for severity in ["critical", "high", "medium", "low", "info"]:
        count = severity_counts.get(severity, 0)
        if count > 0:
            bar = "█" * count
            print(f"  {severity:<8} [{count:>2}] {bar}")
    
    print("\n💡 Static analysis is great for:")
    print("  - Quick checks without API calls")
    print("  - CI/CD pipeline integration")
    print("  - Finding common Python anti-patterns")
    print("  - Security issue detection")
    print("  - Style and formatting issues")

if __name__ == "__main__":
    main()