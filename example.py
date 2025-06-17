"""Example usage of the refactored AgentOps code review system."""

from src import CodeReviewOrchestrator, CodeAnalyzer

# Example problematic code for testing
test_code = '''
def process_data(data=[]):
    """Process some data"""
    password = "admin123"  # Bad: hardcoded password
    
    if data == None:  # Should use 'is None'
        data = []
    
    try:
        result = eval(input("Enter expression: "))  # Dangerous!
    except:  # Bare except
        pass
    
    # Very long line that exceeds the recommended character limit and makes the code harder to read and maintain properly
    
    for i in range(len(data)):  # Could use enumerate
        item = data[i]
        # Missing error handling
        processed = item.upper()
    
    return result  # Bug: result might not be defined


class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        # Missing docstring
        return item.strip().lower()


def calculate_metrics(values):
    total = 0
    for v in values:
        total = total + v  # Could use +=
    
    average = total / len(values)  # Bug: no check for empty list
    return average
'''


if __name__ == "__main__":
    # Example 1: Using static analyzer only
    print("=== Static Analysis Demo ===")
    analyzer = CodeAnalyzer()
    static_issues = analyzer.analyze(test_code)
    
    print(f"\nStatic analysis found {len(static_issues)} issues:")
    for issue in static_issues[:5]:
        print(f"  - Line {issue.line_number}: {issue.message}")
    
    print("\n=== Full Code Review Demo ===")
    print("To use the full AI-powered review:")
    print("1. Set your API key: export ANTHROPIC_API_KEY='your-key' or OPENAI_API_KEY='your-key'")
    print("2. Initialize: orchestrator = CodeReviewOrchestrator(provider='anthropic')")
    print("3. Review code: result = orchestrator.review_code_string(code)")
    print("4. Review files: result = orchestrator.review_file('myfile.py')")
    print("5. Get summary: report = orchestrator.get_summary_report()")
    
    # Example usage (uncomment when API keys are set):
    # orchestrator = CodeReviewOrchestrator(provider="anthropic")
    # result = orchestrator.review_code_string(test_code, "example.py")
    # print(f"\nFound {len(result.issues)} total issues")
    # print(f"Summary: {result.summary}")
    # print(f"Metrics: {result.metrics}")