# Reviewoor (AgentOps)

An AI-powered Python code review agent that combines static analysis with large language models to provide comprehensive code reviews. Supports both Anthropic Claude and OpenAI GPT models.

## Description

Reviewoor is a Python package that performs intelligent code review by:

- Running AST-based static analysis to catch syntax errors, security issues, and common anti-patterns
- Using AI models (Claude or GPT) to identify logic errors, design issues, and provide improvement suggestions
- Combining and deduplicating results from both analysis methods
- Generating detailed reports with severity levels, categories, and metrics

### Key Features

- 🔍 **Dual Analysis**: Combines static analysis with AI-powered review
- 🤖 **Multiple AI Providers**: Support for Anthropic Claude and OpenAI GPT
- 📊 **Comprehensive Metrics**: Complexity, maintainability, and security scores
- 📁 **Flexible Input**: Review single files, code strings, or entire directories
- 📋 **Detailed Reports**: Export results to JSON with categorized issues
- 🎯 **Smart Deduplication**: Avoids duplicate issues from different sources

## Setup

### Prerequisites

- Python 3.12 or higher
- An API key for either Anthropic Claude or OpenAI GPT

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/reviewoor.git
cd reviewoor
```

2. Create and activate a virtual environment using `uv`:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package and dependencies:

```bash
uv pip install -e .
```

4. (Optional) Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

5. Set up your API keys by creating a `.env` file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# The file should contain:
# ANTHROPIC_API_KEY=your-anthropic-api-key
# OPENAI_API_KEY=your-openai-api-key
```

Then source the `.env` file before running the application:

```bash
source .env
```

## Usage

### Basic Usage

```python
from src import CodeReviewOrchestrator

# Initialize with Anthropic (default)
orchestrator = CodeReviewOrchestrator(provider="anthropic")

# Or use OpenAI
orchestrator = CodeReviewOrchestrator(provider="openai")

# Review a single file
result = orchestrator.review_file("path/to/your/file.py")
print(f"Found {len(result.issues)} issues")
print(f"Summary: {result.summary}")

# Review a code string
code = """
def calculate_average(numbers=[]):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)
"""
result = orchestrator.review_code_string(code, "example.py")

# Review an entire directory
results = orchestrator.review_directory("./src", pattern="*.py")

# Generate a summary report
report = orchestrator.get_summary_report()
print(f"Total issues: {report['total_issues']}")
print(f"Issues by severity: {report['issues_by_severity']}")

# Export results to JSON
orchestrator.export_results("review_results.json")
```

### Using Static Analysis Only

If you don't have API keys or want to run static analysis only:

```python
from src import CodeAnalyzer

analyzer = CodeAnalyzer()
issues = analyzer.analyze(your_code)

for issue in issues:
    print(f"Line {issue.line_number}: [{issue.severity.value}] {issue.message}")
```

## Examples

For comprehensive examples, check out the `examples/` directory which contains:
- `01_basic_review.py` - Basic code review demonstration
- `02_file_review.py` - Review existing Python files
- `03_directory_review.py` - Review entire directories
- `04_static_analysis_only.py` - Static analysis without API keys
- `05_custom_review.py` - Compare different AI providers
- `06_security_focused.py` - Security vulnerability detection

Run all examples:
```bash
cd examples
./run_all.sh
```

### Example 1: Quick Code Review

```python
from src import CodeReviewOrchestrator

# Quick review of a problematic function
code = '''
def process_user_data(data=[]):
    password = "admin123"  # Hardcoded password
    
    if data == None:  # Should use 'is None'
        data = []
    
    try:
        result = eval(input("Enter expression: "))  # Dangerous!
    except:  # Bare except
        pass
    
    return result  # May not be defined
'''

orchestrator = CodeReviewOrchestrator(provider="anthropic")
result = orchestrator.review_code_string(code, "user_processor.py")

# Display issues by severity
for issue in sorted(result.issues, key=lambda x: x.severity.value):
    print(f"[{issue.severity.value}] Line {issue.line_number}: {issue.message}")
    if issue.suggestion:
        print(f"  → Suggestion: {issue.suggestion}")
```

### Example 2: Directory Review with Filtering

```python
from src import CodeReviewOrchestrator

orchestrator = CodeReviewOrchestrator(provider="openai")

# Review all Python files in a directory
results = orchestrator.review_directory("./my_project", pattern="*.py")

# Filter critical and high severity issues
critical_issues = []
for result in results:
    for issue in result.issues:
        if issue.severity.value in ["critical", "high"]:
            critical_issues.append((result.file_path, issue))

# Display critical issues
for file_path, issue in critical_issues:
    print(f"{file_path}:{issue.line_number} - {issue.message}")
```

### Example 3: Running the Included Example

```bash
# First, ensure your .env file is sourced
source .env

# Run static analysis only (no API key needed)
python example.py

# Run full AI-powered review (requires API key in .env)
python example.py
```

## Troubleshooting

### Common Issues

1. **"API key not found" error**
   - Ensure you've created and sourced the `.env` file:

     ```bash
     test -f .env || echo ".env file missing!"
     grep ANTHROPIC_API_KEY .env || echo "ANTHROPIC_API_KEY not in .env"
     source .env
     echo $ANTHROPIC_API_KEY  # Should show your key (or masked output)
     ```

   - Alternatively, pass API keys directly to the orchestrator:

     ```python
     orchestrator = CodeReviewOrchestrator(
         provider="anthropic", 
         api_key="your-key"
     )
     ```

2. **"Module not found" errors**
   - Ensure you're in the activated virtual environment:

     ```bash
     which python  # Should point to .venv/bin/python
     ```

   - Reinstall the package: `uv pip install -e .`

3. **Type checking errors with pyright**
   - Ensure pyright is configured to use the correct virtual environment
   - Check `pyrightconfig.json` has the correct venv settings
   - Install type stubs: `uv pip install types-*` for any missing types

4. **Static analysis not detecting issues**
   - The static analyzer focuses on common Python issues
   - Some issues may only be detected by the AI model
   - Check the AST can parse your code (no syntax errors)

5. **AI model timeouts or rate limits**
   - Implement retry logic in your code
   - Consider batching file reviews with delays
   - Check your API usage limits with your provider

### Debug Mode

To see detailed logs:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Now run your review
orchestrator = CodeReviewOrchestrator(provider="anthropic")
result = orchestrator.review_file("test.py")
```

### Getting Help

- Check the example code in `example.py` for working examples
- Review the source code documentation in `src/`
- Ensure your Python code is syntactically valid before review
- For API-specific issues, consult your provider's documentation

## Performance Considerations

- Static analysis is fast and runs locally
- AI review time depends on code size and API latency
- For large codebases, consider:
  - Reviewing files in parallel with multiple orchestrator instances
  - Implementing caching for previously reviewed files
  - Using static analysis only for initial screening

## Limitations

- Currently supports Python code only
- AI models may have token limits for very large files
- Some complex logic errors may not be detected
- Results quality depends on the AI model used
