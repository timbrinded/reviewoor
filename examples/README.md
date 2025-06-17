# Reviewoor Examples

This directory contains runnable examples demonstrating different features of the Reviewoor code review agent.

## Prerequisites

1. Install dependencies:
   ```bash
   cd ..  # Go to project root
   uv pip install -e .
   ```

2. Set up your API keys:
   ```bash
   cp ../.env.example ../.env
   # Edit ../.env and add your API keys
   source ../.env
   ```

## Examples

### 01_basic_review.py
**Basic Code Review** - The simplest way to review code
- Shows how to review a code string
- Demonstrates fallback to static analysis when no API key is available
- Groups and displays issues by severity
- Shows code metrics and summary

```bash
uv run 01_basic_review.py
```

### 02_file_review.py
**File Review** - Review existing Python files
- Reviews a specific file on disk
- Shows top issues found
- Exports results to JSON
- Demonstrates the file review API

```bash
uv run 02_file_review.py
```

### 03_directory_review.py
**Directory Review** - Review all Python files in a directory
- Reviews multiple files in a directory
- Provides summary statistics
- Highlights files with critical issues
- Shows aggregate metrics across all files

```bash
uv run 03_directory_review.py
```

### 04_static_analysis_only.py
**Static Analysis Only** - Use without API keys
- Runs only the AST-based static analyzer
- Great for quick checks or CI/CD pipelines
- Shows all types of issues the static analyzer can detect
- No API keys required!

```bash
uv run 04_static_analysis_only.py
```

### 05_custom_review.py
**Provider Comparison** - Compare Anthropic vs OpenAI results
- Reviews the same code with different AI providers
- Compares the results side by side
- Shows which issues are AI-specific vs static
- Requires at least one API key (works best with both)

```bash
uv run 05_custom_review.py
```

### 06_security_focused.py
**Security Review** - Focus on security vulnerabilities
- Reviews code specifically for security issues
- Categorizes vulnerabilities by type
- Provides security-specific recommendations
- Shows security score and best practices

```bash
uv run 06_security_focused.py
```

## Running All Examples

To run all examples sequentially:

```bash
# Make sure you're in the examples directory
for example in *.py; do
    if [ "$example" != "__init__.py" ]; then
        echo "Running $example..."
        uv run "$example"
        echo -e "\n---\n"
    fi
done
```

## Tips

1. **No API Keys?** Examples 01 and 04 work without API keys using static analysis
2. **Quick Test:** Run example 01 first to verify your setup
3. **Full Demo:** Run example 03 to see a complete directory review
4. **Security Focus:** Use example 06 for security-focused code reviews

## Output Files

All output files are saved to the `output/` directory (which is gitignored):
- `output/file_review_results.json` - From example 02
- `output/directory_review_results.json` - From example 03
- `output/anthropic_review_results.json` - From example 05 (if using Anthropic)
- `output/openai_review_results.json` - From example 05 (if using OpenAI)
- `output/security_review_results.json` - From example 06

These files contain detailed review results in JSON format for further processing. The output directory is not committed to git.