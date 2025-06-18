# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an AI-powered Python code review agent called AgentOps (or reviewoor). It performs static analysis and AI-based code review using either Anthropic Claude or OpenAI GPT models. The project is structured as a Python package with clear separation between models, analyzers, agents, and orchestration layers.

## Development Commands

### Environment Setup
```bash
# Create virtual environment using uv
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Type Checking and Linting
```bash
# Run type checking with pyright
pyright

# Run linting with ruff (if installed globally or in venv)
ruff check src/
ruff format src/
```

### Testing
```bash
# Run tests with pytest (when tests are implemented)
pytest
```

### Running Examples
```bash
# Set up API keys using .env file
cp .env.example .env
# Edit .env to add your API keys

# Source the .env file
source .env

# Run a specific example using uv
uv run examples/01_basic_review.py

# Or run all examples
cd examples && ./run_all.sh
```

### Important: Using the uv Ecosystem
This project uses `uv` for package management. Always use `uv run` to execute Python scripts:
- `uv run script.py` - Run a Python script
- `uv pip install -e .` - Install the package in development mode
- Never use plain `python` commands - always prefix with `uv run`

### Output Files
All generated output files (JSON results) are saved to the `output/` directory, which is gitignored. This keeps the repository clean and prevents committing potentially large result files.

## Architecture

### Package Structure

- **src/models/**: Data models for code issues, results, and severity levels
- **src/analyzers/**: Static code analysis using AST parsing and regex patterns
- **src/agents/**: AI agent implementations for Anthropic and OpenAI
  - `base.py`: Abstract base class defining the agent interface
  - Provider-specific implementations inherit from BaseCodeReviewAgent
- **src/core/**: Orchestration layer for managing reviews across files
  - `orchestrator.py`: CodeReviewOrchestrator handles file/directory reviews and result aggregation
- **src/evaluation/**: Model benchmarking and evaluation framework
  - `benchmark.py`: Comprehensive benchmarking system with accuracy metrics

### Key Components

1. **CodeAnalyzer** (src/analyzers/static_analyzer.py): Performs AST-based static analysis
   - Detects syntax errors, security issues, code style violations
   - Checks for common Python anti-patterns

2. **BaseCodeReviewAgent** (src/agents/base.py): Abstract base for AI agents
   - Combines static analysis with AI review
   - Handles prompt formatting and response parsing
   - Deduplicates issues from multiple sources

3. **CodeReviewOrchestrator** (src/core/orchestrator.py): High-level API
   - Supports single file, code string, and directory reviews
   - Aggregates results and generates summary reports
   - Exports results to JSON

### Review Process Flow

1. Static analysis runs first using CodeAnalyzer
2. Static issues are passed to the AI model for context
3. AI performs additional analysis for logic errors, design issues, etc.
4. Results are combined and deduplicated
5. Final ReviewResult includes all issues, metrics, and summary

## API Keys and Configuration

- Create a `.env` file in the project root with your API keys:
  - `OPENAI_API_KEY=your-openai-key` for OpenAI GPT reviews (default provider)
  - `ANTHROPIC_API_KEY=your-anthropic-key` for Anthropic Claude reviews
  - `AGENTOPS_API_KEY=your-agentops-key` for tracking and analytics (optional, defaults provided)
- The examples automatically load the `.env` file using python-dotenv
- API keys are read from environment in the agent implementations
- Default provider is now "openai" (changed from "anthropic")
- Provider and model can be specified in CodeReviewOrchestrator constructor:
  - `CodeReviewOrchestrator()` - Uses OpenAI with GPT-4
  - `CodeReviewOrchestrator(provider="anthropic")` - Uses Anthropic with Claude 3.5 Sonnet
  - `CodeReviewOrchestrator(model="gpt-3.5-turbo")` - Uses OpenAI with GPT-3.5 Turbo
  - `CodeReviewOrchestrator(provider="anthropic", model="claude-3-haiku-20240307")` - Uses specific Anthropic model

## Model Evaluation and Benchmarking

The project includes a comprehensive evaluation framework for comparing model performance:

### Using the Evaluation Framework

```python
from src.evaluation import quick_benchmark

# Run benchmark with default models
results = quick_benchmark()

# Or use custom models
results = quick_benchmark([
    ("openai", "gpt-4"),
    ("openai", "gpt-3.5-turbo"),
    ("anthropic", "claude-3-5-sonnet-20241022")
])
```

### AgentOps Integration

All benchmarks are tracked with AgentOps, providing:
- Token usage and cost analysis per model
- Response time distributions
- Error rates and types
- Model-specific performance patterns

View results at: https://app.agentops.ai

### Metrics Tracked

- **Accuracy Metrics**: Precision, Recall, F1 Score
- **Performance**: Response time, tokens used
- **Issue Detection**: Total issues, severity distribution
- **Cost Analysis**: Via AgentOps dashboard

## Known Issues and Solutions

### DeepEval Browser Opening
- DeepEval automatically opens browser when running evaluations by default
- To disable this, the examples set `os.environ["CI"] = "true"` which makes DeepEval think it's running in CI
- The link to results is still printed but won't auto-open in browser

### O3 Model Support
- O3 models (o3-mini, o3) require using `max_completion_tokens` instead of `max_tokens`
- O3-pro models (e.g., o3-pro-2025-06-10) use the responses endpoint with:
  - `input` parameter instead of `messages`
  - `max_output_tokens` instead of `max_completion_tokens`
- The OpenAI agent automatically detects the model type and uses the correct endpoint/parameters
- If the responses endpoint fails, it falls back to chat completions with `max_completion_tokens`
- O3-pro models take significantly longer to process (30-60+ seconds)

### Template String Formatting
When using Python's `.format()` with templates containing JSON, remember to escape literal curly braces by doubling them:
- `{{` for literal `{`
- `}}` for literal `}`
- Single braces `{variable}` for format placeholders

This is particularly important in the review prompt templates that contain JSON examples.

### Issue Deduplication
The `_deduplicate_issues` method in `base.py` must handle cases where `line_number` might be:
- An integer (normal case)
- `None` (for general issues without specific line numbers)
- A list (if the AI returns multiple line numbers)

Always convert non-hashable types to hashable ones before using them in set operations.
