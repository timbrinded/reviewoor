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

# Run a specific example
python examples/01_basic_review.py

# Or run all examples
cd examples && ./run_all.sh
```

## Architecture

### Package Structure

- **src/models/**: Data models for code issues, results, and severity levels
- **src/analyzers/**: Static code analysis using AST parsing and regex patterns
- **src/agents/**: AI agent implementations for Anthropic and OpenAI
  - `base.py`: Abstract base class defining the agent interface
  - Provider-specific implementations inherit from BaseCodeReviewAgent
- **src/core/**: Orchestration layer for managing reviews across files
  - `orchestrator.py`: CodeReviewOrchestrator handles file/directory reviews and result aggregation

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

- Set `ANTHROPIC_API_KEY` environment variable for Claude reviews
- Set `OPENAI_API_KEY` environment variable for GPT reviews
- API keys are read from environment in the agent implementations
- Default provider is "anthropic" but can be specified in CodeReviewOrchestrator constructor
