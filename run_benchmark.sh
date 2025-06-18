#!/bin/bash

# Advanced DeepEval Benchmark Runner
# This script provides convenient ways to run different benchmark configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not activated. Activating..."
    source .venv/bin/activate
fi

# Load environment variables from .env file if it exists
if [[ -f ".env" ]]; then
    print_info "Loading environment variables from .env file"
    source .env
fi

# Check if required environment variables are set
if [[ -z "$OPENAI_API_KEY" && -z "$ANTHROPIC_API_KEY" ]]; then
    print_error "No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file"
    exit 1
fi

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Advanced DeepEval Benchmark Runner

OPTIONS:
    --quick                 Quick test with minimal models
    --openai-only          Test only OpenAI models
    --anthropic-only       Test only Anthropic models
    --security-only        Test only security test suite
    --performance-only     Test only performance test suite
    --quality-only         Test only code quality test suite
    --no-tools             Run without tools enabled
    --security-focused     Run with security-focused prompt
    --performance-focused  Run with performance-focused prompt
    --all-run-types        Run all run types (standard, no-tools, security-focused, performance-focused)
    --save-dataset         Save results as DeepEval dataset
    --models MODEL_LIST    Specific models to test (e.g., "openai:gpt-4o openai:gpt-4")
    --help                 Show this help message

EXAMPLES:
    # Quick test with one model
    $0 --quick

    # Test only OpenAI models with security focus
    $0 --openai-only --security-only --security-focused

    # Comprehensive test with all run types
    $0 --all-run-types --save-dataset

    # Test specific models
    $0 --models "openai:gpt-4o-mini anthropic:claude-3-5-haiku"

    # Performance comparison
    $0 --performance-only --performance-focused --all-run-types

EOF
}

# Parse command line arguments
ARGS=()
MODELS=""
RUN_TYPES=""
TEST_SUITES=""
SAVE_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODELS="openai:gpt-4o-mini-2024-07-18"
            RUN_TYPES="standard"
            TEST_SUITES="security"
            shift
            ;;
        --openai-only)
            if [[ -z "$OPENAI_API_KEY" ]]; then
                print_error "OPENAI_API_KEY not set"
                exit 1
            fi
            MODELS="openai:gpt-4o-mini-2024-07-18 openai:gpt-4o-2024-08-06"
            shift
            ;;
        --anthropic-only)
            if [[ -z "$ANTHROPIC_API_KEY" ]]; then
                print_error "ANTHROPIC_API_KEY not set"
                exit 1
            fi
            MODELS="anthropic:claude-3-5-haiku-20241022 anthropic:claude-3-5-sonnet-20241022"
            shift
            ;;
        --security-only)
            TEST_SUITES="security"
            shift
            ;;
        --performance-only)
            TEST_SUITES="performance"
            shift
            ;;
        --quality-only)
            TEST_SUITES="code_quality"
            shift
            ;;
        --no-tools)
            RUN_TYPES="no-tools"
            shift
            ;;
        --security-focused)
            RUN_TYPES="security-focused"
            shift
            ;;
        --performance-focused)
            RUN_TYPES="performance-focused"
            shift
            ;;
        --all-run-types)
            RUN_TYPES="standard no-tools security-focused performance-focused"
            shift
            ;;
        --save-dataset)
            SAVE_DATASET="--save-dataset"
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Build command
CMD="uv run python examples/12_deepeval_advanced_benchmark.py"

if [[ -n "$MODELS" ]]; then
    CMD="$CMD --models $MODELS"
fi

if [[ -n "$RUN_TYPES" ]]; then
    CMD="$CMD --run-types $RUN_TYPES"
fi

if [[ -n "$TEST_SUITES" ]]; then
    CMD="$CMD --test-suites $TEST_SUITES"
fi

if [[ -n "$SAVE_DATASET" ]]; then
    CMD="$CMD $SAVE_DATASET"
fi

# Add any remaining arguments
for arg in "${ARGS[@]}"; do
    CMD="$CMD $arg"
done

# Print what we're about to run
print_info "Running benchmark with command:"
echo "  $CMD"
echo

# Confirm before running (unless it's a quick test)
if [[ "$MODELS" != "openai:gpt-4o-mini-2024-07-18" ]]; then
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Benchmark cancelled"
        exit 0
    fi
fi

# Run the benchmark
print_info "Starting benchmark..."
start_time=$(date +%s)

if eval "$CMD"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    print_success "Benchmark completed successfully!"
    print_info "Total time: ${duration} seconds"
    
    # Show output directory
    print_info "Results saved to: output/"
    ls -la output/ | tail -5
else
    print_error "Benchmark failed!"
    exit 1
fi