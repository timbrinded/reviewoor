#!/bin/bash
# Run all examples in sequence

echo "🚀 Running all Reviewoor examples..."
echo "=================================="
echo ""

# Check if we're in the examples directory
if [ ! -f "01_basic_review.py" ]; then
    echo "Error: Please run this script from the examples directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "   Some examples will use static analysis only."
    echo "   To use AI features, create ../.env with your API keys."
    echo ""
fi

# Source .env if it exists
if [ -f "../.env" ]; then
    source ../.env
    echo "✅ Loaded environment variables from .env"
    echo ""
fi

# Counter for examples
count=0
failed=0

# Run each example
for example in [0-9]*.py; do
    count=$((count + 1))
    echo "▶️  Running $example"
    echo "-----------------------------------"
    
    if uv run "$example"; then
        echo "✅ $example completed successfully"
    else
        echo "❌ $example failed"
        failed=$((failed + 1))
    fi
    
    echo ""
    echo "==================================="
    echo ""
    
    # Small pause between examples
    sleep 1
done

# Summary
echo "📊 Summary"
echo "----------"
echo "Total examples run: $count"
echo "Successful: $((count - failed))"
echo "Failed: $failed"
echo ""

# List any output files created
echo "📁 Output files created in ../output/:"
if [ -d "../output" ]; then
    for file in ../output/*.json; do
        if [ -f "$file" ]; then
            echo "   - $(basename "$file")"
        fi
    done
else
    echo "   (no output directory found)"
fi

echo ""
echo "✨ All examples completed!"