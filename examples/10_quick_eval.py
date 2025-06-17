#!/usr/bin/env python3
"""
Example 10: Quick Evaluation
Simple example showing how to use the evaluation framework.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import quick_benchmark

def main():
    print("=== Quick Model Evaluation ===\n")
    
    # Run quick benchmark with default models
    print("Running benchmark on available models...")
    print("This will test each model on security, performance, and quality issues.\n")
    
    results = quick_benchmark()
    
    print("\n✨ Evaluation complete!")
    print("\nKey insights:")
    print("- Check the output folder for detailed JSON results")
    print("- Visit https://app.agentops.ai for token usage and cost analysis")
    print("- Use the full benchmark module for custom test cases")

if __name__ == "__main__":
    main()