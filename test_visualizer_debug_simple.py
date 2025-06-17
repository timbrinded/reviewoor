#!/usr/bin/env python3
"""
Test script to verify visualizer debug output - simplified version
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import just the visualizer module directly
import evaluation.visualizer
from evaluation.visualizer import BenchmarkVisualizer

# Test with sample data that mimics the format from example 09
test_data = [
    {
        "model": "gpt-4",
        "provider": "openai",
        "total_issues_found": 42,
        "response_time": 45.2,
        "accuracy_score": 0.85,
        "false_positive_rate": 0.15
    },
    {
        "model": "gpt-3.5-turbo", 
        "provider": "openai",
        "total_issues_found": 35,
        "response_time": 18.5,
        "accuracy_score": 0.72,
        "false_positive_rate": 0.28
    }
]

print("Testing visualizer with example 09 format data...")
print("="*60)

visualizer = BenchmarkVisualizer()
visualizer.load_results(test_data)

print("\nCreating accuracy comparison chart...")
# Create a test output directory
output_dir = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(output_dir, exist_ok=True)

visualizer.create_accuracy_comparison(os.path.join(output_dir, "test_accuracy.png"))

print("\n" + "="*60)
print("Debug output complete. Check above for details.")
print(f"Chart saved to: {os.path.join(output_dir, 'test_accuracy.png')}")