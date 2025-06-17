#!/usr/bin/env python3
"""
Test script to verify visualizer debug output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.visualizer import BenchmarkVisualizer

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
visualizer.create_accuracy_comparison()

print("\n" + "="*60)
print("Debug output complete. Check above for details.")