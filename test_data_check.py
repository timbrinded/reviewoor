#!/usr/bin/env python3
"""Check what data is in the visualizations"""

import json
import numpy as np

# Load the benchmark results
with open('output/benchmark_results_20250617_135644.json', 'r') as f:
    results = json.load(f)

print("Benchmark Results Summary:")
print(f"Total models tested: {len(results)}")
print("\nModel Performance:")

for i, result in enumerate(results[:5]):  # Show first 5
    accuracy = result.get('accuracy_score', 0)
    fp_rate = result.get('false_positive_rate', 0)
    
    # Calculate metrics the same way as visualizer
    precision = 1 - fp_rate if fp_rate < 1 else 0
    recall = accuracy
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{i+1}. {result['provider']}/{result['model']}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   FP Rate: {fp_rate:.3f}")
    print(f"   Calculated Precision: {precision:.3f}")
    print(f"   Calculated Recall: {recall:.3f}")
    print(f"   Calculated F1: {f1:.3f}")

# Check if any values would be zero
print("\n\nChecking for zero values:")
all_accuracies = [r.get('accuracy_score', 0) for r in results]
all_fp_rates = [r.get('false_positive_rate', 0) for r in results]

print(f"Min accuracy: {min(all_accuracies)}")
print(f"Max accuracy: {max(all_accuracies)}")
print(f"Min FP rate: {min(all_fp_rates)}")
print(f"Max FP rate: {max(all_fp_rates)}")
print(f"Number of zero accuracies: {sum(1 for a in all_accuracies if a == 0)}")

# Show the model with zero accuracy
for r in results:
    if r.get('accuracy_score', 0) == 0:
        print(f"\nModel with zero accuracy: {r['provider']}/{r['model']}")