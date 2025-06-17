#!/usr/bin/env python3
"""Test visualization directly"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load the data
with open('output/benchmark_results_20250617_135644.json', 'r') as f:
    results = json.load(f)

# Create a simple bar chart
models = [f"{r['provider']}/{r['model'][:20]}" for r in results[:5]]  # First 5 only
accuracy_scores = [r['accuracy_score'] for r in results[:5]]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(models)), accuracy_scores)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel('Accuracy Score')
ax.set_title('Test Visualization')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('output/test_viz.png')
print(f"Saved test visualization to output/test_viz.png")
print(f"Accuracy scores plotted: {accuracy_scores}")