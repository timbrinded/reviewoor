#!/usr/bin/env python3
"""Create a single test chart to verify visualization"""

import json
import matplotlib.pyplot as plt

# Load the benchmark results
with open('output/benchmark_results_20250617_135644.json', 'r') as f:
    results = json.load(f)

# Extract data for first 5 models
models = []
precision_vals = []
recall_vals = []
f1_vals = []

for r in results[:5]:
    models.append(f"{r['provider']}/{r['model'][:15]}...")  # Truncate long names
    
    accuracy = r.get('accuracy_score', 0)
    fp_rate = r.get('false_positive_rate', 0)
    
    precision = 1 - fp_rate if fp_rate < 1 else 0
    recall = accuracy
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_vals.append(precision)
    recall_vals.append(recall)
    f1_vals.append(f1)

# Create chart
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(models))
width = 0.25

bars1 = ax.bar([i - width for i in x], precision_vals, width, label='Precision', alpha=0.8, color='blue')
bars2 = ax.bar(x, recall_vals, width, label='Recall', alpha=0.8, color='green')
bars3 = ax.bar([i + width for i in x], f1_vals, width, label='F1 Score', alpha=0.8, color='red')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Test: Model Accuracy Metrics')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/test_accuracy_chart.png', dpi=150)
print("Saved test chart to output/test_accuracy_chart.png")

# Print what should be visible
print("\nValues that should be visible in the chart:")
for i, model in enumerate(models):
    print(f"{model}: P={precision_vals[i]:.2f}, R={recall_vals[i]:.2f}, F1={f1_vals[i]:.2f}")