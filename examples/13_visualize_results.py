#!/usr/bin/env python3
"""
Example 11: Visualize Benchmark Results
This example shows how to visualize benchmark results with charts and graphs.
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import BenchmarkVisualizer

def find_latest_benchmark_results():
    """Find the most recent benchmark results file"""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    
    if not os.path.exists(output_dir):
        return None
    
    # Look for benchmark result files
    result_files = []
    for file in os.listdir(output_dir):
        if file.startswith("benchmark_") and file.endswith(".json"):
            result_files.append(os.path.join(output_dir, file))
    
    if not result_files:
        return None
    
    # Return the most recent file
    return max(result_files, key=os.path.getctime)

def create_sample_results():
    """Create sample results for demonstration"""
    return {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "model": "gpt-4",
                "provider": "openai",
                "total_time": 45.2,
                "total_issues": 42,
                "metrics": {
                    "avg_precision": 0.85,
                    "avg_recall": 0.78,
                    "avg_f1": 0.81
                },
                "severity_distribution": {
                    "critical": 5,
                    "high": 12,
                    "medium": 15,
                    "low": 10
                }
            },
            {
                "model": "gpt-3.5-turbo",
                "provider": "openai",
                "total_time": 18.5,
                "total_issues": 35,
                "metrics": {
                    "avg_precision": 0.72,
                    "avg_recall": 0.65,
                    "avg_f1": 0.68
                },
                "severity_distribution": {
                    "critical": 3,
                    "high": 8,
                    "medium": 14,
                    "low": 10
                }
            },
            {
                "model": "claude-3-5-sonnet-20241022",
                "provider": "anthropic",
                "total_time": 38.7,
                "total_issues": 45,
                "metrics": {
                    "avg_precision": 0.88,
                    "avg_recall": 0.82,
                    "avg_f1": 0.85
                },
                "severity_distribution": {
                    "critical": 6,
                    "high": 14,
                    "medium": 16,
                    "low": 9
                }
            },
            {
                "model": "o3-mini",
                "provider": "openai",
                "total_time": 25.3,
                "total_issues": 38,
                "metrics": {
                    "avg_precision": 0.79,
                    "avg_recall": 0.71,
                    "avg_f1": 0.75
                },
                "severity_distribution": {
                    "critical": 4,
                    "high": 10,
                    "medium": 15,
                    "low": 9
                }
            }
        ]
    }

def main():
    print("=== Benchmark Results Visualization ===\n")
    
    # Find existing results or create sample data
    results_file = find_latest_benchmark_results()
    
    visualizer = BenchmarkVisualizer()
    
    if results_file:
        print(f"Found benchmark results: {os.path.basename(results_file)}")
        visualizer = BenchmarkVisualizer(results_file)
    else:
        print("No benchmark results found. Creating sample data for demonstration...")
        sample_data = create_sample_results()
        visualizer.load_results(sample_data['results'])
        
        # Save sample data
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        sample_file = os.path.join(output_dir, "sample_benchmark_results.json")
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Sample data saved to: {sample_file}\n")
    
    # Create visualization directory
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Generate individual charts
    print("\n1. Creating accuracy comparison chart...")
    visualizer.create_accuracy_comparison(
        os.path.join(viz_dir, "accuracy_comparison.png")
    )
    
    print("2. Creating performance scatter plot...")
    visualizer.create_performance_scatter(
        os.path.join(viz_dir, "performance_scatter.png")
    )
    
    print("3. Creating issue distribution heatmap...")
    visualizer.create_issue_distribution_heatmap(
        os.path.join(viz_dir, "issue_heatmap.png")
    )
    
    print("4. Creating radar chart...")
    visualizer.create_radar_chart(
        os.path.join(viz_dir, "radar_chart.png")
    )
    
    # Generate complete report
    print("\n5. Generating complete visual report...")
    visualizer.create_summary_report(viz_dir)
    
    print("\n✨ Visualization complete!")
    print(f"\nAll visualizations saved to: {viz_dir}")
    print("\n📊 Chart descriptions:")
    print("   - accuracy_comparison.png: Bar chart comparing precision, recall, and F1 scores")
    print("   - performance_scatter.png: Scatter plot showing speed vs accuracy trade-offs")
    print("   - issue_heatmap.png: Heatmap of issue severity distribution by model")
    print("   - radar_chart.png: Multi-dimensional comparison of model capabilities")
    print("\n🌐 Open the HTML report for an interactive view of all charts together")

if __name__ == "__main__":
    main()