"""
Visualization tools for benchmark results.
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BenchmarkVisualizer:
    """Create visualizations for benchmark results"""
    
    def __init__(self, results_path: Optional[str] = None):
        self.results = []
        if results_path and os.path.exists(results_path):
            print(f"DEBUG: Loading results from file: {results_path}")
            with open(results_path, 'r') as f:
                data = json.load(f)
                # Handle both wrapped and direct list formats
                if isinstance(data, list):
                    self.results = data
                else:
                    self.results = data.get('results', [])
    
    def load_results(self, results: List[Dict[str, Any]]):
        """Load results directly from a list"""
        self.results = results
        print(f"DEBUG: Loaded {len(results)} results via load_results method")
        if results:
            print(f"DEBUG: First result keys: {list(results[0].keys())}")
    
    def create_accuracy_comparison(self, save_path: Optional[str] = None):
        """Create a bar chart comparing accuracy metrics across models"""
        if not self.results:
            print("No results to visualize")
            return
        
        print(f"DEBUG: Number of results: {len(self.results)}")
        print(f"DEBUG: Results structure: {self.results[:1] if self.results else 'No results'}")
        
        # Prepare data
        models = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i, result in enumerate(self.results):
            print(f"\nDEBUG: Processing result {i}:")
            print(f"  Provider: {result.get('provider', 'N/A')}")
            print(f"  Model: {result.get('model', 'N/A')}")
            print(f"  Result keys: {list(result.keys())}")
            
            model_name = f"{result['provider']}/{result['model']}"
            models.append(model_name)
            metrics = result.get('metrics', {})
            
            print(f"  Metrics found: {metrics}")
            print(f"  Available metric keys: {list(metrics.keys()) if metrics else 'No metrics'}")
            
            # Check if we have the metrics in the expected format
            if metrics and 'avg_precision' in metrics:
                precision = metrics.get('avg_precision', 0)
                recall = metrics.get('avg_recall', 0)
                f1 = metrics.get('avg_f1', 0)
                print(f"  Using direct metrics")
            else:
                # Fall back to calculating from accuracy_score and false_positive_rate
                accuracy = result.get('accuracy_score', 0)
                fp_rate = result.get('false_positive_rate', 0)
                print(f"  No direct metrics, using accuracy_score={accuracy}, fp_rate={fp_rate}")
                
                if accuracy > 0 or fp_rate > 0:
                    precision = max(0, 1 - fp_rate)
                    recall = accuracy
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision = 0
                    recall = 0
                    f1 = 0
            
            print(f"  Precision: {precision}")
            print(f"  Recall: {recall}")
            print(f"  F1: {f1}")
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        print(f"\nDEBUG: Final data for plotting:")
        print(f"  Models: {models}")
        print(f"  Precision scores: {precision_scores}")
        print(f"  Recall scores: {recall_scores}")
        print(f"  F1 scores: {f1_scores}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.25
        
        # Create bars - convert numpy arrays to proper types
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DEBUG: Saved accuracy comparison to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)  # Close the figure to prevent memory issues
        return fig
    
    def create_performance_scatter(self, save_path: Optional[str] = None):
        """Create a scatter plot of response time vs accuracy"""
        if not self.results:
            print("No results to visualize")
            return
        
        print(f"\nDEBUG: create_performance_scatter - Processing {len(self.results)} results")
        
        # Prepare data
        response_times = []
        f1_scores = []
        labels = []
        providers = []
        
        for i, result in enumerate(self.results):
            # Handle different time field names
            time_val = result.get('total_time', result.get('response_time', 0))
            
            # Calculate F1 score
            if 'metrics' in result and result['metrics'] and 'avg_f1' in result['metrics']:
                f1_val = result['metrics']['avg_f1']
            else:
                # Calculate from accuracy and FP rate
                accuracy = result.get('accuracy_score', 0)
                fp_rate = result.get('false_positive_rate', 0)
                if accuracy > 0 or fp_rate > 0:
                    precision = max(0, 1 - fp_rate)
                    recall = accuracy
                    f1_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    f1_val = 0
            
            print(f"DEBUG: Result {i}: time={time_val}, f1={f1_val}, model={result.get('model', 'N/A')}")
            
            response_times.append(time_val)
            f1_scores.append(f1_val)
            labels.append(result['model'])
            providers.append(result['provider'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by provider
        colors = {'openai': 'blue', 'anthropic': 'red'}
        for i, (x, y, label, provider) in enumerate(zip(response_times, f1_scores, labels, providers)):
            ax.scatter(x, y, s=200, c=colors.get(provider, 'gray'), alpha=0.6, edgecolors='black', linewidth=1)
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Customize plot
        ax.set_xlabel('Total Response Time (seconds)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Model Performance: Speed vs Accuracy', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        patches = [mpatches.Patch(color=color, label=provider.capitalize()) 
                  for provider, color in colors.items()]
        ax.legend(handles=patches)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DEBUG: Saved accuracy comparison to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)  # Close the figure to prevent memory issues
        return fig
    
    def create_issue_distribution_heatmap(self, save_path: Optional[str] = None):
        """Create a heatmap of issue severities by model"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Prepare data
        severity_types = ['critical', 'high', 'medium', 'low', 'info']
        models = []
        severity_data = []
        
        for result in self.results:
            model_name = f"{result['provider']}/{result['model']}"
            models.append(model_name)
            
            severity_dist = result.get('severity_distribution', {})
            row = [severity_dist.get(sev, 0) for sev in severity_types]
            severity_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(severity_data, index=models, columns=severity_types)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Issue Count'}, ax=ax)
        
        # Customize plot
        ax.set_title('Issue Severity Distribution by Model', fontsize=16, fontweight='bold')
        ax.set_xlabel('Severity Level', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DEBUG: Saved accuracy comparison to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)  # Close the figure to prevent memory issues
        return fig
    
    def create_radar_chart(self, save_path: Optional[str] = None):
        """Create a radar chart comparing multiple metrics across models"""
        if not self.results:
            print("No results to visualize")
            return
        
        print(f"\nDEBUG: create_radar_chart - Processing {len(self.results)} results")
        
        # Prepare data - normalize metrics to 0-1 scale
        categories = ['Precision', 'Recall', 'F1 Score', 'Speed', 'Issue Detection']
        models_data = []
        
        # Calculate max values for normalization
        time_values = [r.get('total_time', r.get('response_time', 0)) for r in self.results]
        max_time = max(time_values) if time_values and max(time_values) > 0 else 1
        
        issue_values = [r.get('total_issues', r.get('total_issues_found', 0)) for r in self.results]
        max_issues = max(issue_values) if issue_values and max(issue_values) > 0 else 1
        
        print(f"DEBUG: Time values: {time_values}")
        print(f"DEBUG: Issue values: {issue_values}")
        print(f"DEBUG: Max time: {max_time}, Max issues: {max_issues}")
        
        for i, result in enumerate(self.results):
            model_name = f"{result['provider']}/{result['model']}"
            
            print(f"\nDEBUG: Processing model {i}: {model_name}")
            print(f"  Has 'metrics' key: {'metrics' in result}")
            print(f"  Result keys: {list(result.keys())}")
            
            if 'metrics' in result and result['metrics']:
                metrics = result.get('metrics', {})
                precision = metrics.get('avg_precision', 0)
                recall = metrics.get('avg_recall', 0)
                f1 = metrics.get('avg_f1', 0)
                print(f"  Using metrics: precision={precision}, recall={recall}, f1={f1}")
            else:
                # Calculate from accuracy and FP rate (used by example benchmark script)
                accuracy = result.get('accuracy_score', 0)
                fp_rate = result.get('false_positive_rate', 0)
                
                # Convert accuracy/FPR to precision/recall/F1
                # Note: This is an approximation since we don't have true positives/negatives
                if accuracy > 0 or fp_rate > 0:
                    # Approximate precision as 1 - FPR (assumes most findings are relevant)
                    precision = max(0, 1 - fp_rate)
                    # Use accuracy as recall (portion of issues found)
                    recall = accuracy
                    # Calculate F1 score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision = 0
                    recall = 0
                    f1 = 0
                    
                print(f"  Calculated from accuracy_score={accuracy}, fp_rate={fp_rate}")
                print(f"  Calculated: precision={precision}, recall={recall}, f1={f1}")
            
            # Speed score (inverse of time, normalized)
            time_value = result.get('total_time', result.get('response_time', 0))
            speed_score = 1 - (time_value / max_time)
            
            # Issue detection score (normalized)
            issues_value = result.get('total_issues', result.get('total_issues_found', 0))
            detection_score = issues_value / max_issues
            
            print(f"  Time value: {time_value}, Speed score: {speed_score}")
            print(f"  Issues value: {issues_value}, Detection score: {detection_score}")
            
            values = [
                precision,
                recall,
                f1,
                speed_score,
                detection_score
            ]
            
            print(f"  Final values for radar: {values}")
            
            models_data.append((model_name, values))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot data for each model
        for i, (model_name, values) in enumerate(models_data[:5]):  # Limit to 5 models for clarity
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        if hasattr(ax, 'set_theta_offset'):
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        
        # Add title and legend
        ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DEBUG: Saved accuracy comparison to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)  # Close the figure to prevent memory issues
        return fig
    
    def create_summary_report(self, output_dir: str):
        """Create a complete visual report with all charts"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate all visualizations
        print("Generating accuracy comparison chart...")
        self.create_accuracy_comparison(
            os.path.join(output_dir, f"accuracy_comparison_{timestamp}.png")
        )
        
        print("Generating performance scatter plot...")
        self.create_performance_scatter(
            os.path.join(output_dir, f"performance_scatter_{timestamp}.png")
        )
        
        print("Generating issue distribution heatmap...")
        self.create_issue_distribution_heatmap(
            os.path.join(output_dir, f"issue_heatmap_{timestamp}.png")
        )
        
        print("Generating radar chart...")
        self.create_radar_chart(
            os.path.join(output_dir, f"radar_chart_{timestamp}.png")
        )
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Review Model Benchmark Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                .chart-container {{
                    background-color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <h1>Code Review Model Benchmark Report</h1>
            <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="chart-container">
                <h2>Accuracy Comparison</h2>
                <img src="accuracy_comparison_{timestamp}.png" alt="Accuracy Comparison">
            </div>
            
            <div class="chart-container">
                <h2>Performance vs Accuracy</h2>
                <img src="performance_scatter_{timestamp}.png" alt="Performance Scatter">
            </div>
            
            <div class="chart-container">
                <h2>Issue Severity Distribution</h2>
                <img src="issue_heatmap_{timestamp}.png" alt="Issue Heatmap">
            </div>
            
            <div class="chart-container">
                <h2>Multi-Metric Comparison</h2>
                <img src="radar_chart_{timestamp}.png" alt="Radar Chart">
            </div>
        </body>
        </html>
        """
        
        html_path = os.path.join(output_dir, f"benchmark_report_{timestamp}.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n✨ Visual report generated in: {output_dir}")
        print(f"   Open {html_path} to view the complete report")
        
        return output_dir