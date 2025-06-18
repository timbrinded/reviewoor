#!/usr/bin/env python3
"""
Dataset Manager for DeepEval
Utilities for managing DeepEval datasets for code review benchmarking
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# DeepEval imports
import deepeval
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Configure DeepEval
confident_api_key = os.getenv("CONFIDENT_API_KEY")
if confident_api_key:
    deepeval.login_with_confident_api_key(api_key=confident_api_key)
    print("✓ Logged in to DeepEval with Confident AI")


def list_datasets():
    """List available datasets in the datasets directory"""
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        print("No datasets directory found")
        return
    
    datasets = [f for f in os.listdir(datasets_dir) if f.endswith('.json')]
    
    if not datasets:
        print("No datasets found")
        return
    
    print(f"Found {len(datasets)} datasets:")
    for dataset in sorted(datasets):
        path = os.path.join(datasets_dir, dataset)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    test_count = len(data)
                else:
                    test_count = data.get('test_cases', 0) if isinstance(data, dict) else 0
                
                # Get file modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                print(f"  {dataset} ({test_count} test cases, modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
        except Exception as e:
            print(f"  {dataset} (error reading: {e})")


def create_dataset_from_results(results_file: str, dataset_name: str):
    """Create a dataset from benchmark results JSON file"""
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading results file: {e}")
        return
    
    dataset = EvaluationDataset()
    test_cases_added = 0
    
    # Process results based on format
    if isinstance(results, list):
        # Handle multiple model results format
        for model_result in results:
            if 'run_types' in model_result:
                # Advanced benchmark format
                for run_type, run_data in model_result['run_types'].items():
                    # Note: We'd need the actual test cases here
                    # This is more complex and would require storing test cases in results
                    pass
    
    print(f"Dataset creation from results file is complex - test cases need to be preserved in results")
    print(f"Consider using --save-dataset option when running benchmarks instead")


def export_dataset_summary(dataset_file: str):
    """Export a summary of dataset contents"""
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return
    
    try:
        with open(dataset_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return
    
    print(f"\nDataset Summary: {os.path.basename(dataset_file)}")
    print("=" * 50)
    
    if isinstance(data, list):
        test_cases = data
    elif isinstance(data, dict) and 'test_cases' in data:
        test_cases = data['test_cases']
    else:
        print("Unknown dataset format")
        return
    
    print(f"Total test cases: {len(test_cases)}")
    
    # Analyze by category
    categories = {}
    run_types = {}
    models = {}
    
    for test_case in test_cases:
        metadata = test_case.get('additional_metadata', {})
        
        # Count by category
        category = metadata.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        
        # Count by run type
        run_type = metadata.get('run_type', 'unknown')
        run_types[run_type] = run_types.get(run_type, 0) + 1
        
        # Count by model
        model = metadata.get('model', 'unknown')
        models[model] = models.get(model, 0) + 1
    
    print(f"\nBy Category:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    
    print(f"\nBy Run Type:")
    for run_type, count in sorted(run_types.items()):
        print(f"  {run_type}: {count}")
    
    print(f"\nBy Model:")
    for model, count in sorted(models.items()):
        print(f"  {model}: {count}")


def merge_datasets(dataset_files: List[str], output_name: str):
    """Merge multiple datasets into one"""
    
    merged_test_cases = []
    
    for dataset_file in dataset_files:
        if not os.path.exists(dataset_file):
            print(f"Warning: Dataset file not found: {dataset_file}")
            continue
        
        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                test_cases = data
            elif isinstance(data, dict) and 'test_cases' in data:
                test_cases = data['test_cases']
            else:
                print(f"Warning: Unknown format in {dataset_file}")
                continue
            
            merged_test_cases.extend(test_cases)
            print(f"Added {len(test_cases)} test cases from {dataset_file}")
            
        except Exception as e:
            print(f"Error reading {dataset_file}: {e}")
            continue
    
    if not merged_test_cases:
        print("No test cases to merge")
        return
    
    # Save merged dataset
    os.makedirs("datasets", exist_ok=True)
    output_file = f"datasets/{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    merged_data = {
        "name": output_name,
        "created": datetime.now().isoformat(),
        "test_cases": merged_test_cases,
        "source_files": dataset_files
    }
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\nMerged dataset saved: {output_file}")
    print(f"Total test cases: {len(merged_test_cases)}")


def main():
    parser = argparse.ArgumentParser(description="DeepEval Dataset Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available datasets')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show dataset summary')
    summary_parser.add_argument('dataset', help='Dataset file to summarize')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple datasets')
    merge_parser.add_argument('datasets', nargs='+', help='Dataset files to merge')
    merge_parser.add_argument('--output', required=True, help='Output dataset name')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create dataset from results')
    create_parser.add_argument('results_file', help='Benchmark results JSON file')
    create_parser.add_argument('--name', required=True, help='Dataset name')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_datasets()
    elif args.command == 'summary':
        export_dataset_summary(args.dataset)
    elif args.command == 'merge':
        merge_datasets(args.datasets, args.output)
    elif args.command == 'create':
        create_dataset_from_results(args.results_file, args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()