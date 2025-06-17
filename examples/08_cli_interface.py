#!/usr/bin/env python3
"""
Example 08: Command-Line Interface
This example shows how to use runtime arguments to select providers and models.
"""

import os
import sys
import agentops
import argparse
from dotenv import load_dotenv

# Load .env file from parent directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

def show_usage_error(message):
    """Show error message and usage examples"""
    print(f"❌ Error: {message}\n")
    print("Usage examples:")
    print("  # Use default (OpenAI GPT-4):")
    print("  python 08_cli_interface.py")
    print("  # Use Anthropic Claude:")
    print("  python 08_cli_interface.py --anthropic")
    print("  # Use specific model:")
    print("  python 08_cli_interface.py --model gpt-3.5-turbo")
    print("\nFor more help: python 08_cli_interface.py --help")

# Sample code to review
sample_code = '''
import requests

def fetch_data(url):
    """Fetch data from a URL"""
    try:
        response = requests.get(url, timeout=30)
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_api_response(data):
    """Process API response data"""
    if not data:
        return []
    
    results = []
    for item in data.get('items', []):
        if 'id' in item and 'name' in item:
            results.append({
                'id': item['id'],
                'name': item['name'],
                'value': item.get('value', 0)
            })
    
    return results
'''

def main():
    # Initialize AgentOps
    AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") or 'a0aac5f4-2b60-43c2-bb9e-f79247f5a2dc'
    
    parser = argparse.ArgumentParser(
        description='Code Review with custom provider and model selection',
        epilog='''
Examples:
  # Use default (OpenAI GPT-4):
  %(prog)s
  
  # Use Anthropic Claude:
  %(prog)s --anthropic
  
  # Use specific model:
  %(prog)s --model gpt-3.5-turbo
  %(prog)s -p anthropic -m claude-3-haiku-20240307
  
  # Review a specific file:
  %(prog)s --file ../src/agents/base.py
  %(prog)s -a -f ../example.py
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--provider', '-p',
        choices=['openai', 'anthropic'],
        default='openai',
        help='AI provider to use (default: openai)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model to use (e.g., gpt-4, gpt-3.5-turbo, claude-3-5-sonnet-20241022)'
    )
    parser.add_argument(
        '--anthropic', '-a',
        action='store_true',
        help='Use Anthropic instead of OpenAI (shorthand for --provider anthropic)'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='File to review (optional, uses sample code if not provided)'
    )
    
    args = parser.parse_args()
    
    # Handle --anthropic flag
    if args.anthropic:
        args.provider = 'anthropic'
    
    # Initialize AgentOps with appropriate tags
    provider_tag = f"{args.provider} code review"
    model_tag = args.model if args.model else f"{args.provider} default"
    agentops.init(
        api_key=AGENTOPS_API_KEY,
        default_tags=[provider_tag, model_tag, 'cli']
    )
    
    print(f"=== Code Review CLI Example ===\n")
    
    # Check for API keys
    if args.provider == 'anthropic' and not os.getenv("ANTHROPIC_API_KEY"):
        show_usage_error("ANTHROPIC_API_KEY not found in environment. Please set it in your .env file.")
        agentops.end_trace(end_state="Error")
        return 1
    elif args.provider == 'openai' and not os.getenv("OPENAI_API_KEY"):
        show_usage_error("OPENAI_API_KEY not found in environment. Please set it in your .env file.")
        agentops.end_trace(end_state="Error")
        return 1
    
    # Initialize orchestrator with specified options
    orchestrator_args = {'provider': args.provider}
    if args.model:
        orchestrator_args['model'] = args.model
    
    try:
        orchestrator = CodeReviewOrchestrator(**orchestrator_args)
        print(f"✅ Initialized with:")
        print(f"   Provider: {orchestrator.provider}")
        print(f"   Model: {orchestrator.model}\n")
    except ValueError as e:
        show_usage_error(str(e))
        agentops.end_session("Failure")
        return 1
    
    # Determine what to review
    if args.file:
        if not os.path.exists(args.file):
            show_usage_error(f"File not found: {args.file}")
            agentops.end_session("Failure")
            return 1
        
        print(f"Reviewing file: {args.file}")
        result = orchestrator.review_file(args.file)
    else:
        print("Reviewing sample code...")
        result = orchestrator.review_code_string(sample_code, "sample.py")
    
    # Display results
    print(f"\n📊 Review Results:")
    print(f"   Total issues: {len(result.issues)}")
    print(f"   Review time: {result.review_time:.2f}s")
    print(f"   Model used: {result.model_used}\n")
    
    # Group by severity
    by_severity = {}
    for issue in result.issues:
        sev = issue.severity.value
        if sev not in by_severity:
            by_severity[sev] = 0
        by_severity[sev] += 1
    
    if by_severity:
        print("Issues by severity:")
        for sev in ["critical", "high", "medium", "low", "info"]:
            if sev in by_severity:
                print(f"   {sev}: {by_severity[sev]}")
    
    # Show metrics if available
    if result.metrics:
        print("\nCode metrics:")
        for metric, value in result.metrics.items():
            print(f"   {metric}: {value}/10")
    
    # End AgentOps trace
    agentops.end_trace(end_state="Success")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())