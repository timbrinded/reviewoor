#!/usr/bin/env python3
"""
Example 07: Model Selection
This example demonstrates how to use different providers and models.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

# Sample code to review
sample_code = '''
def calculate_average(numbers):
    """Calculate the average of a list of numbers"""
    if not numbers:
        return 0
    
    total = sum(numbers)
    count = len(numbers)
    
    # Potential division by zero if count is 0 (already handled above)
    average = total / count
    
    return average

def find_max(items):
    """Find the maximum value in a list"""
    if not items:
        return None
    
    max_value = items[0]
    for item in items[1:]:
        if item > max_value:
            max_value = item
    
    return max_value
'''

def main():
    print("=== Model Selection Example ===\n")
    
    # Example 1: Default (OpenAI with gpt-4)
    print("1. Using default settings (OpenAI GPT-4):")
    orchestrator = CodeReviewOrchestrator()
    print(f"   Provider: {orchestrator.provider}")
    print(f"   Model: {orchestrator.model}\n")
    
    # Example 2: Explicitly use Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("2. Using Anthropic Claude:")
        orchestrator = CodeReviewOrchestrator(provider="anthropic")
        print(f"   Provider: {orchestrator.provider}")
        print(f"   Model: {orchestrator.model}\n")
    else:
        print("2. Skipping Anthropic example (no API key found)\n")
    
    # Example 3: Use a specific OpenAI model
    print("3. Using specific OpenAI model (gpt-3.5-turbo):")
    orchestrator = CodeReviewOrchestrator(provider="openai", model="gpt-3.5-turbo")
    print(f"   Provider: {orchestrator.provider}")
    print(f"   Model: {orchestrator.model}\n")
    
    # Example 4: Use a specific Anthropic model
    if os.getenv("ANTHROPIC_API_KEY"):
        print("4. Using specific Anthropic model:")
        orchestrator = CodeReviewOrchestrator(
            provider="anthropic", 
            model="claude-3-haiku-20240307"
        )
        print(f"   Provider: {orchestrator.provider}")
        print(f"   Model: {orchestrator.model}\n")
    
    # Example 5: Pass custom API key
    custom_api_key = os.getenv("CUSTOM_OPENAI_KEY")
    if custom_api_key:
        print("5. Using custom API key:")
        orchestrator = CodeReviewOrchestrator(
            provider="openai",
            model="gpt-4-turbo-preview",
            api_key=custom_api_key
        )
        print(f"   Provider: {orchestrator.provider}")
        print(f"   Model: {orchestrator.model}\n")
    else:
        print("5. Skipping custom API key example (CUSTOM_OPENAI_KEY not set)\n")
    
    # Perform a review with the last configured orchestrator
    print("Performing code review with the last configuration...")
    result = orchestrator.review_code_string(sample_code, "sample.py")
    
    print(f"\n✅ Review completed!")
    print(f"   Issues found: {len(result.issues)}")
    print(f"   Model used: {result.model_used}")
    print(f"   Review time: {result.review_time:.2f}s")
    
    # Show available models
    print("\n📋 Available Models:")
    print("\nOpenAI Models:")
    print("  - gpt-4 (default)")
    print("  - gpt-4-turbo-preview")
    print("  - gpt-3.5-turbo")
    print("  - gpt-3.5-turbo-16k")
    print("  - o3-mini")
    print("  - o3 (requires access)")
    print("  - o3-pro (uses completions endpoint)")
    
    print("\nAnthropic Models:")
    print("  - claude-3-5-sonnet-20241022 (default)")
    print("  - claude-3-opus-20240229")
    print("  - claude-3-sonnet-20240229")
    print("  - claude-3-haiku-20240307")
    
    print("\n💡 Tips:")
    print("  - OpenAI is now the default provider")
    print("  - You can specify any model supported by the provider")
    print("  - Models have different speed/quality/cost tradeoffs")
    print("  - GPT-4 and Claude 3.5 Sonnet are best for complex analysis")
    print("  - GPT-3.5-turbo and Claude Haiku are faster and cheaper")

if __name__ == "__main__":
    main()