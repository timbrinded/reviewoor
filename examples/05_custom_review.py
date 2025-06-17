#!/usr/bin/env python3
"""
Example 05: Custom Review with Different Providers
This example shows how to use different AI providers and compare results.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator, AnthropicReviewAgent, OpenAIReviewAgent

# Code with subtle bugs that AI might catch
subtle_bugs_code = '''
class ShoppingCart:
    def __init__(self):
        self.items = []
        self.discount = 0.0
    
    def add_item(self, item, price, quantity=1):
        """Add item to cart"""
        self.items.append({
            'item': item,
            'price': price,
            'quantity': quantity
        })
    
    def remove_item(self, item_name):
        """Remove item from cart"""
        for i, item in enumerate(self.items):
            if item['item'] == item_name:
                del self.items[i]  # Bug: modifying list while iterating
    
    def calculate_total(self):
        """Calculate total with discount"""
        total = 0
        for item in self.items:
            total += item['price'] * item['quantity']
        
        # Bug: discount might be > 1.0
        return total * (1 - self.discount)
    
    def apply_discount(self, discount_percent):
        """Apply discount percentage"""
        self.discount = discount_percent / 100  # No validation

class InventoryManager:
    def __init__(self):
        self.inventory = {}
    
    def add_stock(self, item, quantity):
        """Add stock for an item"""
        if item in self.inventory:
            self.inventory[item] += quantity
        else:
            self.inventory[item] = quantity
    
    def remove_stock(self, item, quantity):
        """Remove stock for an item"""
        if item in self.inventory:
            self.inventory[item] -= quantity  # Bug: can go negative
            if self.inventory[item] == 0:
                del self.inventory[item]
        # Bug: no error if item doesn't exist
    
    def get_stock(self, item):
        """Get current stock for an item"""
        return self.inventory.get(item, 0)

def calculate_fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        # Bug: inefficient recursive implementation
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
'''

def main():
    print("=== Custom Review with Provider Comparison ===\n")
    
    # Check which providers are available
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not has_anthropic and not has_openai:
        print("⚠️  No API keys found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file")
        return
    
    results = {}
    
    # Review with Anthropic if available
    if has_anthropic:
        print("🤖 Reviewing with Anthropic Claude...")
        orchestrator = CodeReviewOrchestrator(provider="anthropic")
        result = orchestrator.review_code_string(subtle_bugs_code, "subtle_bugs.py")
        results["Anthropic"] = result
        print(f"   Found {len(result.issues)} issues")
    
    # Review with OpenAI if available
    if has_openai:
        print("🤖 Reviewing with OpenAI GPT...")
        orchestrator = CodeReviewOrchestrator(provider="openai")
        result = orchestrator.review_code_string(subtle_bugs_code, "subtle_bugs.py")
        results["OpenAI"] = result
        print(f"   Found {len(result.issues)} issues")
    
    # Compare results if we have both
    if len(results) == 2:
        print("\n📊 Comparison of Results:")
        print("-" * 60)
        
        # Compare issue counts
        for provider, result in results.items():
            print(f"\n{provider}:")
            print(f"  Total issues: {len(result.issues)}")
            print(f"  Review time: {result.review_time:.2f}s")
            
            # Count by severity
            sev_counts = {}
            for issue in result.issues:
                sev = issue.severity.value
                sev_counts[sev] = sev_counts.get(sev, 0) + 1
            print(f"  By severity: {sev_counts}")
            
            # Show unique categories
            categories = set(issue.category for issue in result.issues)
            print(f"  Categories: {', '.join(categories)}")
        
        print("\n💡 AI-Specific Issues Found:")
        # Show non-static issues (confidence < 1.0)
        for provider, result in results.items():
            ai_issues = [i for i in result.issues if i.confidence < 1.0]
            if ai_issues:
                print(f"\n{provider} AI insights ({len(ai_issues)} issues):")
                for issue in ai_issues[:3]:  # Show top 3
                    print(f"  - Line {issue.line_number}: {issue.message}")
    
    else:
        # Single provider results
        provider = list(results.keys())[0]
        result = list(results.values())[0]
        
        print(f"\n📊 {provider} Review Results:")
        print("-" * 60)
        
        # Show all issues
        for i, issue in enumerate(result.issues, 1):
            print(f"\n{i}. [{issue.severity.value}] {issue.category}")
            print(f"   Line {issue.line_number}: {issue.message}")
            if issue.suggestion:
                print(f"   → {issue.suggestion}")
        
        print(f"\n📝 Summary: {result.summary}")
        
        if result.metrics:
            print("\n📈 Code Metrics:")
            for metric, score in result.metrics.items():
                print(f"  - {metric}: {score}/10")
    
    # Export results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    for provider, result in results.items():
        filename = os.path.join(output_dir, f"{provider.lower()}_review_results.json")
        orchestrator.export_results(filename)
        print(f"\n💾 {provider} results exported to: {filename}")

if __name__ == "__main__":
    main()