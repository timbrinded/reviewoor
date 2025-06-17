#!/usr/bin/env python3
"""
Example 06: Security-Focused Review
This example demonstrates reviewing code specifically for security vulnerabilities.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CodeReviewOrchestrator

# Code with various security vulnerabilities
insecure_code = '''
import os
import sqlite3
import pickle
import subprocess
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Hardcoded credentials
DATABASE_URL = "postgresql://admin:password123@localhost/mydb"
API_KEY = "sk-prod-1234567890abcdef"
SECRET_KEY = "my-secret-key-123"

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # SQL Injection vulnerability
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user:
        return "Login successful"
    return "Invalid credentials"

@app.route('/search')
def search():
    # XSS vulnerability
    query = request.args.get('q', '')
    return f"<h1>Search results for: {query}</h1>"

@app.route('/render')
def render():
    # Template injection vulnerability
    template = request.args.get('template', '')
    return render_template_string(template)

@app.route('/execute')
def execute():
    # Command injection vulnerability
    cmd = request.args.get('cmd', 'ls')
    result = os.system(cmd)
    return f"Command executed: {result}"

@app.route('/subprocess')
def run_subprocess():
    # Another command injection
    user_input = request.args.get('input', '')
    result = subprocess.check_output(f"echo {user_input}", shell=True)
    return result

def load_user_data(filename):
    # Insecure deserialization
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_file(content, filename):
    # Path traversal vulnerability
    filepath = f"/uploads/{filename}"
    with open(filepath, 'w') as f:
        f.write(content)

class UserSession:
    def __init__(self):
        self.logged_in = False
        self.user_id = None
    
    def authenticate(self, username, password):
        # Timing attack vulnerability
        stored_password = get_password_from_db(username)
        if password == stored_password:
            self.logged_in = True
            return True
        return False

def get_password_from_db(username):
    # Stub function
    return "password123"

# Weak random number generation
import random
def generate_token():
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

# Logging sensitive information
import logging
def process_payment(card_number, cvv):
    logging.info(f"Processing payment for card {card_number} with CVV {cvv}")
    # Process payment...
'''

def main():
    print("=== Security-Focused Code Review ===\n")
    print("This example reviews code specifically for security vulnerabilities.\n")
    
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No API key found. Running static analysis only...")
        print("   (AI review would find more sophisticated security issues)\n")
        
        from src import CodeAnalyzer
        analyzer = CodeAnalyzer()
        issues = analyzer.analyze(insecure_code)
        
        security_issues = [i for i in issues if "security" in i.category.lower() or "security" in i.message.lower()]
        print(f"Static analysis found {len(security_issues)} security-related issues")
        
        for issue in security_issues:
            print(f"\n[{issue.severity.value}] Line {issue.line_number}: {issue.message}")
        return
    
    # Use AI-powered review
    provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    orchestrator = CodeReviewOrchestrator(provider=provider)
    
    print(f"🔍 Performing security review with {provider}...")
    result = orchestrator.review_code_string(insecure_code, "insecure_app.py")
    
    # Filter for security issues
    security_issues = []
    for issue in result.issues:
        # Check if it's security-related
        if any(term in issue.message.lower() or term in issue.category.lower() 
               for term in ['security', 'injection', 'vulnerability', 'hardcoded', 'password', 
                           'secret', 'api key', 'sql', 'xss', 'csrf', 'authentication']):
            security_issues.append(issue)
    
    print(f"\n🚨 Found {len(security_issues)} security vulnerabilities:\n")
    
    # Group by severity
    by_severity = {}
    for issue in security_issues:
        sev = issue.severity.value
        if sev not in by_severity:
            by_severity[sev] = []
        by_severity[sev].append(issue)
    
    # Display security issues by severity
    for severity in ["critical", "high", "medium", "low"]:
        if severity in by_severity:
            print(f"\n{severity.upper()} Security Issues ({len(by_severity[severity])}):")
            print("=" * 60)
            
            for issue in by_severity[severity]:
                print(f"\n📍 Line {issue.line_number}: {issue.message}")
                if issue.suggestion:
                    print(f"   ✅ Fix: {issue.suggestion}")
    
    # Security summary
    print("\n📊 Security Review Summary:")
    print("-" * 60)
    print(f"Total security issues: {len(security_issues)}")
    
    vuln_types = {}
    for issue in security_issues:
        # Categorize vulnerability types
        msg_lower = issue.message.lower()
        if 'sql injection' in msg_lower:
            vuln_type = 'SQL Injection'
        elif 'command injection' in msg_lower or 'os.system' in msg_lower:
            vuln_type = 'Command Injection'
        elif 'xss' in msg_lower or 'cross-site' in msg_lower:
            vuln_type = 'XSS'
        elif 'hardcoded' in msg_lower or 'password' in msg_lower or 'api key' in msg_lower:
            vuln_type = 'Hardcoded Secrets'
        elif 'deserialization' in msg_lower or 'pickle' in msg_lower:
            vuln_type = 'Insecure Deserialization'
        elif 'path traversal' in msg_lower:
            vuln_type = 'Path Traversal'
        else:
            vuln_type = 'Other'
        
        vuln_types[vuln_type] = vuln_types.get(vuln_type, 0) + 1
    
    print("\nVulnerability Types Found:")
    for vuln_type, count in sorted(vuln_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {vuln_type}: {count}")
    
    # Overall metrics
    if result.metrics and 'security_score' in result.metrics:
        print(f"\n🛡️  Security Score: {result.metrics['security_score']}/10")
    
    print("\n💡 Security Best Practices:")
    print("  - Never hardcode credentials or secrets")
    print("  - Use parameterized queries to prevent SQL injection")
    print("  - Validate and sanitize all user input")
    print("  - Use secure random number generation")
    print("  - Implement proper authentication and authorization")
    print("  - Avoid dangerous functions like eval(), exec(), pickle.load()")
    
    # Export results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "security_review_results.json")
    orchestrator.export_results(output_file)
    print(f"\n💾 Full security review exported to: {output_file}")

if __name__ == "__main__":
    main()