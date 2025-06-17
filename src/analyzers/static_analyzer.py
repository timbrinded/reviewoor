"""Static code analyzer for Python."""

import ast
import re
from typing import List

from ..models import CodeIssue, Severity


class CodeAnalyzer:
    """Static code analyzer for Python - provides baseline analysis"""
    
    def __init__(self):
        self.issues = []
    
    def analyze(self, code: str) -> List[CodeIssue]:
        """Perform static analysis on Python code"""
        self.issues = []
        
        # Try to parse the code
        try:
            tree = ast.parse(code)
            self._analyze_ast(tree, code)
        except SyntaxError as e:
            self.issues.append(CodeIssue(
                severity=Severity.CRITICAL,
                category="Syntax Error",
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column=e.offset
            ))
            return self.issues
        
        # Analyze code patterns
        self._check_common_issues(code)
        self._check_style_issues(code)
        
        return self.issues
    
    def _analyze_ast(self, tree: ast.AST, code: str):
        """Analyze the AST for potential issues"""
        class IssueVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    self.analyzer.issues.append(CodeIssue(
                        severity=Severity.LOW,
                        category="Documentation",
                        message=f"Function '{node.name}' is missing a docstring",
                        line_number=node.lineno
                    ))
                
                # Check function complexity (simplified)
                if len(node.body) > 50:
                    self.analyzer.issues.append(CodeIssue(
                        severity=Severity.MEDIUM,
                        category="Complexity",
                        message=f"Function '{node.name}' is too long ({len(node.body)} statements)",
                        line_number=node.lineno,
                        suggestion="Consider breaking this function into smaller functions"
                    ))
                
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_Try(self, node):
                # Check for bare except
                for handler in node.handlers:
                    if handler.type is None:
                        self.analyzer.issues.append(CodeIssue(
                            severity=Severity.HIGH,
                            category="Exception Handling",
                            message="Bare except clause catches all exceptions",
                            line_number=handler.lineno,
                            suggestion="Specify the exception types you want to catch"
                        ))
                self.generic_visit(node)
            
            def visit_Compare(self, node):
                # Check for == None instead of is None
                for i, op in enumerate(node.ops):
                    if isinstance(op, ast.Eq) and i < len(node.comparators):
                        comparator = node.comparators[i]
                        if isinstance(comparator, ast.Constant) and comparator.value is None:
                            self.analyzer.issues.append(CodeIssue(
                                severity=Severity.MEDIUM,
                                category="Code Style",
                                message="Use 'is None' instead of '== None'",
                                line_number=node.lineno,
                                suggestion="Replace '== None' with 'is None'"
                            ))
                self.generic_visit(node)
        
        visitor = IssueVisitor(self)
        visitor.visit(tree)
    
    def _check_common_issues(self, code: str):
        """Check for common code issues using regex patterns"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for hardcoded passwords
            if re.search(r'(password|passwd|pwd)\s*=\s*["\'](?!.*\{)', line, re.IGNORECASE):
                self.issues.append(CodeIssue(
                    severity=Severity.CRITICAL,
                    category="Security",
                    message="Possible hardcoded password detected",
                    line_number=i,
                    suggestion="Use environment variables or secure credential storage"
                ))
            
            # Check for eval usage
            if re.search(r'\beval\s*\(', line):
                self.issues.append(CodeIssue(
                    severity=Severity.HIGH,
                    category="Security",
                    message="Use of eval() is dangerous and should be avoided",
                    line_number=i,
                    suggestion="Consider using ast.literal_eval() or alternative approaches"
                ))
            
            # Check for mutable default arguments
            if re.match(r'^\s*def\s+\w+\s*\(.*=\s*(\[|\{)', line):
                self.issues.append(CodeIssue(
                    severity=Severity.HIGH,
                    category="Bug Risk",
                    message="Mutable default argument detected",
                    line_number=i,
                    suggestion="Use None as default and create the mutable object inside the function"
                ))
    
    def _check_style_issues(self, code: str):
        """Check for PEP 8 style issues (simplified)"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Line length
            if len(line) > 120:
                self.issues.append(CodeIssue(
                    severity=Severity.LOW,
                    category="Style",
                    message=f"Line too long ({len(line)} > 120 characters)",
                    line_number=i
                ))
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                self.issues.append(CodeIssue(
                    severity=Severity.LOW,
                    category="Style",
                    message="Trailing whitespace",
                    line_number=i
                ))