"""Code analysis tools for AI agents to use during code review"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
import json


def check_imports(code: str) -> Dict[str, Any]:
    """Check for unused imports and import issues"""
    try:
        tree = ast.parse(code)
        imports = []
        used_names = set()
        
        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        'name': f"{node.module}.{alias.name}" if node.module else alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        
        # Find unused imports
        unused = []
        for imp in imports:
            name = imp['alias'] or imp['name'].split('.')[-1]
            if name not in used_names:
                unused.append(imp)
        
        return {
            'success': True,
            'total_imports': len(imports),
            'unused_imports': unused,
            'import_lines': [imp['line'] for imp in imports]
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_complexity(code: str) -> Dict[str, Any]:
    """Analyze cyclomatic complexity of functions"""
    try:
        tree = ast.parse(code)
        functions = []
        
        class ComplexityAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.complexity = 0
                
            def visit_FunctionDef(self, node):
                old_func = self.current_function
                old_complexity = self.complexity
                
                self.current_function = node.name
                self.complexity = 1  # Base complexity
                
                # Visit function body
                self.generic_visit(node)
                
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'complexity': self.complexity,
                    'is_complex': self.complexity > 10
                })
                
                self.current_function = old_func
                self.complexity = old_complexity
                
            def visit_If(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
                
            def visit_ExceptHandler(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
        
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)
        
        return {
            'success': True,
            'functions': functions,
            'complex_functions': [f for f in functions if f['is_complex']],
            'max_complexity': max([f['complexity'] for f in functions], default=0)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def find_security_patterns(code: str) -> Dict[str, Any]:
    """Find potential security vulnerabilities using regex patterns"""
    patterns = {
        'hardcoded_secret': [
            (r'(api_key|API_KEY|password|PASSWORD|secret|SECRET)\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
            (r'(token|TOKEN)\s*=\s*["\'][^"\']+["\']', 'Hardcoded token detected')
        ],
        'sql_injection': [
            (r'(query|sql)\s*=\s*[fF]?["\'].*\{.*\}', 'Potential SQL injection via f-string'),
            (r'(query|sql)\s*=\s*.*\+\s*[^"\']', 'Potential SQL injection via string concatenation')
        ],
        'command_injection': [
            (r'subprocess\.(call|run|Popen).*shell\s*=\s*True', 'Subprocess with shell=True is dangerous'),
            (r'os\.system\s*\(', 'os.system usage is dangerous')
        ],
        'insecure_random': [
            (r'random\.(random|randint|choice)\s*\(', 'Using random module for security is insecure')
        ],
        'eval_usage': [
            (r'\beval\s*\(', 'eval() usage is dangerous'),
            (r'\bexec\s*\(', 'exec() usage is dangerous')
        ]
    }
    
    findings = []
    
    for category, pattern_list in patterns.items():
        for pattern, message in pattern_list:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                findings.append({
                    'category': category,
                    'message': message,
                    'line': line_num,
                    'match': match.group(0)
                })
    
    return {
        'success': True,
        'total_findings': len(findings),
        'findings': findings,
        'categories_found': list(set(f['category'] for f in findings))
    }


def check_type_consistency(code: str) -> Dict[str, Any]:
    """Check for type consistency issues"""
    try:
        tree = ast.parse(code)
        issues = []
        
        class TypeChecker(ast.NodeVisitor):
            def __init__(self):
                self.variables = {}
                
            def visit_Assign(self, node):
                # Track variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        value_type = self._infer_type(node.value)
                        
                        if var_name in self.variables and self.variables[var_name] != value_type:
                            issues.append({
                                'variable': var_name,
                                'line': node.lineno,
                                'issue': f'Type changed from {self.variables[var_name]} to {value_type}'
                            })
                        
                        self.variables[var_name] = value_type
                
                self.generic_visit(node)
            
            def _infer_type(self, node):
                if isinstance(node, ast.Constant):
                    return type(node.value).__name__
                elif isinstance(node, ast.List):
                    return 'list'
                elif isinstance(node, ast.Dict):
                    return 'dict'
                elif isinstance(node, ast.Set):
                    return 'set'
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        return f'{node.func.id}()'
                return 'unknown'
        
        checker = TypeChecker()
        checker.visit(tree)
        
        return {
            'success': True,
            'type_changes': issues,
            'total_issues': len(issues)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def detect_code_smells(code: str) -> Dict[str, Any]:
    """Detect common code smells"""
    smells = []
    
    # Long lines
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            smells.append({
                'type': 'long_line',
                'line': i,
                'message': f'Line too long ({len(line)} chars)',
                'severity': 'low'
            })
    
    # TODO comments
    for i, line in enumerate(lines, 1):
        if 'TODO' in line or 'FIXME' in line or 'HACK' in line:
            smells.append({
                'type': 'todo_comment',
                'line': i,
                'message': 'Unresolved TODO/FIXME/HACK comment',
                'severity': 'medium'
            })
    
    # Magic numbers
    magic_number_pattern = r'(?<!["\'])\b(?!0\b|1\b|2\b)\d{2,}\b(?!["\'])'
    for match in re.finditer(magic_number_pattern, code):
        line_num = code[:match.start()].count('\n') + 1
        smells.append({
            'type': 'magic_number',
            'line': line_num,
            'message': f'Magic number {match.group(0)} should be a named constant',
            'severity': 'low'
        })
    
    # Dead code (commented out code blocks)
    commented_code_pattern = r'^\s*#\s*(if|for|while|def|class|import|from)\s'
    for i, line in enumerate(lines, 1):
        if re.match(commented_code_pattern, line):
            smells.append({
                'type': 'dead_code',
                'line': i,
                'message': 'Commented out code should be removed',
                'severity': 'low'
            })
    
    return {
        'success': True,
        'code_smells': smells,
        'total_smells': len(smells),
        'smell_types': list(set(s['type'] for s in smells))
    }


def get_function_metrics(code: str) -> Dict[str, Any]:
    """Get metrics for all functions in the code"""
    try:
        tree = ast.parse(code)
        metrics = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines
                last_line = node.lineno
                for child in ast.walk(node):
                    if hasattr(child, 'lineno'):
                        last_line = max(last_line, child.lineno)
                
                lines_of_code = last_line - node.lineno + 1
                
                # Count parameters
                num_params = len(node.args.args)
                
                # Count return statements
                returns = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
                
                # Check for docstring
                has_docstring = (
                    node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)
                )
                
                metrics.append({
                    'name': node.name,
                    'line': node.lineno,
                    'lines_of_code': lines_of_code,
                    'parameters': num_params,
                    'returns': returns,
                    'has_docstring': has_docstring,
                    'is_too_long': lines_of_code > 50,
                    'too_many_params': num_params > 5
                })
        
        return {
            'success': True,
            'functions': metrics,
            'total_functions': len(metrics),
            'functions_without_docstring': [f for f in metrics if not f['has_docstring']],
            'long_functions': [f for f in metrics if f['is_too_long']]
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# Tool definitions for AI agents
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_imports",
            "description": "Analyze imports in the code to find unused imports and import-related issues",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to analyze"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_complexity",
            "description": "Analyze the cyclomatic complexity of functions in the code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to analyze"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_security_patterns",
            "description": "Scan code for potential security vulnerabilities using pattern matching",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to scan"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_type_consistency",
            "description": "Check for type consistency issues and type changes in variables",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to analyze"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_code_smells",
            "description": "Detect common code smells like long lines, TODOs, magic numbers, and dead code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to analyze"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_function_metrics",
            "description": "Get detailed metrics for all functions including lines of code, parameters, and documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to analyze"
                    }
                },
                "required": ["code"]
            }
        }
    }
]