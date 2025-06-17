#!/usr/bin/env python3
"""
Example Python file with various code issues for testing the review agent.
This file intentionally contains bugs, bad practices, and style issues.
"""

import os
import random
import pickle
from typing import List, Dict, Any

# Global variables (bad practice)
GLOBAL_COUNTER = 0
USER_DATA = []

class UserManager:
    """Manages user data and operations"""
    
    def __init__(self, admin_password="admin123"):  # Hardcoded password
        self.users = {}
        self.admin_password = admin_password
        self.session_tokens = []
    
    def add_user(self, username, password, email=None, data=[]):  # Mutable default
        """Add a new user to the system"""
        if username in self.users:
            return False
        
        self.users[username] = {
            'password': password,  # Storing password in plaintext
            'email': email,
            'data': data,
            'created_at': str(random.random())  # Weak randomness
        }
        return True
    
    def authenticate(self, username, password):
        # Missing docstring
        user = self.users.get(username)
        if user == None:  # Should use 'is None'
            return False
        
        # Timing attack vulnerability
        if user['password'] == password:
            token = self.generate_token()
            self.session_tokens.append(token)
            return token
        return False
    
    def generate_token(self):
        """Generate session token"""
        # Weak token generation
        return str(random.randint(1000, 9999))
    
    def delete_user(self, username):
        """Delete a user"""
        try:
            del self.users[username]
        except:  # Bare except
            pass
    
    def get_all_users(self):
        """Return all users"""
        return self.users  # Exposing internal data structure

def process_data(data_list):
    """Process a list of data items"""
    result = []
    
    for i in range(len(data_list)):  # Could use enumerate
        item = data_list[i]
        
        # Very long line that exceeds recommended character limits and makes the code harder to read and maintain in most editors and terminals
        
        if type(item) == str:  # Should use isinstance
            processed = eval(f"item.{item.lower()}()")  # Dangerous eval!
            result.append(processed)
        elif type(item) == int:
            result.append(item ** 2)
    
    average = sum(result) / len(result)  # No check for empty list
    return average

class DataProcessor:
    def __init__(self):
        self.cache = {}
    
    def process(self, input_data):
        # Missing docstring and type hints
        if input_data in self.cache:
            return self.cache[input_data]
        
        # SQL injection vulnerability
        query = f"SELECT * FROM data WHERE id = '{input_data}'"
        
        # Simulating some processing
        result = input_data.upper() if isinstance(input_data, str) else input_data
        
        self.cache[input_data] = result
        return result
    
    def save_state(self, filename):
        """Save processor state"""
        # Insecure deserialization vulnerability
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load_state(self, filename):
        """Load processor state"""
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)  # Dangerous!

def calculate_fibonacci(n):
    """Calculate nth Fibonacci number recursively"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        # Inefficient recursive implementation
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def unsafe_file_operation(user_input):
    """Perform file operation based on user input"""
    # Path traversal vulnerability
    filepath = f"/tmp/{user_input}"
    
    # Command injection vulnerability
    os.system(f"cat {filepath}")
    
    with open(filepath, 'r') as f:
        return f.read()

# Function with complexity issues
def complex_function(data, options=None):
    """A function that's too complex and should be refactored"""
    if options is None:
        options = {}
    
    result = []
    errors = []
    processed = 0
    
    for item in data:
        try:
            if 'skip_invalid' in options and options['skip_invalid']:
                if not item:
                    continue
            
            if isinstance(item, dict):
                if 'id' in item:
                    if item['id'] > 0:
                        if 'name' in item:
                            if len(item['name']) > 0:
                                result.append(item)
                                processed += 1
                            else:
                                errors.append("Empty name")
                        else:
                            errors.append("Missing name")
                    else:
                        errors.append("Invalid ID")
                else:
                    errors.append("Missing ID")
            elif isinstance(item, str):
                if item:
                    result.append({'name': item, 'id': processed})
                    processed += 1
            else:
                errors.append(f"Unknown type: {type(item)}")
        except Exception as e:
            errors.append(str(e))
    
    return {
        'results': result,
        'errors': errors,
        'processed': processed
    }

# Global code execution (bad practice)
print("Module loaded")  # Should not have side effects on import
GLOBAL_COUNTER += 1

if __name__ == "__main__":
    # Example usage with issues
    manager = UserManager()
    manager.add_user("admin", "password123")  # Weak password
    
    # Using global variables
    USER_DATA.append(manager)
    
    # Risky operations
    token = manager.authenticate("admin", "password123")
    print(f"Token: {token}")  # Logging sensitive information
    
    # Inefficient operation
    fib_result = calculate_fibonacci(35)  # This will be very slow
    print(f"Fibonacci(35) = {fib_result}")