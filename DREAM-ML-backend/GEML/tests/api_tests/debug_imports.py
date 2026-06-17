#!/usr/bin/env python3
"""
Debug script to help identify import path issues.
Run this from DREAM-ML-backend/GEML/api/tests/ directory.
"""

import os
import sys

def debug_paths():
    """Debug the current path setup."""
    print("=== DEBUG PATH INFORMATION ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print()
    
    print("Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    print()
    
    # Check directory structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current dir: {current_dir}")
    
    api_dir = os.path.dirname(current_dir)
    print(f"API dir: {api_dir}")
    
    geml_dir = os.path.dirname(api_dir)
    print(f"GEML dir: {geml_dir}")
    
    project_root = os.path.dirname(geml_dir)
    print(f"Project root: {project_root}")
    print()
    
    # Check for key files
    files_to_check = [
        os.path.join(geml_dir, '__init__.py'),
        os.path.join(api_dir, '__init__.py'),
        os.path.join(api_dir, 'views.py'),
        os.path.join(project_root, 'manage.py'),
    ]
    
    print("Checking for key files:")
    for file_path in files_to_check:
        exists = "✓" if os.path.exists(file_path) else "✗"
        print(f"  {exists} {file_path}")
    print()

def test_imports():
    """Test different import strategies."""
    print("=== TESTING IMPORTS ===")
    
    # Add paths to sys.path for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.dirname(current_dir)
    geml_dir = os.path.dirname(api_dir)
    project_root = os.path.dirname(geml_dir)
    
    # Add paths if not already present
    for path in [project_root, geml_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    import_strategies = [
        ("from GEML.api import views", lambda: __import__('GEML.api.views', fromlist=['views'])),
        ("from api import views", lambda: __import__('api.views', fromlist=['views'])),
        ("import views", lambda: __import__('views')),
    ]
    
    for strategy_name, import_func in import_strategies:
        try:
            module = import_func()
            print(f"✓ SUCCESS: {strategy_name}")
            print(f"    Module: {module}")
            print(f"    File: {getattr(module, '__file__', 'N/A')}")
        except Exception as e:
            print(f"✗ FAILED: {strategy_name}")
            print(f"    Error: {e}")
        print()

if __name__ == '__main__':
    debug_paths()
    test_imports()