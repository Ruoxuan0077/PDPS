"""
Core execution components.
Provides parallel execution and single-GPU runner.
"""
from .executor import execute
from .runner import Runner

__all__ = ['execute', 'Runner']

