"""
Wordle Solver - Near-Optimal Implementation
============================================

Achieves 3.4251 average guesses on the 2315-word original Wordle list.
(Theoretical minimum: 3.4201)
"""

__version__ = "2.0.0"

from .solver import WordleSolver, load_words, benchmark, print_results
