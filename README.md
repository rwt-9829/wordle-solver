# Wordle Solver

A near-optimal Wordle solver achieving **3.4251 average guesses** on the original 2315-word Wordle list. This is within 0.005 of the theoretical minimum (3.4201).

## Performance

| Metric | Value |
|--------|-------|
| **Average Guesses** | 3.4251 |
| **Total Guesses** | 7929 (on 2315 words) |
| **Failures** | 0 |
| **Best First Guess** | SALET |

**Distribution:**
- 2 guesses: 78 words (3.4%)
- 3 guesses: 1219 words (52.7%)
- 4 guesses: 976 words (42.2%)
- 5 guesses: 40 words (1.7%)
- 6 guesses: 2 words (0.1%)

## Installation

```bash
cd wordle-solver
pip install -r requirements.txt
```

## Quick Start

```python
from src.solver import WordleSolver, load_words

# Load word lists
answers = load_words("words/answers.txt")
guesses = load_words("words/allowed_guesses.txt")

# Create solver
solver = WordleSolver(answers, guesses, first_guess="salet")

# Solve a word
n_guesses, guess_list = solver.solve("crane", verbose=True)
print(f"Solved in {n_guesses} guesses: {' -> '.join(guess_list)}")
```

## Run Benchmark

```bash
python -m src.solver
```

Or in Python:

```python
from src.solver import WordleSolver, load_words, benchmark, print_results

answers = load_words("words/answers.txt")
guesses = load_words("words/allowed_guesses.txt")
solver = WordleSolver(answers, guesses, first_guess="salet")

results = benchmark(solver, answers)
print_results(results)
```

## Algorithm

The solver uses several key optimizations:

1. **Exhaustive Distinguishing Search**: For any candidate set, scans all 14,855 valid guesses to find the one that maximizes distinct feedback patterns
2. **Precomputed Decision Tree**: Optimal second and third guesses are precomputed for all possible game states after SALET
3. **Candidate Preference**: When multiple guesses have equal distinguishing power, prefers guesses that could be the answer
4. **Numba JIT Compilation**: Feedback matrix computation is parallelized and JIT-compiled for speed
5. **Disk Caching**: Feedback matrix and precomputed guesses are cached to `cache/` for instant loading on subsequent runs (~0.3s vs ~4s)

## Project Structure

```
wordle-solver/
├── src/
│   ├── __init__.py      # Package exports
│   └── solver.py        # Main solver implementation
├── cache/               # Auto-generated cache files (gitignored)
├── words/
│   ├── answers.txt      # 2315 possible answers (original Wordle)
│   └── allowed_guesses.txt  # 14855 valid guesses
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- NumPy
- Numba

## Theory

The theoretical minimum average for Wordle with the original 2315-word list is **3.4201** (7920 total guesses), proven by Alex Selby. This solver achieves 3.4251 (7929 guesses), just 9 guesses above optimal.

Reaching the exact theoretical minimum requires a full decision tree optimization, which is computationally expensive but could be added in the future.

## License

MIT
