# SOTA Wordle Solver

A state-of-the-art Wordle solver that runs efficiently on local hardware. Achieves near-optimal performance (~3.45-3.50 average guesses) without requiring heavy computation or training.

## Features

- **CSP + Entropy + Endgame Lookahead**: Three-stage algorithm combining constraint satisfaction, information theory, and bounded search
- **Fast**: Full benchmark over 2315 words completes in ~1-2 minutes
- **Hard Mode Support**: Respects hard mode constraints
- **Interactive Mode**: Get suggestions while playing Wordle
- **Comprehensive Benchmarking**: Compare solver variants and track statistics

## Quick Start

```bash
# Test solver on specific words
python -m src.cli test crane jazzy query

# Watch the solver play
python -m src.cli play --word crane

# Interactive mode (get suggestions while you play)
python -m src.cli solve

# Run full benchmark
python -m src.cli benchmark
```

## Installation

```bash
# Clone the repository
cd wordle-solver

# No external dependencies required for basic usage!
# For benchmarking, you may want numpy for speedups:
pip install numpy
```

## Usage

### Interactive Solving

Use the solver to get suggestions while playing Wordle:

```bash
python -m src.cli solve
```

Enter feedback as:
- **Emoji**: `🟩🟨⬛⬛🟨`
- **Letters**: `GYBBY` (G=green, Y=yellow, B=black)
- **Numbers**: `21001` (2=green, 1=yellow, 0=gray)

### Watch the Solver Play

```bash
# Random word
python -m src.cli play

# Specific word
python -m src.cli play --word jazzy

# Hide the target word (for suspense)
python -m src.cli play --hide
```

### Run Benchmarks

```bash
# Benchmark the default (best) solver
python -m src.cli benchmark

# Compare all solver variants
python -m src.cli benchmark --all

# Quick test (100 words)
python -m src.cli benchmark --limit 100

# Save results
python -m src.cli benchmark -o results.csv
```

### Analyze Starting Words

```bash
# Find best starting words by entropy
python -m src.cli analyze

# Show top 50 and benchmark them
python -m src.cli analyze --top 50 --benchmark
```

## Algorithm Overview

### Phase 1: Entropy-Based Selection
For each possible guess, compute the expected information gain (entropy) by analyzing how the guess partitions remaining candidates. Higher entropy = more information gained on average.

### Phase 2: CSP Constraint Propagation
Maintain a constraint state tracking:
- Fixed positions (green letters)
- Forbidden positions (yellow letters)
- Letter count bounds (from duplicate handling)

Apply constraints BEFORE entropy calculation for better accuracy.

### Phase 3: Endgame Lookahead
When few candidates remain (≤12), use bounded minimax search to find the optimal guess. This handles worst-case scenarios that pure entropy misses.

## Performance

| Solver Variant | Mean Guesses | Max Guesses | Runtime |
|----------------|--------------|-------------|---------|
| Entropy Only   | ~3.60        | 6           | ~30s    |
| CSP-Entropy    | ~3.50        | 6           | ~45s    |
| CSP + Endgame  | ~3.45        | 5-6         | ~90s    |

Target performance: **3.45-3.50 average guesses** (near SOTA without heavy compute)

## Project Structure

```
wordle-solver/
├── src/
│   ├── __init__.py
│   ├── feedback.py          # Core feedback computation
│   ├── constraints.py       # CSP constraint propagation
│   ├── entropy_solver.py    # Entropy-based solver
│   ├── csp_entropy_solver.py # CSP-aware entropy solver
│   ├── endgame_search.py    # Bounded lookahead search
│   ├── solver.py            # Unified solver interface
│   ├── benchmark.py         # Benchmarking system
│   └── cli.py               # Command-line interface
├── words/
│   ├── answers.txt          # 2315 possible answers
│   └── start_words.json     # Precomputed best starters
└── README.md
```

## API Usage

```python
from src.solver import WordleSolver, SolverType

# Create solver
solver = WordleSolver(
    solver_type=SolverType.ENDGAME,  # Best performance
    hard_mode=False
)

# Auto-solve a word
guesses, solved = solver.solve("crane")
print(f"Solved in {len(guesses)} guesses: {guesses}")

# Interactive usage
solver.reset()
guess = solver.next_guess()  # Get suggestion
feedback = solver.parse_feedback("🟩🟨⬛⬛🟨")  # Parse result
solver.update(guess, feedback)  # Update state
```

## Hard Mode

The solver fully supports Wordle's hard mode:

```bash
python -m src.cli solve --hard
python -m src.cli benchmark --hard
```

In hard mode, all discovered hints must be used in subsequent guesses.

## License

MIT License - feel free to use and modify.

## Acknowledgments

- Word lists from the original Wordle game
- Algorithm inspired by information theory and constraint satisfaction research
