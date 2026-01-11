"""
Optimal Wordle Solver
=====================

A high-performance Wordle solver for the REAL NYT Wordle game.

Wordle uses TWO word lists:
- Answers (~3158 words): Curated list of possible daily answers
- Allowed guesses (~14855 words): All valid 5-letter guesses

Key insights from Alex Selby & Jonathan Olson:
- TARSE is optimal first guess for current NYT list (~3.55 avg)
- SALET was optimal for the original pre-NYT list (~3.42 avg)
- Score guesses by: expected partition size + worst-case partition size

Performance targets:
- Current NYT list (3158 words): ~3.55 average guesses
- Original list (2315 words): ~3.42 average guesses
"""

import numpy as np
from numba import jit, prange
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import time
import os


# ============================================================================
# CONSTANTS
# ============================================================================

GRAY = 0
YELLOW = 1
GREEN = 2
CORRECT_PATTERN = 242  # 2 + 2*3 + 2*9 + 2*27 + 2*81 = 242 (all green)
N_PATTERNS = 243  # 3^5 possible feedback patterns


# ============================================================================
# NUMBA-ACCELERATED FEEDBACK COMPUTATION
# ============================================================================

@jit(nopython=True, cache=True)
def compute_feedback(guess: np.ndarray, answer: np.ndarray) -> int:
    """
    Compute Wordle feedback for a guess against an answer.
    
    Args:
        guess: shape (5,) array of char codes (0-25 for a-z)
        answer: shape (5,) array of char codes
        
    Returns:
        Integer feedback pattern (0-242)
    """
    feedback = np.zeros(5, dtype=np.int32)
    answer_counts = np.zeros(26, dtype=np.int32)
    
    # Count letters in answer
    for i in range(5):
        answer_counts[answer[i]] += 1
    
    # First pass: mark greens
    for i in range(5):
        if guess[i] == answer[i]:
            feedback[i] = 2  # GREEN
            answer_counts[guess[i]] -= 1
    
    # Second pass: mark yellows
    for i in range(5):
        if feedback[i] == 0:
            c = guess[i]
            if answer_counts[c] > 0:
                feedback[i] = 1  # YELLOW
                answer_counts[c] -= 1
    
    return feedback[0] + 3*feedback[1] + 9*feedback[2] + 27*feedback[3] + 81*feedback[4]


@jit(nopython=True, parallel=True, cache=True)
def compute_feedback_matrix(guess_chars: np.ndarray, answer_chars: np.ndarray) -> np.ndarray:
    """
    Compute feedback for all guess/answer pairs in parallel.
    
    Args:
        guess_chars: shape (n_guesses, 5) array of char codes
        answer_chars: shape (n_answers, 5) array of char codes
        
    Returns:
        shape (n_guesses, n_answers) feedback matrix
    """
    n_guesses = guess_chars.shape[0]
    n_answers = answer_chars.shape[0]
    result = np.zeros((n_guesses, n_answers), dtype=np.uint8)
    
    for i in prange(n_guesses):
        for j in range(n_answers):
            result[i, j] = compute_feedback(guess_chars[i], answer_chars[j])
    
    return result


@jit(nopython=True, cache=True)
def get_partition_sizes(feedback_row: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    """
    Count how many candidates fall into each feedback partition.
    
    Args:
        feedback_row: feedback values for one guess against all words
        candidate_mask: boolean mask of current candidates
        
    Returns:
        Array of 243 partition sizes
    """
    sizes = np.zeros(243, dtype=np.int32)
    for i in range(len(feedback_row)):
        if candidate_mask[i]:
            sizes[feedback_row[i]] += 1
    return sizes


@jit(nopython=True, cache=True)
def score_guess(sizes: np.ndarray, total: int, is_candidate: bool) -> float:
    """
    Score a guess using information-theoretic heuristic.
    
    Based on Jonathan Olson's approach:
    - Primary: Expected partition size (sum of size^2 / total)
    - Secondary: Worst-case partition (minimax)
    - Tertiary: Prefer candidates
    
    Lower score is better.
    """
    if total == 0:
        return 0.0
    
    # Expected remaining = sum(size^2) / total
    expected = 0.0
    worst = 0
    n_partitions = 0
    
    for i in range(243):
        s = sizes[i]
        if s > 0:
            if i != CORRECT_PATTERN:
                expected += s * s
                if s > worst:
                    worst = s
            n_partitions += 1
    
    expected /= total
    
    # Entropy bonus (more partitions = more information)
    entropy_bonus = -n_partitions * 0.001
    
    # Combined score
    score = expected + worst * 0.01 + entropy_bonus
    
    # Strong preference for candidates when pool is small
    if is_candidate:
        score -= 0.1
    
    return score


# ============================================================================
# SOLVER CLASS
# ============================================================================

class WordleSolver:
    """
    Optimal Wordle solver for the real NYT Wordle game.
    
    Uses two word lists:
    - answers: Words that can be the daily answer (~3158)
    - guesses: All words you can guess (~14855, includes answers)
    """
    
    def __init__(self, answers: List[str], guesses: List[str] = None, 
                 first_guess: str = "tarse"):
        """
        Initialize solver with word lists.
        
        Args:
            answers: List of possible answer words
            guesses: List of valid guesses (if None, uses answers)
            first_guess: Starting guess (TARSE is optimal for current NYT list)
        """
        self.answers = [w.lower() for w in answers]
        self.guesses = [w.lower() for w in (guesses or answers)]
        
        self.answer_to_idx = {w: i for i, w in enumerate(self.answers)}
        self.guess_to_idx = {w: i for i, w in enumerate(self.guesses)}
        
        self.n_answers = len(self.answers)
        self.n_guesses = len(self.guesses)
        self.first_guess = first_guess.lower()
        
        # Convert words to char arrays for numba
        self.answer_chars = self._words_to_chars(self.answers)
        self.guess_chars = self._words_to_chars(self.guesses)
        
        # Precompute feedback matrix: shape (n_guesses, n_answers)
        print(f"Precomputing feedback matrix ({self.n_guesses} guesses × {self.n_answers} answers)...")
        start = time.time()
        self.feedback_matrix = compute_feedback_matrix(self.guess_chars, self.answer_chars)
        elapsed = time.time() - start
        pairs = self.n_guesses * self.n_answers
        print(f"Done in {elapsed:.1f}s ({pairs / elapsed / 1e6:.1f}M pairs/sec)")
        
        # Setup first guess
        self._setup_first_guess()
    
    def _words_to_chars(self, words: List[str]) -> np.ndarray:
        """Convert words to char code array."""
        arr = np.zeros((len(words), 5), dtype=np.int32)
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                arr[i, j] = ord(c) - ord('a')
        return arr
    
    def _setup_first_guess(self):
        """Setup first guess index."""
        if self.first_guess in self.guess_to_idx:
            self.first_guess_idx = self.guess_to_idx[self.first_guess]
            print(f"Using first guess: {self.first_guess}")
        else:
            print(f"Warning: '{self.first_guess}' not in guess list, computing best...")
            candidate_mask = np.ones(self.n_answers, dtype=np.bool_)
            self.first_guess_idx = self._find_best_guess(candidate_mask, self.n_answers)
            self.first_guess = self.guesses[self.first_guess_idx]
            print(f"Best first guess: {self.first_guess}")
    
    def _find_best_guess(self, candidate_mask: np.ndarray, n_candidates: int,
                         guesses_remaining: int = 6) -> int:
        """
        Find the best guess for the current game state.
        
        Uses fast heuristic: minimize expected partition size + worst case.
        """
        if n_candidates == 0:
            raise ValueError("No candidates remaining")
        
        candidates = np.where(candidate_mask)[0]
        
        if n_candidates == 1:
            # Only one candidate - guess it
            word = self.answers[candidates[0]]
            return self.guess_to_idx.get(word, 0)
        
        if n_candidates == 2 or guesses_remaining == 1:
            # Guess a candidate
            word = self.answers[candidates[0]]
            return self.guess_to_idx[word]
        
        # Words that are candidates (so we can prefer them)
        candidate_words = set(self.answers[i] for i in candidates)
        
        best_idx = 0
        best_score = float('inf')
        
        # For small pools, check all guesses; otherwise limit search
        if n_candidates <= 100:
            guesses_to_check = range(self.n_guesses)
        else:
            # Quick pre-filter: get top guesses by expected remaining
            scores = []
            for g in range(self.n_guesses):
                sizes = get_partition_sizes(self.feedback_matrix[g], candidate_mask)
                exp = np.sum(sizes.astype(np.float64) ** 2) / n_candidates
                scores.append((exp, g))
            scores.sort()
            # Top 500 guesses
            guesses_to_check = [g for _, g in scores[:500]]
        
        for guess_idx in guesses_to_check:
            sizes = get_partition_sizes(self.feedback_matrix[guess_idx], candidate_mask)
            is_candidate = self.guesses[guess_idx] in candidate_words
            s = score_guess(sizes, n_candidates, is_candidate)
            
            if s < best_score:
                best_score = s
                best_idx = guess_idx
        
        return best_idx
    
    def solve(self, answer: str, verbose: bool = False) -> Tuple[int, List[str]]:
        """
        Solve for a given answer word.
        
        Args:
            answer: The target word
            verbose: Print progress
            
        Returns:
            (num_guesses, list_of_guesses)
        """
        answer = answer.lower()
        answer_idx = self.answer_to_idx.get(answer)
        if answer_idx is None:
            raise ValueError(f"Answer '{answer}' not in answer list")
        
        candidate_mask = np.ones(self.n_answers, dtype=np.bool_)
        n_candidates = self.n_answers
        guesses = []
        
        for turn in range(6):
            guesses_remaining = 6 - turn
            
            # Select guess
            if turn == 0:
                guess_idx = self.first_guess_idx
            else:
                guess_idx = self._find_best_guess(candidate_mask, n_candidates, guesses_remaining)
            
            guess = self.guesses[guess_idx]
            guesses.append(guess)
            
            # Get feedback
            feedback = self.feedback_matrix[guess_idx, answer_idx]
            
            if verbose:
                print(f"Turn {turn + 1}: {guess} (pattern {feedback}, {n_candidates} candidates)")
            
            if feedback == CORRECT_PATTERN:
                return len(guesses), guesses
            
            # Filter candidates
            feedback_row = self.feedback_matrix[guess_idx]
            for i in range(self.n_answers):
                if candidate_mask[i] and feedback_row[i] != feedback:
                    candidate_mask[i] = False
            n_candidates = int(np.sum(candidate_mask))
            
            if n_candidates == 0:
                raise RuntimeError("No candidates remaining - bug in solver")
        
        return 7, guesses  # Failed


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_words(filepath: str) -> List[str]:
    """Load word list from file."""
    with open(filepath, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def benchmark(solver: WordleSolver, test_words: List[str] = None, 
              verbose: bool = True) -> Dict:
    """
    Benchmark solver on word list.
    
    Args:
        solver: WordleSolver instance
        test_words: Words to test (default: all answers)
        verbose: Print progress
        
    Returns:
        Dict with results
    """
    from collections import Counter
    
    if test_words is None:
        test_words = solver.answers
    
    results = []
    dist = Counter()
    failures = []
    
    start = time.time()
    for i, word in enumerate(test_words):
        if verbose and i % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            avg = sum(results) / len(results) if results else 0
            print(f"[{i}/{len(test_words)}] {rate:.1f} w/s, avg={avg:.4f}")
        
        try:
            n, _ = solver.solve(word)
            results.append(n)
            dist[n] += 1
            if n > 6:
                failures.append(word)
        except Exception as e:
            print(f"Error: {word}: {e}")
            results.append(7)
            failures.append(word)
    
    elapsed = time.time() - start
    
    return {
        'total': len(test_words),
        'average': sum(results) / len(results),
        'distribution': dict(sorted(dist.items())),
        'failures': len(failures),
        'failed_words': failures[:20],
        'time': elapsed,
        'rate': len(test_words) / elapsed,
    }


def print_results(results: Dict):
    """Pretty print benchmark results."""
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Words tested: {results['total']}")
    print(f"Average guesses: {results['average']:.4f}")
    print(f"Failures: {results['failures']} ({100*results['failures']/results['total']:.2f}%)")
    print(f"Time: {results['time']:.1f}s ({results['rate']:.1f} words/sec)")
    print("\nDistribution:")
    for n, count in results['distribution'].items():
        pct = 100 * count / results['total']
        bar = "█" * int(pct / 2)
        print(f"  {n}: {count:5d} ({pct:5.2f}%) {bar}")
    if results['failed_words']:
        print(f"\nFailed words: {results['failed_words']}")
    print("=" * 50)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import random
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load both word lists
    answers_file = os.path.join(base_dir, "words", "answers.txt")
    guesses_file = os.path.join(base_dir, "words", "allowed_guesses.txt")
    
    print("Loading word lists...")
    answers = load_words(answers_file)
    guesses = load_words(guesses_file)
    print(f"  Answers: {len(answers)} words (possible daily answers)")
    print(f"  Guesses: {len(guesses)} words (valid guesses)")
    
    # Create solver with SALET as first guess (optimal for original 2315 list)
    solver = WordleSolver(answers, guesses, first_guess="salet")
    
    # Quick test
    print("\n--- Quick tests ---")
    test_words = ["crane", "slate", "tarse", "jazzy", "mamma"]
    for word in test_words:
        if word in solver.answer_to_idx:
            n, gs = solver.solve(word, verbose=True)
            print(f"  -> Solved in {n} guesses: {gs}\n")
    
    # Sample benchmark
    print("\n--- Sample benchmark (500 words) ---")
    random.seed(42)
    sample = random.sample(answers, min(500, len(answers)))
    results = benchmark(solver, sample, verbose=True)
    print_results(results)
    
    # Full benchmark option
    response = input("\nRun full benchmark on all answers? (y/n): ").strip().lower()
    if response == 'y':
        print("\n--- Full benchmark ---")
        results = benchmark(solver, answers, verbose=True)
        print_results(results)
