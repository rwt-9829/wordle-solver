"""
Improved Wordle Solver with Deep Lookahead
==========================================

This solver achieves near-optimal performance by using:
1. Better heuristic scoring (minimax-inspired)
2. Deeper lookahead for small candidate sets
3. Special handling for endgame positions

Target: ~3.42 average (close to optimal 3.4212 for SALET)
"""

import numpy as np
from numba import jit, prange
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import time
import os
from collections import defaultdict


# ============================================================================
# CONSTANTS
# ============================================================================

CORRECT_PATTERN = 242  # All green
N_PATTERNS = 243


# ============================================================================
# NUMBA FUNCTIONS
# ============================================================================

@jit(nopython=True, cache=True)
def compute_feedback(guess: np.ndarray, answer: np.ndarray) -> int:
    """Compute Wordle feedback."""
    feedback = np.zeros(5, dtype=np.int32)
    answer_counts = np.zeros(26, dtype=np.int32)
    
    for i in range(5):
        answer_counts[answer[i]] += 1
    
    for i in range(5):
        if guess[i] == answer[i]:
            feedback[i] = 2
            answer_counts[guess[i]] -= 1
    
    for i in range(5):
        if feedback[i] == 0 and answer_counts[guess[i]] > 0:
            feedback[i] = 1
            answer_counts[guess[i]] -= 1
    
    return feedback[0] + 3*feedback[1] + 9*feedback[2] + 27*feedback[3] + 81*feedback[4]


@jit(nopython=True, parallel=True, cache=True)
def compute_feedback_matrix(guess_chars: np.ndarray, answer_chars: np.ndarray) -> np.ndarray:
    """Compute feedback matrix for all guess/answer pairs."""
    n_guesses = guess_chars.shape[0]
    n_answers = answer_chars.shape[0]
    result = np.zeros((n_guesses, n_answers), dtype=np.uint8)
    
    for i in prange(n_guesses):
        for j in range(n_answers):
            result[i, j] = compute_feedback(guess_chars[i], answer_chars[j])
    
    return result


@jit(nopython=True, cache=True)
def get_partition_sizes(feedback_row: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Get partition sizes for each feedback pattern."""
    sizes = np.zeros(243, dtype=np.int32)
    for c in candidates:
        sizes[feedback_row[c]] += 1
    return sizes


@jit(nopython=True, cache=True)
def compute_entropy(sizes: np.ndarray, total: int) -> float:
    """Compute Shannon entropy of partition distribution."""
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for s in sizes:
        if s > 0:
            p = s / total
            entropy -= p * np.log2(p)
    
    return entropy


@jit(nopython=True, cache=True)
def score_guess_advanced(sizes: np.ndarray, total: int, is_candidate: bool) -> float:
    """
    Advanced scoring function combining multiple heuristics.
    Lower score is better.
    
    Components:
    1. Expected remaining (sum of size^2 / total) - primary
    2. Worst case partition size - secondary
    3. Number of partitions (more = better) - tertiary
    4. Candidate preference - bonus
    """
    if total == 0:
        return 0.0
    
    expected = 0.0
    worst = 0
    n_partitions = 0
    
    for i in range(243):
        s = sizes[i]
        if s > 0:
            n_partitions += 1
            if i != CORRECT_PATTERN:
                expected += s * s
                if s > worst:
                    worst = s
    
    expected /= total
    
    # Combined score with tuned weights
    # Primary: expected remaining (most important)
    # Secondary: worst case (avoids catastrophic branches)
    # Tertiary: partition count (information gained)
    score = expected + worst * 0.05 - n_partitions * 0.001
    
    # Significant bonus for candidates (can solve in 1 guess)
    if is_candidate:
        score -= 0.15
    
    return score


# ============================================================================
# IMPROVED SOLVER CLASS
# ============================================================================

class ImprovedSolver:
    """
    High-performance Wordle solver with deep lookahead.
    """
    
    def __init__(self, answers: List[str], guesses: List[str] = None,
                 first_guess: str = "salet"):
        """Initialize solver."""
        self.answers = [w.lower() for w in answers]
        self.guesses = [w.lower() for w in (guesses or answers)]
        
        self.answer_to_idx = {w: i for i, w in enumerate(self.answers)}
        self.guess_to_idx = {w: i for i, w in enumerate(self.guesses)}
        
        self.n_answers = len(self.answers)
        self.n_guesses = len(self.guesses)
        self.first_guess = first_guess.lower()
        
        # Convert to char arrays
        self.answer_chars = self._words_to_chars(self.answers)
        self.guess_chars = self._words_to_chars(self.guesses)
        
        # Precompute feedback matrix
        print(f"Computing feedback matrix ({self.n_guesses} x {self.n_answers})...")
        t0 = time.time()
        self.feedback_matrix = compute_feedback_matrix(self.guess_chars, self.answer_chars)
        print(f"Done in {time.time() - t0:.1f}s")
        
        # Setup first guess
        if first_guess in self.guess_to_idx:
            self.first_guess_idx = self.guess_to_idx[first_guess]
        else:
            raise ValueError(f"First guess '{first_guess}' not in guess list")
        
        # Memoization for small sets
        self.memo: Dict[frozenset, Tuple[int, int]] = {}
        
        # Precompute guess ordering for full candidate set
        self._precompute_orderings()
    
    def _words_to_chars(self, words: List[str]) -> np.ndarray:
        """Convert words to char arrays."""
        arr = np.zeros((len(words), 5), dtype=np.int32)
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                arr[i, j] = ord(c) - ord('a')
        return arr
    
    def _precompute_orderings(self):
        """Precompute guess orderings."""
        all_candidates = np.arange(self.n_answers, dtype=np.int32)
        
        scores = []
        for g in range(self.n_guesses):
            sizes = get_partition_sizes(self.feedback_matrix[g], all_candidates)
            is_cand = self.guesses[g] in self.answer_to_idx
            s = score_guess_advanced(sizes, self.n_answers, is_cand)
            scores.append((s, g))
        
        scores.sort()
        self.sorted_guesses = [g for _, g in scores]
        self.top_guesses = self.sorted_guesses[:500]
    
    def _get_partitions(self, guess_idx: int, candidates: np.ndarray) -> Dict[int, np.ndarray]:
        """Get partitions for a guess."""
        feedback_row = self.feedback_matrix[guess_idx]
        partitions = defaultdict(list)
        
        for c in candidates:
            partitions[feedback_row[c]].append(c)
        
        return {k: np.array(v, dtype=np.int32) for k, v in partitions.items()}
    
    def _solve_small(self, candidates: np.ndarray, depth: int, max_depth: int = 6) -> Tuple[int, int]:
        """
        Solve small candidate sets optimally with minimax.
        Returns (total_guesses, best_guess_idx).
        """
        n = len(candidates)
        
        if n == 0:
            return 0, -1
        
        if n == 1:
            g = self.guess_to_idx.get(self.answers[candidates[0]], 0)
            return 1, g
        
        if depth >= max_depth:
            return n * 10, -1  # Penalty for exceeding depth
        
        # Check memo
        key = frozenset(candidates.tolist())
        if key in self.memo:
            return self.memo[key]
        
        # n=2: guess any candidate
        if n == 2:
            g = self.guess_to_idx.get(self.answers[candidates[0]], 0)
            self.memo[key] = (n, g)
            return n, g
        
        candidate_words = set(self.answers[c] for c in candidates)
        
        # Try all guesses, ordered by heuristic
        best_total = n * max_depth
        best_guess = -1
        
        # Limit guesses to try based on set size
        if n <= 8:
            guesses_to_try = range(self.n_guesses)
        elif n <= 20:
            guesses_to_try = self.top_guesses[:1000]
        else:
            guesses_to_try = self.top_guesses[:200]
        
        for guess_idx in guesses_to_try:
            partitions = self._get_partitions(guess_idx, candidates)
            
            # Each candidate uses one guess
            total = n
            
            # Add cost of solving each non-trivial partition
            for pattern, part in sorted(partitions.items(), key=lambda x: -len(x[1])):
                if pattern == CORRECT_PATTERN:
                    continue  # Already counted in total
                
                if total >= best_total:
                    break  # Beta cutoff
                
                sub_total, _ = self._solve_small(part, depth + 1, max_depth)
                total += sub_total
            
            if total < best_total:
                best_total = total
                best_guess = guess_idx
        
        self.memo[key] = (best_total, best_guess)
        return best_total, best_guess
    
    def _find_best_guess(self, candidates: np.ndarray, depth: int = 1) -> int:
        """Find best guess using hybrid approach."""
        n = len(candidates)
        
        if n == 0:
            raise ValueError("No candidates")
        
        if n == 1:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        if n == 2:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        # For small sets (≤20), use minimax
        if n <= 20:
            _, best_guess = self._solve_small(candidates, depth)
            return best_guess
        
        # For medium/large sets, use heuristic with lookahead
        candidate_words = set(self.answers[c] for c in candidates)
        
        best_score = float('inf')
        best_idx = 0
        
        # Determine how many guesses to try
        if n <= 50:
            guesses_to_try = self.top_guesses[:1000]
        elif n <= 100:
            guesses_to_try = self.top_guesses[:500]
        else:
            guesses_to_try = self.top_guesses[:200]
        
        for guess_idx in guesses_to_try:
            sizes = get_partition_sizes(self.feedback_matrix[guess_idx], candidates)
            is_cand = self.guesses[guess_idx] in candidate_words
            score = score_guess_advanced(sizes, n, is_cand)
            
            if score < best_score:
                best_score = score
                best_idx = guess_idx
        
        return best_idx
    
    def solve(self, answer: str, verbose: bool = False) -> Tuple[int, List[str]]:
        """Solve for a given answer."""
        answer = answer.lower()
        answer_idx = self.answer_to_idx.get(answer)
        if answer_idx is None:
            raise ValueError(f"Unknown answer: {answer}")
        
        candidates = np.arange(self.n_answers, dtype=np.int32)
        guesses = []
        
        for turn in range(6):
            # Get best guess
            if turn == 0:
                guess_idx = self.first_guess_idx
            else:
                guess_idx = self._find_best_guess(candidates, turn + 1)
            
            guess = self.guesses[guess_idx]
            guesses.append(guess)
            
            feedback = self.feedback_matrix[guess_idx, answer_idx]
            
            if verbose:
                print(f"Turn {turn + 1}: {guess} ({len(candidates)} candidates)")
            
            if feedback == CORRECT_PATTERN:
                return len(guesses), guesses
            
            # Filter candidates
            feedback_row = self.feedback_matrix[guess_idx]
            candidates = np.array([c for c in candidates if feedback_row[c] == feedback], dtype=np.int32)
        
        return 7, guesses


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_words(filepath: str) -> List[str]:
    """Load words from file."""
    with open(filepath, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def benchmark(solver: ImprovedSolver, words: List[str] = None, verbose: bool = True) -> Dict:
    """Benchmark solver."""
    from collections import Counter
    
    if words is None:
        words = solver.answers
    
    results = []
    dist = Counter()
    failures = []
    
    t0 = time.time()
    for i, word in enumerate(words):
        if verbose and (i % 500 == 0):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            avg = sum(results) / len(results) if results else 0
            print(f"[{i}/{len(words)}] {rate:.1f} w/s, avg={avg:.4f}")
        
        n, _ = solver.solve(word)
        results.append(n)
        dist[n] += 1
        if n > 6:
            failures.append(word)
    
    elapsed = time.time() - t0
    
    return {
        'total': len(words),
        'average': sum(results) / len(results),
        'total_guesses': sum(results),
        'distribution': dict(sorted(dist.items())),
        'failures': len(failures),
        'failed_words': failures,
        'time': elapsed,
        'rate': len(words) / elapsed,
    }


def print_results(results: Dict):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Words tested: {results['total']}")
    print(f"Total guesses: {results['total_guesses']}")
    print(f"Average: {results['average']:.4f}")
    print(f"Failures: {results['failures']}")
    print(f"Time: {results['time']:.1f}s ({results['rate']:.1f} words/sec)")
    print("\nDistribution:")
    for n, count in results['distribution'].items():
        pct = 100 * count / results['total']
        bar = "█" * int(pct / 2)
        print(f"  {n}: {count:5d} ({pct:5.2f}%) {bar}")
    if results['failed_words']:
        print(f"\nFailed: {results['failed_words'][:10]}")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import random
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    answers = load_words(os.path.join(base_dir, "words", "answers.txt"))
    guesses = load_words(os.path.join(base_dir, "words", "allowed_guesses.txt"))
    
    print(f"Answers: {len(answers)}, Guesses: {len(guesses)}")
    
    # Test different starting words
    starters = ['salet', 'trace', 'crane', 'slate']
    
    for starter in starters:
        print(f"\n{'='*60}")
        print(f"Testing starter: {starter}")
        print('='*60)
        
        solver = ImprovedSolver(answers, guesses, first_guess=starter)
        
        # Sample test
        random.seed(42)
        sample = random.sample(answers, 200)
        results = benchmark(solver, sample, verbose=True)
        print_results(results)
