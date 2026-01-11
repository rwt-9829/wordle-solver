"""
Minimax Wordle Solver with Alpha-Beta Pruning
=============================================

This implements the optimal minimax algorithm for Wordle, based on Alex Selby's approach.

Key insight: We want to minimize the TOTAL number of guesses across all possible answers.
This is equivalent to minimizing average guesses.

Algorithm:
- f(H) = |H| + min_{t∈T} Σ_{s≠GGGGG} f(P(H,t,s))
  where H = set of remaining candidates
        t = guess word
        s = feedback pattern
        P(H,t,s) = partition of H that gives feedback s with guess t

Optimizations:
1. Beta cutoff: If running sum exceeds best known, prune
2. Memoization: Cache results for identical candidate sets
3. Heuristic ordering: Try best guesses first
4. Small set optimization: Solve small sets optimally

Target: 3.4201 average (7920 total guesses for 2315 words)
"""

import numpy as np
from numba import jit, prange
from typing import List, Dict, Tuple, Optional, FrozenSet
from functools import lru_cache
import time
import os
from collections import defaultdict


# ============================================================================
# CONSTANTS
# ============================================================================

CORRECT_PATTERN = 242  # 3^5 - 1 = GGGGG
N_PATTERNS = 243


# ============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ============================================================================

@jit(nopython=True, cache=True)
def compute_feedback(guess: np.ndarray, answer: np.ndarray) -> int:
    """Compute Wordle feedback for a guess against an answer."""
    feedback = np.zeros(5, dtype=np.int32)
    answer_counts = np.zeros(26, dtype=np.int32)
    
    for i in range(5):
        answer_counts[answer[i]] += 1
    
    for i in range(5):
        if guess[i] == answer[i]:
            feedback[i] = 2
            answer_counts[guess[i]] -= 1
    
    for i in range(5):
        if feedback[i] == 0:
            c = guess[i]
            if answer_counts[c] > 0:
                feedback[i] = 1
                answer_counts[c] -= 1
    
    return feedback[0] + 3*feedback[1] + 9*feedback[2] + 27*feedback[3] + 81*feedback[4]


@jit(nopython=True, parallel=True, cache=True)
def compute_feedback_matrix(guess_chars: np.ndarray, answer_chars: np.ndarray) -> np.ndarray:
    """Compute feedback for all guess/answer pairs."""
    n_guesses = guess_chars.shape[0]
    n_answers = answer_chars.shape[0]
    result = np.zeros((n_guesses, n_answers), dtype=np.uint8)
    
    for i in prange(n_guesses):
        for j in range(n_answers):
            result[i, j] = compute_feedback(guess_chars[i], answer_chars[j])
    
    return result


@jit(nopython=True, cache=True)
def get_partitions(feedback_row: np.ndarray, candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get partition sizes and members for each feedback pattern.
    
    Returns:
        sizes: array of 243 sizes
        partitions: 2D array where partitions[pattern] contains candidate indices
    """
    sizes = np.zeros(243, dtype=np.int32)
    
    # First pass: count sizes
    for c in candidates:
        sizes[feedback_row[c]] += 1
    
    return sizes


@jit(nopython=True, cache=True)
def score_guess_heuristic(feedback_matrix: np.ndarray, guess_idx: int, 
                          candidates: np.ndarray, n_candidates: int) -> float:
    """
    Score a guess using expected remaining + worst case heuristic.
    Lower is better.
    """
    feedback_row = feedback_matrix[guess_idx]
    sizes = np.zeros(243, dtype=np.int32)
    
    for c in candidates:
        sizes[feedback_row[c]] += 1
    
    # Expected remaining = sum(size^2) / total
    expected = 0.0
    worst = 0
    
    for i in range(243):
        s = sizes[i]
        if s > 0 and i != CORRECT_PATTERN:
            expected += s * s
            if s > worst:
                worst = s
    
    expected /= n_candidates
    
    # Combined score (weighted)
    return expected + worst * 0.1


# ============================================================================
# MINIMAX SOLVER CLASS
# ============================================================================

class MinimaxSolver:
    """
    Optimal Wordle solver using minimax with alpha-beta pruning.
    """
    
    def __init__(self, answers: List[str], guesses: List[str] = None,
                 first_guess: str = "salet", max_depth: int = 6):
        """
        Initialize solver.
        
        Args:
            answers: List of possible answer words
            guesses: List of valid guesses (defaults to answers)
            first_guess: Pre-computed optimal first guess
            max_depth: Maximum number of guesses allowed
        """
        self.answers = [w.lower() for w in answers]
        self.guesses = [w.lower() for w in (guesses or answers)]
        
        self.answer_to_idx = {w: i for i, w in enumerate(self.answers)}
        self.guess_to_idx = {w: i for i, w in enumerate(self.guesses)}
        
        self.n_answers = len(self.answers)
        self.n_guesses = len(self.guesses)
        self.first_guess = first_guess.lower()
        self.max_depth = max_depth
        
        # Convert to char arrays
        self.answer_chars = self._words_to_chars(self.answers)
        self.guess_chars = self._words_to_chars(self.guesses)
        
        # Precompute feedback matrix
        print(f"Computing feedback matrix ({self.n_guesses} x {self.n_answers})...")
        t0 = time.time()
        self.feedback_matrix = compute_feedback_matrix(self.guess_chars, self.answer_chars)
        print(f"Done in {time.time() - t0:.1f}s")
        
        # Memoization cache: frozenset of candidate indices -> (best_total, best_guess_idx)
        self.cache: Dict[FrozenSet[int], Tuple[int, int]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Verbose tracking
        self.verbose_minimax = False
        self.minimax_calls = 0
        self.last_status_time = time.time()
        
        # Precompute sorted guesses by heuristic for each small candidate set size
        self._precompute_guess_ordering()
    
    def _words_to_chars(self, words: List[str]) -> np.ndarray:
        """Convert words to char arrays."""
        arr = np.zeros((len(words), 5), dtype=np.int32)
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                arr[i, j] = ord(c) - ord('a')
        return arr
    
    def _precompute_guess_ordering(self):
        """Precompute guess ordering based on entropy/expected information."""
        # For the full candidate set, compute heuristic scores
        all_candidates = np.arange(self.n_answers, dtype=np.int32)
        
        scores = []
        for g in range(self.n_guesses):
            s = score_guess_heuristic(self.feedback_matrix, g, all_candidates, self.n_answers)
            # Prefer guesses that are also candidates
            if self.guesses[g] in self.answer_to_idx:
                s -= 0.05
            scores.append((s, g))
        
        scores.sort()
        self.sorted_guesses = np.array([g for _, g in scores], dtype=np.int32)
        
        # Top guesses for quick lookup
        self.top_guesses = self.sorted_guesses[:200]
    
    def minimax(self, candidates: np.ndarray, depth: int, beta: int) -> Tuple[int, int]:
        """
        Minimax search with alpha-beta pruning.
        
        Args:
            candidates: Array of candidate indices
            depth: Current guess number (1-indexed)
            beta: Upper bound (prune if we exceed this)
        
        Returns:
            (total_guesses, best_guess_idx) for solving all candidates
        """
        self.minimax_calls += 1
        n = len(candidates)
        
        # Status update every 2 seconds
        if self.verbose_minimax and time.time() - self.last_status_time > 2.0:
            cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses + 1) * 100
            print(f"  [minimax] calls={self.minimax_calls}, depth={depth}, n={n}, "
                  f"cache={len(self.cache)} ({cache_rate:.0f}% hit), beta={beta}")
            self.last_status_time = time.time()
        
        if n == 0:
            return 0, -1
        
        if n == 1:
            # Only one candidate - guess it directly
            return 1, self.guess_to_idx.get(self.answers[candidates[0]], candidates[0])
        
        if depth >= self.max_depth:
            # Out of guesses - return penalty
            return n * 10, -1
        
        # Check cache
        key = frozenset(candidates.tolist())
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        
        # For n=2, just guess any candidate
        if n == 2:
            best_total = n
            best_guess = self.guess_to_idx.get(self.answers[candidates[0]], 0)
            self.cache[key] = (best_total, best_guess)
            return best_total, best_guess
        
        # For small n, use limited depth search
        # For very small sets, try more guesses
        if n <= 5:
            guesses_to_try = self._get_top_guesses_for_candidates(candidates, 100)
        elif n <= 15:
            guesses_to_try = self._get_top_guesses_for_candidates(candidates, 50)
        else:
            # Use heuristic only for larger sets
            guesses_to_try = self._get_top_guesses_for_candidates(candidates, 20)
        
        best_total = beta + 1
        best_guess = -1
        
        for guess_idx in guesses_to_try:
            feedback_row = self.feedback_matrix[guess_idx]
            
            # Group candidates by feedback pattern
            partitions = defaultdict(list)
            for c in candidates:
                fb = feedback_row[c]
                if fb != CORRECT_PATTERN:
                    partitions[fb].append(c)
            
            # Total = n (one guess for each candidate) + sum over partitions
            total = n
            
            # Process each partition (sorted by size for better pruning)
            sorted_parts = sorted(partitions.items(), key=lambda x: -len(x[1]))
            
            for pattern, part_candidates in sorted_parts:
                if total >= best_total:
                    break
                
                part_array = np.array(part_candidates, dtype=np.int32)
                sub_total, _ = self.minimax(part_array, depth + 1, best_total - total)
                total += sub_total
            
            if total < best_total:
                best_total = total
                best_guess = guess_idx
        
        self.cache[key] = (best_total, best_guess)
        return best_total, best_guess
    
    def _get_top_guesses_for_candidates(self, candidates: np.ndarray, top_k: int) -> List[int]:
        """Get top guesses for a specific candidate set, sorted by heuristic score."""
        n = len(candidates)
        candidate_words = set(self.answers[c] for c in candidates)
        
        scores = []
        for guess_idx in self.top_guesses[:min(500, self.n_guesses)]:
            score = score_guess_heuristic(self.feedback_matrix, guess_idx, candidates, n)
            if self.guesses[guess_idx] in candidate_words:
                score -= 0.2  # Stronger preference for candidates
            scores.append((score, guess_idx))
        
        # Also consider all candidates as potential guesses
        for c in candidates:
            guess_idx = self.guess_to_idx.get(self.answers[c])
            if guess_idx is not None:
                score = score_guess_heuristic(self.feedback_matrix, guess_idx, candidates, n) - 0.2
                scores.append((score, guess_idx))
        
        scores.sort()
        seen = set()
        result = []
        for _, idx in scores:
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
                if len(result) >= top_k:
                    break
        return result
    
    def find_best_guess(self, candidates: np.ndarray, depth: int = 1) -> int:
        """Find the best guess for current candidates."""
        n = len(candidates)
        
        if n == 0:
            raise ValueError("No candidates")
        
        if n == 1:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        if n == 2:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        # Use minimax for larger sets (but with limited depth for speed)
        if n <= 50:
            self.minimax_calls = 0
            self.last_status_time = time.time()
            _, best_guess = self.minimax(candidates, depth, n * 6)
            if self.verbose_minimax:
                print(f"  [find_best_guess] n={n}, minimax_calls={self.minimax_calls}")
            return best_guess
        
        # For larger sets, use heuristic
        return self._heuristic_best_guess(candidates)
    
    def _heuristic_best_guess(self, candidates: np.ndarray) -> int:
        """Find best guess using heuristic (no recursion)."""
        n = len(candidates)
        best_score = float('inf')
        best_idx = 0
        
        candidate_words = set(self.answers[c] for c in candidates)
        
        for guess_idx in self.top_guesses[:500]:
            score = score_guess_heuristic(self.feedback_matrix, guess_idx, candidates, n)
            
            # Prefer candidates
            if self.guesses[guess_idx] in candidate_words:
                score -= 0.1
            
            if score < best_score:
                best_score = score
                best_idx = guess_idx
        
        return best_idx
    
    def solve(self, answer: str, verbose: bool = False) -> Tuple[int, List[str]]:
        """
        Solve for a given answer.
        
        Args:
            answer: Target word
            verbose: Print progress
        
        Returns:
            (num_guesses, list_of_guesses)
        """
        answer = answer.lower()
        answer_idx = self.answer_to_idx.get(answer)
        if answer_idx is None:
            raise ValueError(f"Unknown answer: {answer}")
        
        candidates = np.arange(self.n_answers, dtype=np.int32)
        guesses = []
        
        for turn in range(self.max_depth):
            n_cand = len(candidates)
            
            # Get best guess
            if turn == 0 and self.first_guess:
                guess_idx = self.guess_to_idx[self.first_guess]
                if verbose:
                    print(f"  Turn {turn + 1}: using preset first guess")
            else:
                if verbose:
                    print(f"  Turn {turn + 1}: finding best guess for {n_cand} candidates...")
                    t0 = time.time()
                guess_idx = self.find_best_guess(candidates, turn + 1)
                if verbose:
                    elapsed = time.time() - t0
                    print(f"  Turn {turn + 1}: search took {elapsed:.2f}s")
            
            guess = self.guesses[guess_idx]
            guesses.append(guess)
            
            feedback = self.feedback_matrix[guess_idx, answer_idx]
            
            if verbose:
                fb_str = ''.join(['⬛🟨🟩'[int(c)] for c in np.base_repr(feedback, 3).zfill(5)])
                print(f"  Turn {turn + 1}: {guess} -> {fb_str} ({n_cand} -> ", end='')
            
            if feedback == CORRECT_PATTERN:
                if verbose:
                    print("SOLVED!)")
                return len(guesses), guesses
            
            # Filter candidates
            feedback_row = self.feedback_matrix[guess_idx]
            candidates = np.array([c for c in candidates if feedback_row[c] == feedback], dtype=np.int32)
            
            if verbose:
                print(f"{len(candidates)} candidates)")
                if len(candidates) <= 10:
                    cand_words = [self.answers[c] for c in candidates]
                    print(f"        remaining: {cand_words}")
        
        return len(guesses) + 1, guesses  # Failed


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_words(filepath: str) -> List[str]:
    """Load words from file."""
    with open(filepath, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def benchmark(solver: MinimaxSolver, words: List[str] = None, 
              verbose: bool = True) -> Dict:
    """Benchmark solver on word list."""
    from collections import Counter
    
    if words is None:
        words = solver.answers
    
    results = []
    dist = Counter()
    failures = []
    
    # Enable verbose minimax for detailed progress
    solver.verbose_minimax = verbose
    
    t0 = time.time()
    for i, word in enumerate(words):
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        avg = sum(results) / len(results) if results else 0
        eta = (len(words) - i) / rate if rate > 0 else 0
        
        if verbose:
            print(f"\n[{i+1}/{len(words)}] Solving '{word}' | "
                  f"avg={avg:.4f}, rate={rate:.2f}/s, "
                  f"ETA={eta/60:.1f}min, cache={len(solver.cache)}")
        
        try:
            n, gs = solver.solve(word, verbose=verbose)
            results.append(n)
            dist[n] += 1
            if verbose:
                print(f"  Result: {n} guesses -> {' -> '.join(gs)}")
            if n > 6:
                failures.append(word)
        except Exception as e:
            print(f"Error: {word}: {e}")
            import traceback
            traceback.print_exc()
            results.append(7)
            failures.append(word)
    
    elapsed = time.time() - t0
    solver.verbose_minimax = False
    
    return {
        'total': len(words),
        'average': sum(results) / len(results),
        'total_guesses': sum(results),
        'distribution': dict(sorted(dist.items())),
        'failures': len(failures),
        'failed_words': failures[:20],
        'time': elapsed,
        'rate': len(words) / elapsed,
        'cache_hits': solver.cache_hits,
        'cache_misses': solver.cache_misses,
    }


def print_results(results: Dict):
    """Pretty print results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Words tested: {results['total']}")
    print(f"Total guesses: {results['total_guesses']}")
    print(f"Average: {results['average']:.4f}")
    print(f"Failures: {results['failures']}")
    print(f"Time: {results['time']:.1f}s ({results['rate']:.1f} words/sec)")
    print(f"Cache: {results['cache_hits']} hits, {results['cache_misses']} misses")
    print("\nDistribution:")
    for n, count in results['distribution'].items():
        pct = 100 * count / results['total']
        bar = "█" * int(pct / 2)
        print(f"  {n}: {count:5d} ({pct:5.2f}%) {bar}")
    if results['failed_words']:
        print(f"\nFailed: {results['failed_words']}")
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
    
    solver = MinimaxSolver(answers, guesses, first_guess="salet")
    
    # Quick test
    print("\n--- Quick tests ---")
    for word in ["crane", "salet", "jazzy", "mamma"]:
        if word in solver.answer_to_idx:
            n, gs = solver.solve(word, verbose=True)
            print(f"  -> {n} guesses: {gs}\n")
    
    # Sample benchmark
    print("\n--- Sample benchmark (200 words) ---")
    random.seed(42)
    sample = random.sample(answers, 200)
    results = benchmark(solver, sample, verbose=True)
    print_results(results)
