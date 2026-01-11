"""
Optimal Wordle Solver - Targeting 3.42 Average
==============================================

This implements a near-optimal solver using:
1. Precomputed optimal second guesses for each first-guess feedback pattern
2. Deep minimax search for small candidate sets
3. Improved heuristics combining expected size, worst case, and entropy
4. Aggressive pruning with alpha-beta style cutoffs

Target: 3.4201 average (7920 total guesses for 2315 words)
"""

import numpy as np
from numba import jit, prange
from typing import List, Dict, Tuple, Optional, Set, FrozenSet
from collections import defaultdict
import time
import os
import sys


# ============================================================================
# CONSTANTS
# ============================================================================

CORRECT_PATTERN = 242  # GGGGG
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
def get_partition_sizes(feedback_row: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Get partition sizes for a guess."""
    sizes = np.zeros(243, dtype=np.int32)
    for c in candidates:
        sizes[feedback_row[c]] += 1
    return sizes


@jit(nopython=True, cache=True)
def score_guess_fast(feedback_matrix: np.ndarray, guess_idx: int, 
                     candidates: np.ndarray, n: int) -> Tuple[float, int]:
    """
    Score a guess using improved heuristic.
    Returns (expected_size, worst_size).
    """
    feedback_row = feedback_matrix[guess_idx]
    sizes = np.zeros(243, dtype=np.int32)
    
    for c in candidates:
        sizes[feedback_row[c]] += 1
    
    expected = 0.0
    worst = 0
    
    for i in range(243):
        s = sizes[i]
        if s > 0 and i != CORRECT_PATTERN:
            expected += s * s
            if s > worst:
                worst = s
    
    expected /= n
    return expected, worst


# ============================================================================
# OPTIMAL SOLVER CLASS
# ============================================================================

class OptimalSolver:
    """
    Near-optimal Wordle solver targeting 3.42 average.
    """
    
    def __init__(self, answers: List[str], guesses: List[str] = None,
                 first_guess: str = "salet", verbose: bool = True):
        """
        Initialize solver.
        
        Args:
            answers: List of possible answer words (2315 for original Wordle)
            guesses: List of valid guesses (14855 for original Wordle)
            first_guess: Pre-computed optimal first guess
            verbose: Print initialization progress
        """
        self.answers = [w.lower() for w in answers]
        self.guesses = [w.lower() for w in (guesses or answers)]
        
        self.answer_to_idx = {w: i for i, w in enumerate(self.answers)}
        self.guess_to_idx = {w: i for i, w in enumerate(self.guesses)}
        
        self.n_answers = len(self.answers)
        self.n_guesses = len(self.guesses)
        self.first_guess = first_guess.lower()
        self.verbose_init = verbose
        
        # Convert to char arrays
        self.answer_chars = self._words_to_chars(self.answers)
        self.guess_chars = self._words_to_chars(self.guesses)
        
        # Precompute feedback matrix
        if verbose:
            print(f"Computing feedback matrix ({self.n_guesses} x {self.n_answers})...")
        t0 = time.time()
        self.feedback_matrix = compute_feedback_matrix(self.guess_chars, self.answer_chars)
        if verbose:
            print(f"Done in {time.time() - t0:.1f}s")
        
        # Memoization cache for minimax
        self.cache: Dict[FrozenSet[int], Tuple[int, int]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Precompute sorted guesses by heuristic score
        self._precompute_guess_rankings()
        
        # Precompute optimal second guesses for each first-guess feedback
        self._precompute_second_guesses()
        
        # Verbose tracking
        self.verbose_solve = False
        self.minimax_calls = 0
        self.last_status_time = time.time()
    
    def _words_to_chars(self, words: List[str]) -> np.ndarray:
        """Convert words to char arrays."""
        arr = np.zeros((len(words), 5), dtype=np.int32)
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                arr[i, j] = ord(c) - ord('a')
        return arr
    
    def _precompute_guess_rankings(self):
        """Precompute globally good guesses based on information gain."""
        if self.verbose_init:
            print("Precomputing guess rankings...")
        
        all_candidates = np.arange(self.n_answers, dtype=np.int32)
        n = self.n_answers
        
        scores = []
        for g in range(self.n_guesses):
            exp, worst = score_guess_fast(self.feedback_matrix, g, all_candidates, n)
            # Combined heuristic
            score = exp + worst * 0.05
            # Slight preference for answer candidates
            if self.guesses[g] in self.answer_to_idx:
                score -= 0.02
            scores.append((score, g))
        
        scores.sort()
        self.ranked_guesses = np.array([g for _, g in scores], dtype=np.int32)
        
        if self.verbose_init:
            print(f"Top 10 guesses: {[self.guesses[g] for _, g in scores[:10]]}")
    
    def _precompute_second_guesses(self):
        """
        Precompute optimal second guess for each first-guess feedback pattern.
        This is key for achieving optimal performance.
        """
        if self.verbose_init:
            print("Precomputing optimal second guesses...")
        
        first_idx = self.guess_to_idx.get(self.first_guess)
        if first_idx is None:
            print(f"Warning: first guess '{self.first_guess}' not found")
            self.second_guesses = {}
            return
        
        first_feedback = self.feedback_matrix[first_idx]
        
        # Group answers by their feedback to first guess
        pattern_to_answers = defaultdict(list)
        for a in range(self.n_answers):
            fb = first_feedback[a]
            if fb != CORRECT_PATTERN:
                pattern_to_answers[fb].append(a)
        
        self.second_guesses = {}
        
        for pattern, answer_indices in pattern_to_answers.items():
            candidates = np.array(answer_indices, dtype=np.int32)
            n = len(candidates)
            
            if n <= 2:
                # Just pick first candidate
                self.second_guesses[pattern] = self.guess_to_idx[self.answers[candidates[0]]]
            else:
                # Find best second guess using deeper search
                best_guess = self._find_best_guess_deep(candidates, depth=2, max_guesses_to_try=300)
                self.second_guesses[pattern] = best_guess
        
        if self.verbose_init:
            print(f"Precomputed {len(self.second_guesses)} second guesses")
    
    def _find_best_guess_deep(self, candidates: np.ndarray, depth: int = 2,
                               max_guesses_to_try: int = 200,
                               exclude_guesses: Set[int] = None) -> int:
        """
        Find best guess using limited-depth minimax.
        """
        n = len(candidates)
        if n == 0:
            return 0
        if n == 1:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        if n == 2:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        exclude = exclude_guesses or set()
        
        # For small sets, first find guesses that maximize distinct patterns
        if n <= 6:
            # Find all guesses that create maximum distinct partitions
            best_distinct = 0
            best_guesses = []
            
            for guess_idx in range(self.n_guesses):
                if guess_idx in exclude:
                    continue
                
                feedback_row = self.feedback_matrix[guess_idx]
                patterns = set()
                for c in candidates:
                    patterns.add(feedback_row[c])
                
                distinct = len(patterns)
                if distinct > best_distinct:
                    best_distinct = distinct
                    best_guesses = [guess_idx]
                elif distinct == best_distinct:
                    best_guesses.append(guess_idx)
            
            if best_distinct == n:
                # Perfect guess exists - pick the one that's a candidate if possible
                for g in best_guesses:
                    if self.guesses[g] in [self.answers[c] for c in candidates]:
                        return g
                return best_guesses[0]
            
            # Use these top guesses for minimax evaluation
            guesses_to_try = best_guesses[:max_guesses_to_try]
        else:
            # Get top guesses by heuristic
            candidate_words = set(self.answers[c] for c in candidates)
            
            scores = []
            for guess_idx in self.ranked_guesses[:max(500, max_guesses_to_try * 2)]:
                if guess_idx in exclude:
                    continue
                exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
                score = exp + worst * 0.1
                if self.guesses[guess_idx] in candidate_words:
                    score -= 0.15  # Prefer candidates
                scores.append((score, guess_idx))
            
            # Also add all candidates
            for c in candidates:
                guess_idx = self.guess_to_idx.get(self.answers[c])
                if guess_idx is not None and guess_idx not in exclude:
                    exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
                    score = exp + worst * 0.1 - 0.15
                    scores.append((score, guess_idx))
            
            scores.sort()
            
            # Deduplicate and exclude already-used guesses
            seen = set()
            guesses_to_try = []
            for _, g in scores:
                if g not in seen and g not in exclude:
                    seen.add(g)
                    guesses_to_try.append(g)
                    if len(guesses_to_try) >= max_guesses_to_try:
                        break
            
            # Fallback if all guesses excluded
            if not guesses_to_try:
                for _, g in scores:
                    if g not in seen:
                        guesses_to_try.append(g)
                        break
        
        best_total = float('inf')
        best_guess = guesses_to_try[0] if guesses_to_try else 0
        
        for guess_idx in guesses_to_try:
            total = self._evaluate_guess(guess_idx, candidates, depth, best_total)
            if total < best_total:
                best_total = total
                best_guess = guess_idx
        
        return best_guess
    
    def _evaluate_guess(self, guess_idx: int, candidates: np.ndarray, 
                        depth: int, beta: float) -> float:
        """
        Evaluate total guesses needed if we make this guess.
        Uses minimax with pruning.
        """
        n = len(candidates)
        feedback_row = self.feedback_matrix[guess_idx]
        
        # Group candidates by feedback
        partitions = defaultdict(list)
        for c in candidates:
            fb = feedback_row[c]
            partitions[fb].append(c)
        
        total = 0.0
        
        # Sort partitions by size (largest first for better pruning)
        sorted_parts = sorted(partitions.items(), key=lambda x: -len(x[1]))
        
        for pattern, part in sorted_parts:
            part_size = len(part)
            
            if pattern == CORRECT_PATTERN:
                # Direct hit
                total += part_size
            else:
                # Need to continue solving
                total += part_size  # One guess for each
                
                if total >= beta:
                    return beta + 1  # Prune
                
                if depth > 1 and part_size > 1:
                    # Recursively evaluate
                    part_arr = np.array(part, dtype=np.int32)
                    sub_total = self._minimax_total(part_arr, depth - 1, beta - total)
                    total += sub_total
                    
                    if total >= beta:
                        return beta + 1
        
        return total
    
    def _minimax_total(self, candidates: np.ndarray, depth: int, beta: float) -> float:
        """
        Compute minimum additional guesses needed to solve all candidates.
        """
        n = len(candidates)
        
        if n <= 1:
            return 0
        
        if n == 2:
            return 1  # One of them will be wrong, need 1 more guess
        
        # Check cache
        key = frozenset(candidates.tolist())
        if key in self.cache:
            self.cache_hits += 1
            cached_total, _ = self.cache[key]
            return cached_total
        self.cache_misses += 1
        
        if depth <= 0:
            # Use heuristic estimate
            best_guess = self._find_best_guess_heuristic(candidates)
            exp, worst = score_guess_fast(self.feedback_matrix, best_guess, candidates, n)
            # Rough estimate: log2(n) additional guesses on average
            import math
            estimate = max(0, math.log2(n) - 0.5) * 0.7
            return estimate
        
        # Try top guesses
        guesses_to_try = self._get_top_guesses(candidates, min(50, 20 + n))
        
        best_total = beta
        best_guess = guesses_to_try[0] if guesses_to_try else 0
        
        for guess_idx in guesses_to_try:
            total = 0.0
            feedback_row = self.feedback_matrix[guess_idx]
            
            partitions = defaultdict(list)
            for c in candidates:
                fb = feedback_row[c]
                if fb != CORRECT_PATTERN:
                    partitions[fb].append(c)
            
            # Process partitions
            for pattern, part in sorted(partitions.items(), key=lambda x: -len(x[1])):
                part_size = len(part)
                
                if total >= best_total:
                    break
                
                if part_size == 1:
                    continue  # Will be solved directly
                
                if part_size == 2:
                    total += 1  # One more guess needed
                else:
                    part_arr = np.array(part, dtype=np.int32)
                    sub = self._minimax_total(part_arr, depth - 1, best_total - total)
                    total += sub
            
            if total < best_total:
                best_total = total
                best_guess = guess_idx
        
        self.cache[key] = (best_total, best_guess)
        return best_total
    
    def _find_best_guess_heuristic(self, candidates: np.ndarray,
                                    exclude_guesses: Set[int] = None) -> int:
        """Find best guess using fast heuristic."""
        n = len(candidates)
        if n == 1:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        exclude = exclude_guesses or set()
        candidate_words = set(self.answers[c] for c in candidates)
        
        best_score = float('inf')
        best_idx = 0
        
        # For very small sets, we need to find a guess that distinguishes ALL candidates
        if n <= 6:
            # Find guesses that create maximum distinct partitions
            for guess_idx in range(self.n_guesses):
                if guess_idx in exclude:
                    continue
                
                feedback_row = self.feedback_matrix[guess_idx]
                patterns = set()
                for c in candidates:
                    patterns.add(feedback_row[c])
                
                # Best guess creates n distinct patterns (one per candidate)
                distinct = len(patterns)
                if distinct == n:
                    # Perfect - each candidate gets unique feedback
                    return guess_idx
                
                # Otherwise score by: fewer distinct = worse
                exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
                score = exp + worst * 0.1 - distinct * 0.5
                if self.guesses[guess_idx] in candidate_words:
                    score -= 0.15
                
                if score < best_score:
                    best_score = score
                    best_idx = guess_idx
            
            return best_idx
        
        # Check top ranked guesses
        for guess_idx in self.ranked_guesses[:500]:
            if guess_idx in exclude:
                continue
                
            exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
            score = exp + worst * 0.1
            if self.guesses[guess_idx] in candidate_words:
                score -= 0.15
            
            if score < best_score:
                best_score = score
                best_idx = guess_idx
        
        return best_idx
    
    def _get_top_guesses(self, candidates: np.ndarray, k: int) -> List[int]:
        """Get top k guesses for candidates by heuristic."""
        n = len(candidates)
        candidate_words = set(self.answers[c] for c in candidates)
        
        scores = []
        for guess_idx in self.ranked_guesses[:max(300, k * 5)]:
            exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
            score = exp + worst * 0.1
            if self.guesses[guess_idx] in candidate_words:
                score -= 0.2
            scores.append((score, guess_idx))
        
        # Add all candidates
        for c in candidates:
            guess_idx = self.guess_to_idx.get(self.answers[c])
            if guess_idx is not None:
                exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
                scores.append((exp + worst * 0.1 - 0.2, guess_idx))
        
        scores.sort()
        
        seen = set()
        result = []
        for _, g in scores:
            if g not in seen:
                seen.add(g)
                result.append(g)
                if len(result) >= k:
                    break
        
        return result
    
    def find_best_guess(self, candidates: np.ndarray, depth: int = 2,
                        exclude_guesses: Set[int] = None) -> int:
        """Find best guess for current candidates."""
        n = len(candidates)
        
        if n == 0:
            raise ValueError("No candidates")
        if n == 1:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        if n == 2:
            # When 2 candidates, always guess one of them
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        # For small sets, use deeper search
        if n <= 20:
            return self._find_best_guess_deep(candidates, depth=3, max_guesses_to_try=150,
                                               exclude_guesses=exclude_guesses)
        elif n <= 50:
            return self._find_best_guess_deep(candidates, depth=2, max_guesses_to_try=100,
                                               exclude_guesses=exclude_guesses)
        else:
            return self._find_best_guess_heuristic(candidates, exclude_guesses=exclude_guesses)
    
    def solve(self, answer: str, verbose: bool = False) -> Tuple[int, List[str]]:
        """
        Solve for a given answer.
        
        Returns:
            (num_guesses, list_of_guesses)
        """
        answer = answer.lower()
        answer_idx = self.answer_to_idx.get(answer)
        if answer_idx is None:
            raise ValueError(f"Unknown answer: {answer}")
        
        self.verbose_solve = verbose
        candidates = np.arange(self.n_answers, dtype=np.int32)
        guesses = []
        used_guesses = set()  # Track used guess indices to avoid repeats
        
        for turn in range(6):
            n_cand = len(candidates)
            
            if turn == 0:
                # Use precomputed first guess
                guess_idx = self.guess_to_idx[self.first_guess]
                if verbose:
                    print(f"  Turn {turn+1}: using first guess '{self.first_guess}'")
            elif turn == 1 and len(guesses) == 1:
                # Use precomputed second guess if available
                first_fb = self.feedback_matrix[self.guess_to_idx[self.first_guess], answer_idx]
                if first_fb in self.second_guesses:
                    guess_idx = self.second_guesses[first_fb]
                    if verbose:
                        print(f"  Turn {turn+1}: using precomputed second guess")
                else:
                    guess_idx = self.find_best_guess(candidates, exclude_guesses=used_guesses)
                    if verbose:
                        print(f"  Turn {turn+1}: computed guess for {n_cand} candidates")
            else:
                if verbose:
                    t0 = time.time()
                guess_idx = self.find_best_guess(candidates, exclude_guesses=used_guesses)
                if verbose:
                    print(f"  Turn {turn+1}: found guess for {n_cand} candidates in {time.time()-t0:.2f}s")
            
            guess = self.guesses[guess_idx]
            guesses.append(guess)
            used_guesses.add(guess_idx)
            
            feedback = self.feedback_matrix[guess_idx, answer_idx]
            
            if verbose:
                fb_str = self._feedback_to_emoji(feedback)
                print(f"  Turn {turn+1}: {guess} -> {fb_str} ({n_cand} -> ", end='')
            
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
        
        # Failed (shouldn't happen with optimal play)
        return 7, guesses
    
    def _feedback_to_emoji(self, feedback: int) -> str:
        """Convert feedback pattern to emoji string."""
        chars = []
        for _ in range(5):
            chars.append(['⬛', '🟨', '🟩'][feedback % 3])
            feedback //= 3
        return ''.join(chars)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_words(filepath: str) -> List[str]:
    """Load words from file."""
    with open(filepath, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def benchmark(solver: OptimalSolver, words: List[str] = None,
              verbose: bool = True, progress_every: int = 100) -> Dict:
    """Benchmark solver on word list."""
    from collections import Counter
    
    if words is None:
        words = solver.answers
    
    results = []
    dist = Counter()
    failures = []
    
    t0 = time.time()
    for i, word in enumerate(words):
        if verbose and (i % progress_every == 0 or i < 10):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            avg = sum(results) / len(results) if results else 0
            eta = (len(words) - i) / rate if rate > 0 else 0
            print(f"[{i+1}/{len(words)}] avg={avg:.4f}, rate={rate:.1f}/s, "
                  f"ETA={eta/60:.1f}min, cache={len(solver.cache)}")
        
        try:
            n, gs = solver.solve(word, verbose=False)
            results.append(n)
            dist[n] += 1
            if n > 6:
                failures.append((word, gs))
                if verbose:
                    print(f"  FAIL: {word} ({n} guesses)")
        except Exception as e:
            print(f"  ERROR: {word}: {e}")
            results.append(7)
            failures.append((word, []))
    
    elapsed = time.time() - t0
    
    return {
        'total': len(words),
        'average': sum(results) / len(results),
        'total_guesses': sum(results),
        'distribution': dict(sorted(dist.items())),
        'failures': len(failures),
        'failed_words': failures[:20],
        'time': elapsed,
        'rate': len(words) / elapsed,
        'cache_size': len(solver.cache),
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
    print(f"Cache size: {results['cache_size']}")
    print("\nDistribution:")
    for n, count in results['distribution'].items():
        pct = 100 * count / results['total']
        bar = "█" * int(pct / 2)
        print(f"  {n}: {count:5d} ({pct:5.2f}%) {bar}")
    if results['failed_words']:
        print(f"\nFailed words: {[w for w, _ in results['failed_words']]}")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import random
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Loading word lists...")
    answers = load_words(os.path.join(base_dir, "words", "answers.txt"))
    guesses = load_words(os.path.join(base_dir, "words", "allowed_guesses.txt"))
    print(f"Answers: {len(answers)}, Guesses: {len(guesses)}")
    
    solver = OptimalSolver(answers, guesses, first_guess="salet")
    
    # Quick tests
    print("\n--- Quick tests ---")
    for word in ["crane", "jazzy", "mamma", "paste"]:
        if word in solver.answer_to_idx:
            n, gs = solver.solve(word, verbose=True)
            print(f"  Result: {n} guesses: {' -> '.join(gs)}\n")
    
    # Sample benchmark
    print("\n--- Sample benchmark (200 words) ---")
    random.seed(42)
    sample = random.sample(answers, 200)
    results = benchmark(solver, sample, verbose=True)
    print_results(results)
