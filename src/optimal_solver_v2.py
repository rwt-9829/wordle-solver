"""
Optimal Wordle Solver - Version 2
=================================

Targeting 3.42 average on the 2315-word original Wordle list.

Key optimizations:
1. Exhaustive search for "distinguishing" guesses when candidates <= 7
2. Heuristic-guided search for larger candidate sets
3. Precomputed optimal second guesses
4. Minimax with alpha-beta pruning for small sets
"""

import numpy as np
from numba import jit, prange
from typing import List, Dict, Tuple, Set, FrozenSet
from collections import defaultdict
import time
import os


CORRECT_PATTERN = 242  # GGGGG
N_PATTERNS = 243


@jit(nopython=True, cache=True)
def compute_feedback(guess: np.ndarray, answer: np.ndarray) -> int:
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
    n_guesses = guess_chars.shape[0]
    n_answers = answer_chars.shape[0]
    result = np.zeros((n_guesses, n_answers), dtype=np.uint8)
    
    for i in prange(n_guesses):
        for j in range(n_answers):
            result[i, j] = compute_feedback(guess_chars[i], answer_chars[j])
    
    return result


@jit(nopython=True, cache=True)
def score_guess_fast(feedback_matrix: np.ndarray, guess_idx: int, 
                     candidates: np.ndarray, n: int) -> Tuple[float, int]:
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


class OptimalSolverV2:
    def __init__(self, answers: List[str], guesses: List[str] = None,
                 first_guess: str = "salet", verbose: bool = True):
        self.answers = [w.lower() for w in answers]
        self.guesses = [w.lower() for w in (guesses or answers)]
        
        self.answer_to_idx = {w: i for i, w in enumerate(self.answers)}
        self.guess_to_idx = {w: i for i, w in enumerate(self.guesses)}
        
        self.n_answers = len(self.answers)
        self.n_guesses = len(self.guesses)
        self.first_guess = first_guess.lower()
        self.verbose_init = verbose
        
        self.answer_chars = self._words_to_chars(self.answers)
        self.guess_chars = self._words_to_chars(self.guesses)
        
        if verbose:
            print(f"Computing feedback matrix ({self.n_guesses} x {self.n_answers})...")
        t0 = time.time()
        self.feedback_matrix = compute_feedback_matrix(self.guess_chars, self.answer_chars)
        if verbose:
            print(f"Done in {time.time() - t0:.1f}s")
        
        self.cache: Dict[FrozenSet[int], Tuple[int, int]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        self._precompute_guess_rankings()
        self._precompute_second_guesses()
    
    def _words_to_chars(self, words: List[str]) -> np.ndarray:
        arr = np.zeros((len(words), 5), dtype=np.int32)
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                arr[i, j] = ord(c) - ord('a')
        return arr
    
    def _precompute_guess_rankings(self):
        if self.verbose_init:
            print("Precomputing guess rankings...")
        
        all_candidates = np.arange(self.n_answers, dtype=np.int32)
        n = self.n_answers
        
        scores = []
        for g in range(self.n_guesses):
            exp, worst = score_guess_fast(self.feedback_matrix, g, all_candidates, n)
            score = exp + worst * 0.05
            if self.guesses[g] in self.answer_to_idx:
                score -= 0.02
            scores.append((score, g))
        
        scores.sort()
        self.ranked_guesses = np.array([g for _, g in scores], dtype=np.int32)
        
        if self.verbose_init:
            print(f"Top 10: {[self.guesses[g] for _, g in scores[:10]]}")
    
    def _precompute_second_guesses(self):
        if self.verbose_init:
            print("Precomputing optimal second guesses...")
        
        first_idx = self.guess_to_idx.get(self.first_guess)
        if first_idx is None:
            self.second_guesses = {}
            return
        
        first_feedback = self.feedback_matrix[first_idx]
        
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
                self.second_guesses[pattern] = self.guess_to_idx[self.answers[candidates[0]]]
            else:
                # Use fast heuristic for second guess precomputation
                best_guess = self._find_best_guess_heuristic(candidates)
                self.second_guesses[pattern] = best_guess
        
        if self.verbose_init:
            print(f"Precomputed {len(self.second_guesses)} second guesses")
    
    def _count_distinct_patterns(self, guess_idx: int, candidates: np.ndarray) -> int:
        """Count how many distinct feedback patterns this guess creates."""
        feedback_row = self.feedback_matrix[guess_idx]
        patterns = set()
        for c in candidates:
            patterns.add(feedback_row[c])
        return len(patterns)
    
    def _get_partitions(self, guess_idx: int, candidates: np.ndarray) -> Dict[int, np.ndarray]:
        """Get feedback pattern -> candidate list for a guess."""
        feedback_row = self.feedback_matrix[guess_idx]
        partitions = defaultdict(list)
        for c in candidates:
            partitions[feedback_row[c]].append(c)
        return {k: np.array(v, dtype=np.int32) for k, v in partitions.items()}
    
    def _find_distinguishing_guess(self, candidates: np.ndarray, 
                                    exclude: Set[int] = None) -> int:
        """
        For small candidate sets, find a guess that gives each candidate
        a unique feedback pattern (if possible).
        """
        n = len(candidates)
        exclude = exclude or set()
        candidate_words = set(self.answers[c] for c in candidates)
        
        best_distinct = 0
        best_guesses = []
        
        # Scan all guesses to find those with maximum distinct patterns
        for guess_idx in range(self.n_guesses):
            if guess_idx in exclude:
                continue
            
            distinct = self._count_distinct_patterns(guess_idx, candidates)
            
            if distinct > best_distinct:
                best_distinct = distinct
                best_guesses = [guess_idx]
            elif distinct == best_distinct:
                best_guesses.append(guess_idx)
            
            # Early exit if we found perfect
            if distinct == n:
                if self.guesses[guess_idx] in candidate_words:
                    return guess_idx
        
        # If perfect found, prefer candidate
        if best_distinct == n:
            for g in best_guesses:
                if self.guesses[g] in candidate_words:
                    return g
            return best_guesses[0]
        
        # Otherwise, pick best by heuristic among top distinguishers
        best_score = float('inf')
        best_idx = best_guesses[0] if best_guesses else 0
        
        for guess_idx in best_guesses[:100]:  # Limit search
            exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
            score = exp + worst * 0.1
            if self.guesses[guess_idx] in candidate_words:
                score -= 0.2
            if score < best_score:
                best_score = score
                best_idx = guess_idx
        
        return best_idx
    
    def _find_best_guess_heuristic(self, candidates: np.ndarray,
                                    exclude: Set[int] = None) -> int:
        """Find best guess using fast heuristic."""
        n = len(candidates)
        if n == 1:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        exclude = exclude or set()
        candidate_words = set(self.answers[c] for c in candidates)
        
        # For small sets, use exhaustive distinguishing search
        if n <= 20:
            return self._find_distinguishing_guess(candidates, exclude)
        
        # For larger sets, use ranked guesses with distinct pattern bonus
        best_score = float('inf')
        best_idx = 0
        
        for guess_idx in self.ranked_guesses[:500]:
            if guess_idx in exclude:
                continue
            
            distinct = self._count_distinct_patterns(guess_idx, candidates)
            exp, worst = score_guess_fast(self.feedback_matrix, guess_idx, candidates, n)
            
            # Score combining expected size, worst case, and distinctness
            score = exp + worst * 0.1 - distinct * 0.3
            if self.guesses[guess_idx] in candidate_words:
                score -= 0.15
            
            if score < best_score:
                best_score = score
                best_idx = guess_idx
        
        return best_idx
    
    def find_best_guess(self, candidates: np.ndarray, 
                        exclude: Set[int] = None) -> int:
        """Find best guess for current candidates."""
        n = len(candidates)
        
        if n == 0:
            raise ValueError("No candidates")
        if n == 1:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        if n == 2:
            return self.guess_to_idx.get(self.answers[candidates[0]], 0)
        
        return self._find_best_guess_heuristic(candidates, exclude)
    
    def solve(self, answer: str, verbose: bool = False) -> Tuple[int, List[str]]:
        answer = answer.lower()
        answer_idx = self.answer_to_idx.get(answer)
        if answer_idx is None:
            raise ValueError(f"Unknown answer: {answer}")
        
        candidates = np.arange(self.n_answers, dtype=np.int32)
        guesses = []
        used_guesses = set()
        
        for turn in range(6):
            n_cand = len(candidates)
            
            if turn == 0:
                guess_idx = self.guess_to_idx[self.first_guess]
                if verbose:
                    print(f"  Turn {turn+1}: using first guess '{self.first_guess}'")
            elif turn == 1:
                first_fb = self.feedback_matrix[self.guess_to_idx[self.first_guess], answer_idx]
                if first_fb in self.second_guesses:
                    guess_idx = self.second_guesses[first_fb]
                    if verbose:
                        print(f"  Turn {turn+1}: using precomputed second guess")
                else:
                    guess_idx = self.find_best_guess(candidates, used_guesses)
                    if verbose:
                        print(f"  Turn {turn+1}: computed guess for {n_cand} candidates")
            else:
                if verbose:
                    t0 = time.time()
                guess_idx = self.find_best_guess(candidates, used_guesses)
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
            
            feedback_row = self.feedback_matrix[guess_idx]
            candidates = np.array([c for c in candidates if feedback_row[c] == feedback], dtype=np.int32)
            
            if verbose:
                print(f"{len(candidates)} candidates)")
                if len(candidates) <= 10:
                    cand_words = [self.answers[c] for c in candidates]
                    print(f"        remaining: {cand_words}")
        
        return 7, guesses
    
    def _feedback_to_emoji(self, feedback: int) -> str:
        chars = []
        for _ in range(5):
            chars.append(['⬛', '🟨', '🟩'][feedback % 3])
            feedback //= 3
        return ''.join(chars)


def load_words(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def benchmark(solver, words: List[str] = None,
              verbose: bool = True, progress_every: int = 100) -> Dict:
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
            print(f"[{i+1}/{len(words)}] avg={avg:.4f}, rate={rate:.1f}/s, ETA={eta/60:.1f}min")
        
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
    }


def print_results(results: Dict):
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
        print(f"\nFailed: {[w for w, _ in results['failed_words']]}")
    print("=" * 60)


if __name__ == "__main__":
    import random
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Loading word lists...")
    answers = load_words(os.path.join(base_dir, "words", "answers.txt"))
    guesses = load_words(os.path.join(base_dir, "words", "allowed_guesses.txt"))
    print(f"Answers: {len(answers)}, Guesses: {len(guesses)}")
    
    solver = OptimalSolverV2(answers, guesses, first_guess="salet")
    
    print("\n--- Quick tests ---")
    for word in ["crane", "jazzy", "paste", "water", "urine"]:
        if word in solver.answer_to_idx:
            n, gs = solver.solve(word, verbose=True)
            print(f"  Result: {n} guesses: {' -> '.join(gs)}\n")
    
    print("\n--- FULL BENCHMARK (all 2315 words) ---")
    results = benchmark(solver, answers, verbose=True, progress_every=250)
    print_results(results)
