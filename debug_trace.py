"""Debug script for tracing solver behavior."""

from src.feedback import compute_feedback, feedback_to_string
from src.solver import WordleSolver
from src.endgame_search import partition_candidates, compute_entropy, EndgameSearcher

def trace_solve(answer, starting_word='salet'):
    solver = WordleSolver(starting_word=starting_word)
    solver.reset()
    
    print(f"\n=== Tracing solve for: {answer} ===\n")
    
    # Get access to the internal endgame searcher
    endgame = solver._solver.endgame
    
    max_guesses = 6
    for i in range(max_guesses):
        cands = solver.get_candidates()
        print(f"Turn {i+1}: {len(cands)} candidates")
        if len(cands) <= 10:
            print(f"  Candidates: {cands}")
            
            # Show entropy for each candidate as a guess
            for c in cands:
                parts = partition_candidates(c, cands)
                ent = compute_entropy(parts)
                print(f"    entropy({c}) = {ent:.4f}, partitions: {len(parts)}")
            
            # Show what endgame searcher returns
            if len(cands) > 2:
                print(f"\n  Endgame searcher analysis:")
                # Evaluate a few guesses explicitly
                test_guesses = cands[:3] + ['story', 'bawdy'] if 'bawdy' in cands else cands[:5]
                for g in test_guesses:
                    if g in solver._solver.all_guesses:
                        endgame.clear_cache()  # Clear cache for fair evaluation
                        exp = endgame._evaluate_guess(g, cands, 0, None)
                        print(f"    _evaluate_guess({g}) = {exp:.4f}")
                
                endgame.clear_cache()  # Clear cache before main call
                best_g, best_exp = endgame.expected_guesses(cands, 0, None, debug=True)
                print(f"    => Endgame choice: {best_g} (expected: {best_exp:.4f})")
        
        guess = solver.next_guess()
        fb = compute_feedback(guess, answer)
        fb_str = feedback_to_string(fb)
        
        # Also show entropy for the chosen guess
        parts = partition_candidates(guess, cands)
        ent = compute_entropy(parts)
        
        print(f"  Guess: {guess} -> {fb} (entropy={ent:.4f})")
        
        if fb == 242:  # All green
            print(f"\n✓ Solved in {i+1} guesses!")
            return i + 1
        
        solver.update(guess, fb)
        
        # Check if answer is still in candidates
        new_cands = solver.get_candidates()
        if answer not in new_cands:
            print(f"  ERROR: {answer} not in remaining candidates!")
            print(f"  Remaining: {new_cands[:20]}")
            break
    
    print(f"\n✗ Failed to solve in {max_guesses} guesses")
    return max_guesses + 1


if __name__ == "__main__":
    # Test problematic words
    for word in ["jazzy"]:
        trace_solve(word)
