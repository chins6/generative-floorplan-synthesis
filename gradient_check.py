"""Verify the hand-written backward pass of every model against finite differences.

    python gradient_check.py
"""
from floorplan.gradcheck import gradient_check

if __name__ == "__main__":
    print("Numerical gradient check (float64, |num - ana| <= 1e-5 + 1e-3 |ana|):")
    results = [gradient_check(name, seed=s)[0] for name in ("DAE", "VAE", "GNN") for s in (0, 1)]
    print("ALL BACKWARD PASSES CORRECT" if all(results) else "SOME GRADIENTS ARE WRONG")
