"""Check that prediction at t does not change when only r_t is flipped."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.da_akt_project.model import DifficultyAwareAKT


def main() -> None:
    torch.manual_seed(3407)
    model = DifficultyAwareAKT(n_questions=20, n_concepts=10, n_difficulty_bins=10, seq_len=8,
                               d_model=32, n_blocks=2, n_heads=4, d_ff=64, dropout=0.0)
    model.eval()
    q = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    c = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    r1 = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]])
    r2 = r1.clone()
    t = 4
    r2[0, t] = 1 - r2[0, t]
    m = torch.ones_like(q)
    d = torch.ones_like(q)
    with torch.no_grad():
        y1 = model(q, c, r1, m, d)[0, t]
        y2 = model(q, c, r2, m, d)[0, t]
    diff = float((y1 - y2).abs())
    print(f"target_position={t}, abs_logit_diff_after_flipping_current_label={diff:.10f}")
    assert diff < 1e-6, "Current response leaks into current prediction."
    print("PASS: strict previous-interaction protocol has no current-label leakage.")


if __name__ == "__main__":
    main()
