import argparse
import os

import numpy as np
import pandas as pd


def generate(path: str, n_users: int = 120, n_questions: int = 300, n_concepts: int = 40,
             min_len: int = 30, max_len: int = 120, seed: int = 3407) -> None:
    rng = np.random.default_rng(seed)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    q_concept = {f"q{i}": f"c{int(rng.integers(1, n_concepts + 1))}" for i in range(1, n_questions + 1)}
    q_hardness = {f"q{i}": float(rng.normal(0.0, 0.8)) for i in range(1, n_questions + 1)}
    rows = []
    for u in range(1, n_users + 1):
        ability = rng.normal(0.0, 0.8)
        skill_mastery = rng.normal(ability, 0.5, size=n_concepts + 1)
        seq_len = int(rng.integers(min_len, max_len + 1))
        for t in range(1, seq_len + 1):
            qid = f"q{int(rng.integers(1, n_questions + 1))}"
            cid = q_concept[qid]
            cidx = int(cid[1:])
            logit = skill_mastery[cidx] - q_hardness[qid] + 0.01 * t + rng.normal(0, 0.15)
            p = 1.0 / (1.0 + np.exp(-logit))
            correct = int(rng.random() < p)
            skill_mastery[cidx] += 0.06 if correct else 0.025
            rows.append([f"s{u}", qid, cid, correct, t])
    pd.DataFrame(rows, columns=["user_id", "question_id", "concept_id", "correct", "timestamp"]).to_csv(path, index=False)
    print(f"Wrote {len(rows)} interactions to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample_interactions.csv")
    parser.add_argument("--n_users", type=int, default=120)
    parser.add_argument("--n_questions", type=int, default=300)
    parser.add_argument("--n_concepts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    generate(args.output, args.n_users, args.n_questions, args.n_concepts, seed=args.seed)
