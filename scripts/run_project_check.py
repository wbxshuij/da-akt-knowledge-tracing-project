"""End-to-end project check for the course package.

Checks:
1. Python syntax/import compilation.
2. No-current-label-leakage invariant.
3. Quick DA-AKT training pipeline.
4. Quick Plain-AKT ablation pipeline.
5. Comparison report generation.
6. Required output files exist.
"""
import os
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def require_files(paths):
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Required output files missing: " + ", ".join(missing))


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    run([sys.executable, "-m", "compileall", "-q", "src", "scripts"])
    run([sys.executable, "scripts/sanity_check_no_leakage.py"])
    run([sys.executable, "-m", "src.da_akt_project.train", "--config", "configs/test.yaml"])
    run([sys.executable, "-m", "src.da_akt_project.train", "--config", "configs/test_plain_akt_ablation.yaml"])
    run([
        sys.executable,
        "scripts/compare_runs.py",
        "--run_a", "outputs/test_run", "--name_a", "DA-AKT",
        "--run_b", "outputs/test_plain_akt", "--name_b", "Plain-AKT",
        "--output", "outputs/ablation_comparison.md",
    ])
    require_files([
        "outputs/test_run/best_model.pt",
        "outputs/test_run/metrics.json",
        "outputs/test_run/history.csv",
        "outputs/test_run/test_predictions.csv",
        "outputs/test_run/diagnosis_predictions.csv",
        "outputs/test_run/concept_mastery.csv",
        "outputs/test_run/student_diagnosis.csv",
        "outputs/test_run/experiment_report.md",
        "outputs/test_plain_akt/metrics.json",
        "outputs/ablation_comparison.md",
        "outputs/ablation_comparison.csv",
    ])
    print("\nPASS: project check completed successfully.")


if __name__ == "__main__":
    main()
