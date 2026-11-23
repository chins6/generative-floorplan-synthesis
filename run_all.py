"""Reproduce every result in the report from scratch (~30-45 min on a laptop CPU).

    python run_all.py                    # both label sets + comparison
    python run_all.py --skip-training    # only selection + evaluation on the saved checkpoints
    python run_all.py --labels walls     # one label set only

For each label set ('ink' = dark pixels of the PNG, mid-review; 'walls' = official SVG
wall polygons): train DAE, the VAE beta sweep {0.1,0.5,1,4} and the GNN depth sweep
{2,4,6} (90 epochs each) -> choose beta / depth on validation -> benchmark the best
checkpoints on test. Finally compare the two label sets.
"""
import argparse
import subprocess
import sys

PY = sys.executable
TRAIN_JOBS = (
    [["--model", "dae"]]
    + [["--model", "vae", "--beta", b, "--tag", f"beta{b}"] for b in ("0.1", "0.5", "1.0", "4.0")]
    + [["--model", "gnn", "--layers", L, "--tag", f"L{L}"] for L in ("2", "4", "6")]
)


def run(args):
    print("\n$ python", " ".join(args), flush=True)
    subprocess.run([PY] + args, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--labels", choices=["ink", "walls", "both"], default="both")
    opts = ap.parse_args()
    label_sets = ["ink", "walls"] if opts.labels == "both" else [opts.labels]
    run(["gradient_check.py"])
    for labels in label_sets:
        if not opts.skip_training:
            for job in TRAIN_JOBS:
                run(["train.py", "--labels", labels] + job)
        run(["select_models.py", "--labels", labels])
        run(["evaluate.py", "--labels", labels])
    if opts.labels == "both":
        run(["compare_labels.py"])
