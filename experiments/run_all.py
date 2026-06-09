from __future__ import annotations

import subprocess
import sys
import time


SCRIPTS = [
    ("Exp-A — main results (RQ1)",      ["python", "-m", "experiments.exp_a_run_all"]),
    ("Exp-B — ablation (RQ2)",          ["python", "-m", "experiments.exp_b_ablation"]),
    ("Exp-C — latency (RQ3)",           ["python", "-m", "experiments.exp_c_latency"]),
    ("Exp-D — adversarial robustness",  ["python", "-m", "experiments.exp_d_adversarial"]),
    ("Exp-E — interpretability case study",
                                         ["python", "-m", "experiments.exp_e_interpretability"]),
    ("Exp-F — per-rule contribution",   ["python", "-m", "experiments.exp_f_rule_ablation"]),
    ("Report",                           ["python", "-m", "experiments.report"]),
]


def main():
    for title, cmd in SCRIPTS:
        print(f"\n========================== {title} ==========================")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - t0
        print(f"\n[{title}] exit code={r.returncode}  elapsed={elapsed:.1f}s")
        if r.returncode != 0:
            print(f"[{title}] FAILED — stopping.")
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
