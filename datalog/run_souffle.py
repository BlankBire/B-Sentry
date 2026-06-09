from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULES = os.path.join(HERE, "rules", "bsentry.dl")


def _has_souffle() -> bool:
    return shutil.which("souffle") is not None


def _has_wsl_souffle() -> bool:
    try:
        r = subprocess.run(
            ["wsl", "--", "which", "souffle"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and "souffle" in r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _win_to_wsl(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    if len(p) > 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def run(facts_dir: str, out_dir: str, rules: str = DEFAULT_RULES) -> int:
    os.makedirs(out_dir, exist_ok=True)
    if _has_souffle():
        cmd = ["souffle", "-F", facts_dir, "-D", out_dir, rules]
        print("[souffle] $", " ".join(cmd))
        return subprocess.call(cmd)

    if sys.platform.startswith("win") and _has_wsl_souffle():
        cmd = [
            "wsl", "--", "souffle",
            "-F", _win_to_wsl(facts_dir),
            "-D", _win_to_wsl(out_dir),
            _win_to_wsl(rules),
        ]
        print("[souffle/wsl] $", " ".join(cmd))
        return subprocess.call(cmd)

    print(
        "[souffle] not found on PATH and not available via WSL.\n"
        "          Use `python -m layer3.rule_engine` for the Python backend,\n"
        "          or install souffle (apt install souffle / brew install souffle-lang).",
        file=sys.stderr,
    )
    return 127


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=os.path.join(HERE, "facts"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--rules", default=DEFAULT_RULES)
    args = ap.parse_args()
    sys.exit(run(args.facts, args.out, args.rules))
