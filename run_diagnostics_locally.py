"""
Run jobs/diagnostic/*.sh scripts locally.

Each .sh is a SLURM job designed for the HPC cluster. This script:
  1. Reads every .sh file in the target folder.
  2. Extracts the `srun python -u csdi_train_modified.py ...` command
     (including backslash-continued lines).
  3. Strips the `srun` prefix and the `-u` flag, leaving a plain
     `python csdi_train_modified.py ...` invocation.
  4. Extracts `export KEY=VALUE` lines and injects them into the
     subprocess environment (needed for WANDB_API_KEY, WANDB_MODE, etc.).
  5. Applies local overrides (fewer epochs, disable AMP, etc.).
  6. Runs each command sequentially in the repo root.
"""

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Local overrides applied on top of whatever the .sh file specifies.
# ---------------------------------------------------------------------------
OVERRIDES: dict[str, str] = {
    "--epochs": "30",
    "--use_amp": "false",
    "--print_plots": "false",
    "--wandb_project": "overfit-local",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_srun_command(text: str) -> str:
    """Return the full `srun python ...` command, joining backslash continuations."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*srun\s+python\b", line):
            parts = [line.rstrip()]
            j = i + 1
            while parts[-1].endswith("\\") and j < len(lines):
                parts[-1] = parts[-1][:-1].rstrip()
                parts.append(lines[j].rstrip())
                j += 1
            return " ".join(parts)
    raise ValueError("No `srun python ...` command found in script.")


def extract_env_exports(text: str) -> dict[str, str]:
    """Return a dict of every `export KEY=VALUE` line in the script."""
    env: dict[str, str] = {}
    for match in re.finditer(r"^\s*export\s+([A-Z_][A-Z0-9_]*)=(.+)$", text, re.MULTILINE):
        key = match.group(1)
        # Strip surrounding quotes if present
        value = match.group(2).strip().strip("'\"")
        env[key] = value
    return env


def build_argv(srun_cmd: str) -> list[str]:
    """
    Convert:
        srun python -u csdi_train_modified.py <args>
    into:
        python csdi_train_modified.py <args>
    """
    argv = shlex.split(srun_cmd)

    if not argv or argv[0] != "srun":
        raise ValueError(f"Command does not start with `srun`: {srun_cmd!r}")
    argv = argv[1:]  # drop srun

    if len(argv) < 2 or argv[0] != "python":
        raise ValueError(f"Expected `python` after `srun`, got: {argv!r}")

    out = ["python"]
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "-u":            # unbuffered flag — not needed locally
            i += 1
            continue
        if token.endswith("csdi_train_modified.py"):
            out.append("csdi_train_modified.py")
            i += 1
            break
        out.append(token)
        i += 1

    out.extend(argv[i:])
    return out


def upsert_flag(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace `flag value` if already present, otherwise append it."""
    out: list[str] = []
    i = 0
    found = False
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            out.extend([flag, value])
            found = True
            i += 2
        else:
            out.append(argv[i])
            i += 1
    if not found:
        out.extend([flag, value])
    return out


def apply_overrides(argv: list[str]) -> list[str]:
    for flag, value in OVERRIDES.items():
        argv = upsert_flag(argv, flag, value)
    return argv


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scripts(
    scripts_dir: Path,
    workdir: Path,
    dry_run: bool,
    stop_on_error: bool,
    only: list[str] | None,
) -> int:
    sh_files = sorted(glob.glob(str(scripts_dir / "*.sh")))
    if not sh_files:
        print(f"ERROR: No .sh files found in {scripts_dir}", file=sys.stderr)
        return 1

    if only:
        sh_files = [f for f in sh_files if Path(f).stem in only]
        if not sh_files:
            print(f"ERROR: None of the specified --only names matched.", file=sys.stderr)
            return 1

    print(f"Found {len(sh_files)} script(s) to run.\n")

    for idx, sh_path in enumerate(sh_files, start=1):
        name = Path(sh_path).name
        print(f"[{idx}/{len(sh_files)}] {name}")
        print("-" * 60)

        text = Path(sh_path).read_text(encoding="utf-8")

        try:
            srun_cmd = extract_srun_command(text)
        except ValueError as exc:
            print(f"  SKIP — {exc}\n")
            continue

        argv = build_argv(srun_cmd)
        argv = apply_overrides(argv)

        # Merge `export KEY=VALUE` lines from the script into subprocess env.
        script_env = extract_env_exports(text)
        env = {**os.environ, **script_env}

        print("  Command :", shlex.join(argv))
        if script_env:
            print("  Env     :", {k: (v[:8] + "...") if len(v) > 12 else v
                                   for k, v in script_env.items()})

        if dry_run:
            print("  [dry-run] not executed.\n")
            continue

        result = subprocess.run(argv, cwd=workdir, env=env, check=False)
        print(f"  Exit code: {result.returncode}\n")

        if result.returncode != 0 and stop_on_error:
            print("Stopping due to non-zero exit code.", file=sys.stderr)
            return result.returncode

    return 0


def main() -> None:
    repo_root = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(
        description="Run jobs/diagnostic/*.sh scripts locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scripts_dir",
        default=str(repo_root / "jobs" / "diagnostic"),
        help="Folder containing the .sh files.",
    )
    parser.add_argument(
        "--workdir",
        default=str(repo_root),
        help="Working directory for each subprocess (repo root by default).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print rewritten commands without executing them.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop after the first script that exits non-zero.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEM",
        default=None,
        help=(
            "Run only the scripts whose filename stem (without .sh) matches "
            "one of these values. Example: --only A_run_1_seq_len_32_train_size_1_time_bin_1"
        ),
    )
    args = parser.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    workdir = Path(args.workdir).resolve()

    if not scripts_dir.is_dir():
        print(f"ERROR: scripts_dir does not exist: {scripts_dir}", file=sys.stderr)
        sys.exit(1)
    if not workdir.is_dir():
        print(f"ERROR: workdir does not exist: {workdir}", file=sys.stderr)
        sys.exit(1)

    sys.exit(
        run_scripts(
            scripts_dir=scripts_dir,
            workdir=workdir,
            dry_run=args.dry_run,
            stop_on_error=args.stop_on_error,
            only=args.only,
        )
    )


if __name__ == "__main__":
    main()
