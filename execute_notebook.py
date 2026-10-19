#!/usr/bin/env python3
"""Execute ``modeltrain.ipynb`` in a fresh kernel and save the outputs.

Usage:
    python execute_notebook.py [--no-save] [--timeout 3600]

The script fails (exit code 1) if any cell raises, and prints the failing cell
number plus the exception so the notebook can be fixed.  On success it prints
the group-CV metric and self-check totals from ``training_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=str(ROOT / "modeltrain.ipynb"))
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell timeout (s)")
    parser.add_argument("--no-save", action="store_true", help="do not write outputs back")
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}", file=sys.stderr)
        return 2

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )

    print(f"Executing {notebook_path.name} in a fresh kernel ...", flush=True)
    try:
        client.execute()
    except CellExecutionError as error:
        print("\nNOTEBOOK EXECUTION FAILED", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1

    if not args.no_save:
        nbformat.write(notebook, notebook_path)
        print(f"Saved executed notebook to {notebook_path.name}")

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    print(f"Cells executed: {len(code_cells)} Python, {len(markdown_cells)} markdown")

    report_path = ROOT / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        validation = report.get("validation", {})
        ensemble = validation.get("results", {}).get("ensemble", {})
        baseline = validation.get("baseline", {})
        checks = report.get("self_checks", {})
        print(
            "Grouped CV ROC-AUC: "
            f"baseline={baseline.get('mean_roc_auc'):.5f}, "
            f"ensemble={ensemble.get('mean_auc'):.5f}"
        )
        print(f"Self-checks: {checks.get('passed')}/{checks.get('total')} passed")
        print(
            "Notebook Python cells: "
            f"{report.get('notebook', {}).get('python_cells')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
