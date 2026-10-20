#!/usr/bin/env python3
"""
execute_notebook.py
Execute ``modeltrain.ipynb`` in a fresh kernel and fail loudly on any cell error.

Usage
-----
    python execute_notebook.py                # run, do not overwrite the notebook
    python execute_notebook.py --save         # run and store the outputs back
    python execute_notebook.py --save --timeout 3600

The script uses the *active* Python interpreter, so the notebook always runs
against the same environment that installed the dependencies.
"""

import argparse
import os
import sys
import time

NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeltrain.ipynb")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=NOTEBOOK, help="path to the notebook")
    parser.add_argument("--save", action="store_true",
                        help="write the executed notebook (with outputs) back to disk")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="per-cell timeout in seconds")
    parser.add_argument("--kernel", default="python3", help="kernel name to use")
    args = parser.parse_args()

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: {exc}\nInstall with: pip install nbformat nbclient ipykernel",
              file=sys.stderr)
        return 2

    if not os.path.exists(args.notebook):
        print(f"ERROR: notebook not found: {args.notebook}", file=sys.stderr)
        return 2

    notebook = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(args.notebook))}},
    )

    print(f"Executing {args.notebook} with kernel '{args.kernel}' ...", flush=True)
    started = time.time()
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001 - surface the failing cell
        print(f"\nFAILED after {time.time() - started:.1f}s: {exc}", file=sys.stderr)
        # Print the outputs captured so far so the failing cell is identifiable.
        for index, cell in enumerate(notebook.cells):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    print(f"\n--- error in cell {index} ---", file=sys.stderr)
                    print(output.get("ename"), ":", output.get("evalue"), file=sys.stderr)
                    print("\n".join(output.get("traceback", [])), file=sys.stderr)
        return 1

    elapsed = time.time() - started
    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    print(f"OK: {code_cells} code cells executed in {elapsed:.1f}s")

    if args.save:
        nbformat.write(notebook, args.notebook)
        print(f"Saved executed notebook to {args.notebook}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
