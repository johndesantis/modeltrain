"""Execute modeltrain.ipynb in a fresh kernel using this Python environment."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-binary", action="store_true",
        help="Fail before training/export unless both benign and jailbreak labels exist.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    notebook_path = root / "modeltrain.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    # A temporary kernelspec avoids installing a global kernel or using a Python
    # interpreter that does not have this environment's pinned dependencies.
    with tempfile.TemporaryDirectory(prefix="ha-none-kernel-") as temporary:
        kernel_dir = Path(temporary) / "kernels" / "ha-none"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "HA None isolated execution",
            "language": "python",
        }))
        old_jupyter_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temporary + (
            os.pathsep + old_jupyter_path if old_jupyter_path else "")
        try:
            kernel_env = dict(os.environ)
            if args.require_binary:
                kernel_env["HA_NONE_REQUIRE_BINARY"] = "1"
            NotebookClient(
                notebook, timeout=600, kernel_name="ha-none", allow_errors=False,
                resources={"metadata": {"path": str(root)}},
            ).execute(env=kernel_env)
        finally:
            if old_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old_jupyter_path
    # Use the portable kernel name when opened interactively elsewhere.
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text, end="")
    print("Notebook completed successfully in a fresh kernel.")


if __name__ == "__main__":
    main()
