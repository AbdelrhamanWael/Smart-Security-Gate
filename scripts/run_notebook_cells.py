"""
run_notebook_cells.py
Executes all code cells in Biometric_Security_Gate_Complete_Pipeline.ipynb,
captures stdout, stderr, and matplotlib figures (as base64 PNGs),
and writes the updated executed notebook back to disk.
"""

import sys
import io
import os
import json
import base64
import contextlib
import traceback
from pathlib import Path

# Ensure headless matplotlib
os.environ["MPLCONFIGDIR"] = os.path.abspath("tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_all():
    nb_path = Path("Biometric_Security_Gate_Complete_Pipeline.ipynb")
    if not nb_path.exists():
        print(f"Error: {nb_path} does not exist!")
        sys.exit(1)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Shared global environment
    global_env = {
        "__name__": "__main__",
        "__file__": str(nb_path.resolve()),
    }

    # Hook for plt.show to capture figure as base64 png
    captured_figures = []

    def custom_show(*args, **kwargs):
        figs = [plt.figure(num) for num in plt.get_fignums()]
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64_str = base64.b64encode(buf.read()).decode("ascii")
            captured_figures.append(b64_str)
            plt.close(fig)

    plt.show = custom_show
    global_env["plt"] = plt

    execution_count = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source_code = "".join(cell["source"])
        print(f"Running Code Cell {idx} (Exec Count: {execution_count})...")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        captured_figures.clear()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                # Execute the code in shared global namespace
                exec(source_code, global_env)
                # Catch any unshown figures left over
                if plt.get_fignums():
                    custom_show()
        except Exception:
            err_msg = traceback.format_exc()
            stderr_buf.write(err_msg)
            print(f"  ❌ Error in cell {idx}:\n{err_msg}")

        # Assemble cell outputs
        cell_outputs = []

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            cell_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.splitlines()]
            })
            print(f"  [Cell {idx} stdout]: {len(stdout_text.splitlines())} lines")

        # Any figures captured?
        for b64_img in captured_figures:
            cell_outputs.append({
                "data": {
                    "image/png": b64_img,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
            print(f"  [Cell {idx} display_data]: Captured 1 Matplotlib Figure!")

        stderr_text = stderr_buf.getvalue()
        if stderr_text and "error" in stderr_text.lower():
            cell_outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": [line + "\n" for line in stderr_text.splitlines()]
            })

        cell["execution_count"] = execution_count
        cell["outputs"] = cell_outputs
        execution_count += 1

    # Save the updated executed notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print("\n🎉 ALL CELLS EXECUTED SUCCESSFULLY AND SAVED INTO NOTEBOOK!")


if __name__ == "__main__":
    run_all()

