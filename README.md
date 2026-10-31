# Strong-Allee harvesting: notebook, code and synthetic data

[**Open in Google Colab**](https://colab.research.google.com/github/devam2006/mm-allee-notebook-runtime/blob/main/project.ipynb)

Connect a CPU runtime and choose **Runtime → Run all**. The small notebook downloads a fixed, checksum-verified snapshot of its code/data on the first run. No Drive mount, private repository permission or API key is required. Figures appear as the cells run.

The notebook covers WP01–WP05: classical equilibria/history/collapse, forward PINNs, inverse estimation and held-out validation. All population data are synthetic. Recorded long sweeps are loaded from tables, not retrained by the demonstration.

## Files

- `project.ipynb`: small Colab/Jupyter entry point.
- `runtime/`: fixed executable snapshot and its manifest.
- `src/`, `experiments/`, `configs/`: readable scientific code and settings.
- `data/fixtures/wp03_reference/`: six frozen clean trajectories and provenance.
- `datasets/`: reconstructed noisy observations with explicit split/noise labels.
- `artifacts/figures/`: PNG/EPS figure pairs and available PDFs for the report.
- `artifacts/tables/`, `artifacts/manifests/`: recorded experiment evidence.

Python 3.11+ is required. The setup cell installs only missing standard scientific dependencies (NumPy, SciPy, Matplotlib, IPython, JAX). Internet is needed on first use; a downloaded repository copy also includes the local runtime ZIP for offline Jupyter use.

## Interpretation and attribution

The inverse trainer's legacy `converged` label denotes completed training; read the separate L-BFGS outcome. The recorded final sweep has 54 iteration-limit exits; clean parameter recovery is 10/18. Good trajectory fits do not prove accurate parameters. This is a computational integration candidate; human review and the final Overleaf report/presentation remain separate. See NOTICE.md and docs/REFERENCES.md for attribution. AI tools assisted implementation and review under the members' direction. The private development history and discussions are not included here.
