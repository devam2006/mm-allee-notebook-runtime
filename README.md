# Strong-Allee harvesting: notebook, code and synthetic data

[**Download the final notebook + EPS figures ZIP**](https://github.com/devam2006/mm-allee-notebook-runtime/releases/download/notebook-eps-v2/mm-allee-final-notebook-eps.zip)

The ZIP contains exactly one finalized notebook and 17 EPS figures, with no report, PDF or PNG exports. The notebook downloads its fixed public code/data runtime when run; internet is needed on first use.

[**Open in Google Colab**](https://colab.research.google.com/github/devam2006/mm-allee-notebook-runtime/blob/main/project.ipynb)

Connect a CPU runtime and choose **Runtime → Run all**. The small notebook downloads a fixed, checksum-verified snapshot of its code/data on the first run. No Drive mount, private repository permission or API key is required. Figures appear as the cells run.

The notebook covers WP01–WP05: classical equilibria/history/collapse, forward PINNs, inverse estimation and held-out validation. All population data are synthetic. Recorded long sweeps are loaded from tables, not retrained by the demonstration.

## Final notebook checks

The finalized notebook preserves the supplied WP04/WP05 presentation with approved factual corrections and the reviewed WP01–WP03 sections. It is 78,854 bytes (68 cells / 29 code cells). A fresh kernel executed all 29 code cells without errors in 36.12 seconds and displayed 16 figure outputs; all 531 source tests pass. The downloaded 18-file ZIP is 483,322 bytes, SHA-256 `421696d55a9feed67cf7723f6eb90d30111eb659d34abb8e0cfd07df4ceee48d`. The v1 release remains an earlier reproducible version.

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

The inverse trainer's legacy `converged` label denotes completed training; read the separate L-BFGS outcome. The recorded final sweep has 54 iteration-limit exits; clean parameter recovery is 10/18, including Model B near-fold 3/3 under the adopted setting. Good trajectory fits do not prove accurate parameters. This is a computational integration candidate; human review and the final Overleaf report/presentation remain separate. See NOTICE.md and docs/REFERENCES.md for attribution. AI tools assisted implementation and review under the members' direction. The private development history and discussions are not included here.
