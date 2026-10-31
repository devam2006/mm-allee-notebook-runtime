# Strong-Allee harvesting: notebook, code and synthetic data

[**Open the current notebook in Colab**](https://colab.research.google.com/github/devam2006/mm-allee-notebook-runtime/blob/main/project.ipynb) · [**Download notebook (.ipynb)**](https://github.com/devam2006/mm-allee-notebook-runtime/releases/download/notebook-eps-v4/project.ipynb) · [**Download notebook + 19 EPS figures**](https://github.com/devam2006/mm-allee-notebook-runtime/releases/download/notebook-eps-v4/mm-allee-final-notebook-eps.zip)

The notebook contains WP01–WP05. This version imports only Raghav's condensed WP01 presentation onto the latest Rohan WP05 version. Every non-WP01 cell and all 19 EPS figures are unchanged. The ZIP contains exactly one notebook and those EPS figures; report writing remains separate in Overleaf.

Connect a CPU runtime and choose **Runtime → Run all**. The small notebook downloads a fixed, checksum-verified snapshot of its code/data on first use. No Drive mount, private repository permission or API key is required. Figures appear as cells run. All population data are synthetic; recorded historical sweeps are read from tables.

## Current checks

- 88,495 bytes, 62 cells / 27 code cells; largest code cell 6,668 characters.
- Fresh kernel: all 27 code cells passed with zero errors in 45.83 seconds, displaying 17 PNG outputs.
- 195 focused WP01 and Colab-export tests passed. The unchanged source suite previously passed 531 tests; that full suite was not rerun locally for the WP01 presentation change.
- Models, trainers, configs, datasets, frozen evidence and runtime pin are unchanged.
- Notebook SHA-256: `f309913a7f32c191dc7f8d86d3b174f6cd063d41cc2896e9a99e05b37673b799`.
- ZIP: 20 files, 526,471 bytes; SHA-256: `ee61d1224e4eac36fe29c4c6068e7ec1400068cb95356bc00ed7f2956edce0c3`.

Previous versioned releases remain available and unchanged.

## Files

- `project.ipynb`: current Colab/Jupyter entry point.
- `downloads/`: current minimal archive and its separate checksum inventory.
- `runtime/`: fixed executable snapshot and manifest.
- `src/`, `experiments/`, `configs/`: readable scientific code and settings.
- `data/fixtures/wp03_reference/`: six clean synthetic trajectories and provenance.
- `datasets/`: reconstructed noisy observations with split/noise labels.
- `artifacts/figures/`: figure assets, including the two additional WP05 demo plots.
- `artifacts/tables/`, `artifacts/manifests/`: recorded experiment evidence.

Python 3.11+ is required. The setup installs missing scientific dependencies (NumPy, SciPy, Matplotlib, IPython and JAX). A repository download includes the local runtime ZIP for offline Jupyter use.

## Interpretation and attribution

WP05's newer clean and 1% noisy live demonstrations use different settings from the unchanged frozen 108-fit benchmark. They illustrate behavior; they do not replace that benchmark. The inverse trainer's legacy `converged` label denotes completed training; read the separate L-BFGS outcome. The recorded final sweep has 54 iteration-limit exits and 10/18 clean PINN recoveries, including Model B near-fold 3/3. Good trajectory fits do not prove accurate parameters.

Human reviews and the report/presentation remain separate. See NOTICE.md and docs/REFERENCES.md for attribution. AI tools assisted implementation and review under the members' direction. Private development discussions are not included here.
