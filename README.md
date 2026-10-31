# Strong-Allee harvesting: notebook, code and synthetic data

[**Open in Colab**](https://colab.research.google.com/github/devam2006/mm-allee-notebook-runtime/blob/main/project.ipynb) · [**Download notebook**](https://github.com/devam2006/mm-allee-notebook-runtime/releases/download/notebook-eps-v5/project.ipynb) · [**Notebook + 18 EPS figures**](https://github.com/devam2006/mm-allee-notebook-runtime/releases/download/notebook-eps-v5/mm-allee-final-notebook-eps.zip)

WP03 now presents numbered subsections 3.1–3.6: critical harvesting rates, basins, extinction time, numerical validation, qualitative slow decline and reference data. Its core runner skips sensitivity, yield optimisation, policy margins and bottleneck asymptotic comparisons. The owners' finalized WP01 and WP05 notebook sections, and all WP02/WP04 cells, are unchanged.

[**Editable WP03 Overleaf update**](https://github.com/devam2006/mm-allee-notebook-runtime/releases/download/notebook-eps-v5/wp03-overleaf-update.zip) · [**WP03 LaTeX section**](https://github.com/devam2006/mm-allee-notebook-runtime/releases/download/notebook-eps-v5/03_collapse.tex)

Copy the section and its three EPS files into the current Overleaf project, preserving teammates' work. This is an editable section update; no report PDF is supplied.

## Run and verification

Connect a CPU runtime and choose **Runtime → Run all**. First use downloads a fixed 1.57 MB runtime and verifies its SHA-256. No Drive mount, private repository access or API key is needed. Figures appear when cells run. All population examples are synthetic.

- Notebook: 88,192 bytes, 61 cells / 27 code cells; largest code cell 6,668 characters.
- Fresh kernel using the new public runtime: **27 code cells, zero errors, 33.34 seconds**, 16 PNG outputs.
- Complete source suite: **533 tests passed**. Core-mode regressions verify optional routines are not called and core evidence matches historical results.
- Shared models, root/threshold helpers, robust extinction solver, PINN code/configs and reference fixtures are unchanged. The historical full WP03 command remains available.
- Notebook SHA-256: `0d1c8d95cf803d7360b21f2c9a3d45e521825a2cd118a7d31e613be0202322eb`.
- Minimal ZIP: 19 files, 510,803 bytes; SHA-256: `2a5a2199b67fbe2b30bc99698896d5f1d4ed75115c51c0e37a803a59075acd7a`.
- WP03 report update: five files, 166,855 bytes; SHA-256: `e08a324ac37ed86a033e40d10b104a58bc3ee29e5a297f2f80fe5ae311d10c94`.

The current ZIP replaces WP03's historical two-panel bottleneck and optional sensitivity/yield figure with a trajectories-only plot. Other owners' EPS files are unchanged. Earlier versioned releases remain available.

## Resources and interpretation

`runtime/` contains immutable snapshots; `src/`, `experiments/` and `configs/` provide readable code/settings. `data/fixtures/wp03_reference/` contains the six clean trajectories; `datasets/` contains reconstructed noisy observations with split/noise labels. Figures, tables and provenance are in `artifacts/`; current downloads and separate checksum inventories are in `downloads/`.

Python 3.11+ is required. The setup installs missing scientific dependencies. The historical long sweeps are loaded from evidence tables rather than silently retrained.

WP05's newer live demonstrations retain their owner-finalized settings. They are distinct from the frozen 108-fit benchmark: 10/18 clean PINN recoveries, including B near-fold 3/3. The legacy inverse `converged` label denotes completed training; inspect the optimizer outcome. Good trajectory fits do not establish parameter recovery.

Human review, understanding and final report/presentation remain separate. See NOTICE.md and docs/REFERENCES.md for attribution. AI assisted implementation and checking under the members' direction.
