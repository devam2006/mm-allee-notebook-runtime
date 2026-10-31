# Data provenance and labels

Default data are **synthetic population trajectories generated from the stated ODEs**, not field measurements. The project studies mathematical and numerical behaviour and parameter recovery under an assumed model. It does not validate an ecological law for a real species.

WP03 owns clean reference generation, stable IDs and manifests; WP05 owns observation selection, noise/split construction and validation. Their separate clean-reference and fitting-observation CSV schemas are in `INTERFACES.md`; the implemented clean handoff is documented in [WP03_RK45_HANDOFF.md](WP03_RK45_HANDOFF.md). WP03 exports no noisy labels or train/val/test splits. For WP05, use a seeded NumPy generator; default additive noise is `N_observed=N_reference+sigma*N_ref*epsilon`, `epsilon~Normal(0,1)`, with fixed externally declared `N_ref`. Do not clip negative noisy observations silently: either retain them as measurement noise or predeclare a different observation model and describe its statistical effects. Dynamics remain physically nonnegative up to the declared boundary.

Record generating parameters, solver/tolerances, initial conditions, interval, sampling grid, units/scales, seed, noise level, split IDs, code commit and file SHA256. Keep generator truth in evaluation metadata separate from the data passed to the optimiser. Test data must not enter parameter fitting, hyperparameter selection, loss weighting or early stopping.

WP04 (forward PINNs) uses no observation files: `experiments/wp04/run.py` generates its noise-free reference trajectories on the fly with SciPy DOP853 (rtol 1e-12, atol 1e-14) from the parameters in the `configs/wp04_*.json` files, only to *score* predictions (never as training labels). It has no noise, no train/validation/test split and no dataset files; this stands in for WP03's reference fixtures, which are not yet on `main`, and should be replaced by them when they merge. Its curated outputs are `artifacts/tables/wp04_*.csv` with manifests in `artifacts/manifests/wp04_*.json`; all are synthetic.

Any later external dataset requires original URL, author/provider, download date, license/usage terms, units, preprocessing decisions, train/val/test design and a bibliography entry. Every external/adapted figure or published formula needs a verified citation. No claim that synthetic benchmark success establishes reliable real-world prediction.


## WP05 submitted observation protocol

The submitted WP05 utilities add seeded Gaussian noise as `observed = clean + sigma*N_ref*epsilon`, with fixed external `N_ref=100` and declared `t_ref=1`. Configurations specify sigma 0/0.01/0.05 and seeds 11/22/33; observations are not clipped. The code assigns contiguous 60/20/20 index splits and uses training observations for the data objective and validation for checkpoint selection. Test values are scored after fitting. Physics collocation spans the complete observation horizon, so the held-out tail is physics-supported prediction, not extrapolation outside the collocation domain.

The original WP03 fixture files and evaluation-only `truth.json` come from PR #4. The ZIP's curated WP05 results and `wp05_manifest.json` are submitted evidence awaiting reproduction; their `completed` label is not accepted as a project status. Coverage and code/result inconsistencies are recorded in `WP05_SUBMISSION_REVIEW.md`. The original archive hash and imported file hashes are in `artifacts/manifests/wp05_submission.json`.
