# Clean RK45 reference fixtures

Start with [the handoff guide](../../../docs/WP03_RK45_HANDOFF.md). Six deterministic synthetic CSVs, 101 rows each, cover A/B regular dynamics, A finite-time extinction, B decline and two near-fold examples. This is a small reference handoff, not a final inverse dataset or ecological evidence.

`conditions.json` records declared known conditions and fixed scales for WP05's known-K design. `truth.json` and `manifest.json` are evaluation/provenance metadata and must not be passed to an inverse optimiser. No noise or train/validation/test split has been applied; WP05 owns those steps.

Regenerate using the command in the guide. Curated file hashes resolve from the repository root; original generated-output manifests resolve from their output directory.
