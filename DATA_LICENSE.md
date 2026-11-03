# Data licensing and attribution

The calibration records are derived from **Qiskit Calibration Drift**, by
**phanerozoic**, available at
https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift.
The recorded dataset metadata declares **Creative Commons Attribution 4.0
International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/.
The original metadata is preserved in `provenance/calibration_dataset_metadata.json`.

Changes made by this study: selection of the IBM Fez six-qubit region,
receipt- and timestamp-based eligibility filtering, conversion to daily
parameter vectors, and derived calibration summaries. The release includes
the regional extracts and selected parameter vectors, with source revisions,
retrieval receipts and hashes. Full-device upstream archives are linked,
rather than duplicated, in `provenance/upstream_calibration_files.json`.
No endorsement by the upstream dataset author or IBM is implied.

Original circuit descriptions, simulated output records, derived numerical
results and figure data in this repository and its release asset are made
available under CC BY 4.0. Cite the versioned repository and retain upstream
attribution for the calibration-derived parts. These are simulated research
outputs, not new physical-device measurements.

Original program code is licensed under MIT; see `LICENSE`.
