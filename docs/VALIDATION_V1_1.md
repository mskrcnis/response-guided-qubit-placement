# Additional validation: v1.1.0

This release preserves v1.0.0 and adds the separate focused assessment completed on 3 November 2026. The deposit does not assert that the manuscript has incorporated the additional findings.

## Download and verify

```bash
curl -L --fail -o qst-validation-assessment-v1.1.0.zip \
  https://github.com/mskrcnis/response-guided-qubit-placement/releases/download/v1.1.0/qst-validation-assessment-v1.1.0.zip
echo '26c74c410ea62068102b2081e635156af83f90cae09f940936d533379b2f1d47  qst-validation-assessment-v1.1.0.zip' | sha256sum --check
unzip qst-validation-assessment-v1.1.0.zip
cd QST_validation_assessment
python3 scripts/verify_manifest.py
python3 -m venv .venv-summary
.venv-summary/bin/python -m pip install -r requirements-summary.txt
.venv-summary/bin/python scripts/reproduce_assessment.py
```

Summary reproduction needs NumPy and Matplotlib, no private manuscript, credentials or new compilation searches. Experiment reruns have separate dependencies and resource requirements in `START_HERE.md` and `VALIDATION_PROTOCOL.md`. Original experimental kernels and all numerical inputs/results are unchanged from the validated assessment.

## Evidence

- A: 75,600 candidate/vector evaluations, 150 circuit/vector cells.
- B: 7,560 exhaustive placements and 4,590 historical/new search records.
- C: nine fresh processes and 324 evolving-pool searches.
- D: 480 budget checkpoints from 160 process jobs; three cold 0.5-second outputs missed initialization.
- E: 2,322 selected-output/model rows, including eligible recorded-parameter relaxation; six bounded scaling probes.

The hybrid score improves mean selection over product-success on identical candidates, but fully first-order scoring has slightly better mean selected TVD. The swarm offers a small fixed-proposal mean gain over equal-coverage random search. Multistart LightSABRE-Mapomatic has lower mean TVD at each tested budget and cache condition on the preselected five-circuit panel. Both synthetic and recorded-parameter relaxation reverse the extension-panel mean advantage. These are narrower, mixed findings, not broad compiler superiority.

Later eligible calibration dates remain unavailable under the unchanged freshness rule. Recorded T1/T2 values block 108 method outcomes on 29 October. Both ten-qubit coefficient builds fail the declared 4 GiB address-space cap; six- and eight-qubit probes complete. No paid or physical-device jobs were run.

## Contents and boundaries

The archive contains the frozen protocol, configurations, calibration vectors/receipts, source and native circuits, candidate scores, output distributions, timing and failure records, numerical checks, summary CSVs, seven PDF/PNG figures, report, claims table, primary-source comparison and reproduction scripts. Research inputs and outcomes remain available even when unfavorable.

Manuscript drafts, private task instructions, editorial patches, unrelated local file inventory and operational logs are excluded. Distribution changes are documented in `provenance/public_package_scope.json`. All 59 regenerated numerical files and seven PNG figures matched the validated local assessment. File integrity checks are distinct from numerical checks.

Original code is MIT; original research outputs and upstream calibration-derived data are CC BY 4.0 with attribution. Third-party code retains its own terms. GitHub provides a versioned release URL; no DOI or independent archival deposit is claimed.
