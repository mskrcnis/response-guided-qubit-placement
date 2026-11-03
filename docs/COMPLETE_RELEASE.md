# QST consolidated data and code — v1.2.0

One archive contains the original study and the additional validation. No downloads from earlier releases are required. Use this complete release as the data/code citation:
https://github.com/mskrcnis/response-guided-qubit-placement/releases/tag/v1.2.0

## Contents

- `original_study/`: preserved source code, circuit/configuration inputs, all 7,020 original frozen artifact files, compiled circuits, probabilities, baseline comparisons, timing records, licenses and provenance.
- `validation/`: complete public A–E assessment with exhaustive candidates, additional search/timing/model-sensitivity records, source/native circuits, selected calibration inputs, numerical checks, CSV tables, figures and reproduction scripts.
- `SHA256SUMS.json`, `verify_package.py`: integrity manifest and verifier for this combined archive.
- `reproduce_all.py`: rebuild and audit both analyses from saved data; no new experimental searches.

The two analysis folders preserve distinct cohorts and endpoints. They are not a pooled dataset. Negative findings, unavailable conditions and resource-limit failures remain explicit. The additional validation does not establish matched-time superiority over Mapomatic, independent prospective confirmation or hardware validation. All reported outcomes are simulated; real calibration properties are inputs.

## Verify and reproduce

Run commands from this extracted directory. Python 3.12 was used.

```bash
python3 verify_package.py
python3 -m venv .venv
.venv/bin/python -m pip install -r original_study/requirements/core.txt
.venv/bin/python reproduce_all.py
```

The reproduction command audits original saved outputs, regenerates the original five numerical tables and seven numerical figures, and rebuilds all validation summaries and seven figures. It needs no credentials, private manuscript files, older release downloads or new quantum-device jobs. Install dependencies before working offline. Numerical summaries, not historical wall-clock timings or PDF creation timestamps, are the reproduction targets.

For validation-only summaries, the smaller dependency list is `validation/requirements-summary.txt`; run `python validation/scripts/reproduce_assessment.py` in that environment. For new solver runs, read the separate analysis protocols and dependency requirements; the combined saved-data command does not build all five third-party compilers or rerun the campaign.

Verify file hashes before reproduction: output generation can legitimately rewrite generated records. Work in a copy to preserve the original download.

## Provenance and licensing

Scientific files are retained byte-for-byte from the original public study and its public validation assessment. Historical version identifiers inside their provenance records identify the source snapshots, not additional required downloads. Source receipt links, frozen protocols, calibration timestamps, environment details, seed manifests and original licenses are included.

Original code is MIT. Research outputs and upstream calibration-derived data are CC BY 4.0 with attribution; third-party code retains its licenses. See both analysis folders' license and attribution files. Full upstream device archives and third-party build dependencies remain externally indexed; the selected calibration values, regional extracts and scientific records used in the analyses are included. Manuscripts, private prompts, credentials and unrelated project files are excluded. No DOI is claimed for the GitHub release.
