# Response-guided qubit placement — complete data and code

Code and research data for **Circuit-response-guided swarm placement under changing quantum-device calibration**.

## One complete release

Use **[release v1.2.0](https://github.com/mskrcnis/response-guided-qubit-placement/releases/tag/v1.2.0)**. Its `qst-data-code-v1.2.0.zip` asset contains the complete original study and additional validation in one download. You do not need either older release.

Download the named data/code ZIP and its SHA-256 file from the release. GitHub's automatically generated “Source code” ZIP contains the repository checkout, not the large experimental datasets.

```bash
sha256sum --check qst-data-code-v1.2.0.zip.sha256
unzip qst-data-code-v1.2.0.zip
cd QST_Data_Code
python3 verify_package.py
python3 -m venv .venv
.venv/bin/python -m pip install -r original_study/requirements/core.txt
.venv/bin/python reproduce_all.py
```

Python 3.12 was used. The combined saved-data command needs no private files, credentials, older release downloads or quantum-device jobs. See [complete release guide](docs/COMPLETE_RELEASE.md).

## Archive contents

| Folder or file | Contents |
| --- | --- |
| `original_study/` | Original code and all 7,020 frozen artifact files: fifteen circuits, calibration inputs, native executables, probabilities, baseline comparisons and timing records |
| `validation/` | Complete additional A–E assessment: exhaustive candidates, search/cost/time-budget experiments, extra circuits/regions, noise-model sensitivity, failures, numerical checks, tables and figures |
| `reproduce_all.py` | Audit and regenerate both analyses from saved records |
| `verify_package.py`, `SHA256SUMS.json` | Combined file-integrity verification |
| `REPRODUCTION_CHECK.json` | Recorded isolated-copy reproduction result |

The two analysis folders preserve distinct protocols, cohorts and endpoints; they are not a pooled dataset. Historical version identifiers inside frozen provenance describe the source records and do not mean additional downloads are needed. Older releases remain as history.

## Interpretation

The original study covers fifteen four-/five-qubit circuits, a six-node IBM Fez region, eleven paired October cutoffs, internal search controls and five configured compiler baselines. The additional validation separates objective quality, exhaustive coverage, coefficient reuse, matched-time performance and model sensitivity.

The hybrid score improves on product-success scoring on identical candidates, but not on fully first-order scoring in mean selected TVD. Swarm gives a small fixed-proposal mean improvement over random search. Multistart Mapomatic has lower mean TVD at the tested time budgets, and relaxation-model changes reverse the extension-panel mean gain. No physical-device execution, independent prospective confirmation or broad compiler superiority is claimed. The deposited validation does not assert that the manuscript has already incorporated its findings.

## Availability and licensing

[Data and code availability](docs/AVAILABILITY.md). Cite the single consolidated release identified in `CITATION.cff`. No DOI is claimed.

Original code is MIT. Research outputs and upstream calibration-derived data are CC BY 4.0 with attribution. Third-party software retains its own licenses. Full upstream device archives and compiler build dependencies remain indexed externally; scientific records needed for the saved-data analyses are included. Manuscripts, private prompts, credentials and unrelated project files are excluded.

The lightweight repository retains the original scientific code and CLI for inspection. For complete reproduction, use the consolidated release asset above.
