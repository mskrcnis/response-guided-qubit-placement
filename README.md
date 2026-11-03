# Response-guided qubit placement

Code and research data for **Circuit-response-guided swarm placement under
changing quantum-device calibration**.

This artifact preserves the implemented response objective, duplicate-aware
permutation swarm, internal controls, five configured compiler baselines,
calibration inputs, simulated outcomes, and manuscript plot/table data.

## Study scope

- 15 circuit instances, with four or five logical qubits on a six-qubit IBM Fez region.
- 11 paired calibration cutoffs, 19–29 October 2026; 18 October initializes the histories.
- 3 search replicates and 2 seed schedules; 56 proposals per internal search.
- 3,240 internal searches, 2,055 distinct external compilation requests,
  5,940 paired external-comparison method observations, and 270 serial timing trials.
- Exact noisy simulation under independent depolarizing gates and asymmetric
  readout assignment. No physical-device execution is reported.

The mean simulated TVD favors the modified swarm in the reported groups, but
there are ring-circuit losses and substantially greater compilation cost.
The observed period is exploratory and was already exposed during development.
This is not independent confirmation, a matched-runtime superiority claim,
or a learned controller that decides whether to recompile before searching.
See [results and interpretation](docs/INTERPRETATION.txt).

## Additional validation release

[Release v1.1.0](https://github.com/mskrcnis/response-guided-qubit-placement/releases/tag/v1.1.0)
adds a separately reproducible assessment of objective ranking, exhaustive
placement/search coverage, coefficient reuse, matched time budgets, and
noise-model sensitivity. The original v1.0.0 artifacts remain unchanged.
See [validation data and reproduction](docs/VALIDATION_V1_1.md).

The added evidence narrows the interpretation: the hybrid score improves on
product-success scoring on identical candidates, but not on fully first-order
scoring in mean selected TVD. The swarm has a small fixed-proposal mean gain
over random search; multistart Mapomatic has lower mean TVD at the tested time
budgets, and relaxation-model changes reverse the extension-panel mean gain.
No hardware or independent prospective confirmation is claimed. The new
assessment is deposited separately; this release does not assert that the
manuscript has incorporated its findings.

## Start here

Python 3.12 was used for the main experiment. From a fresh clone:

```bash
python3.12 -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements/core.txt
.venv-data/bin/python reproduce.py fetch
.venv-data/bin/python reproduce.py audit
.venv-data/bin/python reproduce.py paper
.venv-data/bin/python reproduce.py smoke
```

`fetch` downloads the versioned, SHA256-checked frozen-artifact asset.
`audit` verifies saved outputs and independently recomputes numerical summaries.
`paper` regenerates five numerical tables and seven numerical figures.
`smoke` runs twelve fresh searches on two circuits and two dates, checking
compiled circuits, scores, arbitrary-input measurement equivalence and noisy outputs.
Generated checks go into `outputs/`; archived experimental results are preserved.

For the five external compilers and complete search replay, follow
[the reproduction guide](docs/REPRODUCING.md). The full external source archives
are retrieved directly from their authors; no quantum-service credentials or
hardware credits are needed.

## Repository contents

| Location | Contents |
| --- | --- |
| `reproduce.py`, `tools/` | Portable commands, integrity checks and fresh replay adapters |
| `scripts/` | 33 preserved original Python modules in the runtime dependency closure |
| `experiments/` | Frozen protocols, circuit inputs, environment inventory and baseline registry |
| `reports/` | Recorded numerical summaries |
| `paper/data/` | Data used for the manuscript's numerical tables and figures |
| `paper/scripts/` | Figure and table exporters |
| `provenance/` | Source-copy hashes, artifact inventory, upstream revisions and download metadata |
| Release asset | Detailed jobs, candidates, output distributions, native QPY files, timing records and regional calibration extracts |

Some preserved modules have historical experiment names. The supported entry
points are documented here; those names do not expand this paper's cohort.
The release intentionally excludes credentials, virtual environments, earlier
manuscripts, downloaded literature, unrelated experiment results and live
collection schedules.

## Availability, attribution and citation

Versioned release: https://github.com/mskrcnis/response-guided-qubit-placement/releases/tag/v1.0.0

Additional validation release: https://github.com/mskrcnis/response-guided-qubit-placement/releases/tag/v1.1.0

The [data and code availability text](docs/AVAILABILITY.md) identifies the code,
data and upstream calibration source. This GitHub release has no asserted DOI.
See `CITATION.cff` for the versioned software citation.

Original code: **MIT**. Research data: see [DATA_LICENSE.md](DATA_LICENSE.md).
Upstream software and calibration data retain their own terms and attribution:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
