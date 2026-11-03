# Reproducing the paper artifact

## Environments

The recorded operating system is Linux. The main experiment used Python
3.12.3, Qiskit 2.5.2, Aer 0.17.2 and NumPy 2.5.4. SQGM uses a separate Python
3.10.19 / Qiskit Terra 0.22.3 environment. Exact observed package inventories
are in `experiments/published_baselines_v1/environment.json`; minimal direct
dependency pins are in `requirements/`. These are observed versions, not a
claim that other versions produce identical compiler outputs.

The parallel-search process sets native numerical libraries to one thread.
The small-circuit density-matrix calculation remains exponential in width.
The frozen artifact is approximately 42 MB compressed and 240 MB extracted;
dependency environments and third-party builds require additional disk space.

## 1. Verify and audit the archived experiment

Use the README installation commands, then:

```bash
.venv-data/bin/python reproduce.py fetch
.venv-data/bin/python reproduce.py verify
.venv-data/bin/python reproduce.py audit
```

An offline copy can be installed with `reproduce.py fetch --archive /path/to/qst-frozen-artifacts-v1.0.0.tar.gz`.
Archive extraction checks the allowlisted inventory and every file hash and
refuses to overwrite a changed file. No shell scripts from the archive run.

The external audit checks complete observation keys, all saved probability
distributions, TVD arithmetic, summary means, relative-reduction denominators,
case counts, per-date/per-circuit summaries, sensitivity analyses and timing
records. It independently enumerates all 94,248 feasible Mapomatic embedding
costs for the 945 Mapomatic requests.

The internal audit checks all 270 trajectories, 3,240 searches, proposal
budgets, native-file signatures, saved independent-channel comparisons and
post-search policy histories. It recomputes fifteen detailed output tables
and compares them with the frozen results.

These saved-output checks are distinct from rerunning the numerical solvers.
`smoke` and the replay commands below execute the solvers again.

## 2. Regenerate manuscript outputs

```bash
.venv-data/bin/python reproduce.py paper
```

This writes seven numerical plots under `paper/figures/` and five generated
LaTeX tables under `paper/tables/`. Figure 1 is the supplied architecture
illustration and is preserved rather than inferred from results. The five
generated tables correspond to manuscript Tables 5–9; descriptive Tables
1–4 and the circuit-inventory table are supported by the protocols and registry.
`paper/data/` contains the exact frozen inputs used for the original manuscript.

## 3. Rerun the internal searches

```bash
.venv-data/bin/python reproduce.py smoke
.venv-data/bin/python reproduce.py full-search
```

The smoke check uses GHZ-4 and the new five-qubit ring instance, all three
search methods, replicate zero with resampled seeds, and the first and last
paired dates: twelve fresh searches. The full command reruns all 3,240
searches, including initialization, and checks every selected executable,
response score and noisy-output TVD against its frozen counterpart. It can
take substantially longer. Neither command overwrites original jobs.

Search outcomes can be checked numerically under the pinned environment;
wall-clock times are machine-dependent and are not expected to match. The
recorded 270 timing trials remain historical observations, separately audited.

## 4. Rerun the five external compiler configurations

```bash
python3.12 -m venv .venv-baselines-modern
.venv-baselines-modern/bin/python -m pip install -r requirements/modern.txt
python3.10 -m venv .venv-baselines-legacy
.venv-baselines-legacy/bin/python -m pip install -r requirements/legacy.txt
python3 tools/fetch_baseline_sources.py --build-qsyn
.venv-data/bin/python tools/replay_baselines.py
```

The Qsyn build additionally needs Git, CMake and a C++ compiler supporting
the pinned upstream project. Its dependencies are fetched by upstream CMake.
The included patch changes build compatibility and exposes initial/final
permutations; it does not modify the mapping algorithm. SQGM is fetched from
its original repository because no top-level redistribution license was
found in the recorded checkout.

The default replay compiles two circuits with all five external methods and
checks topology/native gates, arbitrary-input measurement equivalence,
native signatures, and independently simulated noisy outputs. Custom interpreter
paths can be supplied with `--modern-python` and `--legacy-python`.

```bash
.venv-data/bin/python tools/replay_baselines.py --all
```

The full option recompiles all 2,055 distinct requests and evaluates the
associated dates. Maintained implementations are compared under the recorded
settings; this does not reproduce the original publications' benchmark tables.

## Portability and provenance boundaries

The original numerical modules are retained byte-for-byte. Portable adapters
resolve historical absolute artifact paths within the new checkout and
redirect new audit output. They do not alter objectives, search budgets,
baseline settings, dates, circuit populations or saved results.

The old protocols pinned the entire development workspace, including unrelated
scripts, earlier pilots, external clones and a machine-specific Qsyn binary.
The public release uses an explicit manifest for its own contents. Available
original pins are checked, and omitted original pins are listed in the audit's
`source_coverage.json`. They are not silently counted as verified. The earlier
October 21 pilot-overlap regression is not rerun or counted as new evidence;
the paper-cohort checks remain active.

The original whole-device calibration archives total several gigabytes and
are not duplicated. The selected parameter vectors, all frozen regional
extracts, retrieval receipts, upstream metadata and immutable revision URLs
are supplied. Full original parquet files can be obtained using the URLs in
`provenance/upstream_calibration_files.json` and checked against its SHA256
values. A failure of an upstream URL must be reported, not replaced silently
by the current dataset revision.

No IBM account, credentials, hardware jobs or ongoing collection service are
required for these commands. The archived evidence does not establish
physical-device benefit, independent confirmation or a pre-search ML policy.
