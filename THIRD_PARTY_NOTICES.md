# Third-party materials

No third-party compiler source tree or executable is redistributed here.
`tools/fetch_baseline_sources.py` retrieves upstream repositories at the
recorded commits. Their own terms continue to apply.

| Component | Source | Recorded implementation |
| --- | --- | --- |
| LightSABRE / Qiskit | https://github.com/Qiskit/qiskit | Qiskit 2.5.2 |
| Mapomatic | https://github.com/qiskit-community/mapomatic | `c28d796bc181b01cc88b845814c814265b23bfe4` |
| SQGM | https://github.com/ebony72/sqgm | `55b0eabb8f3e331435f652f091067ca196aea3b0` |
| MQT QMAP | https://github.com/munich-quantum-toolkit/qmap | 3.11.0 |
| Qsyn / Duostra | https://github.com/DVLab-NTU/qsyn | `b77187c87f0bddd69bc6c213700642ee6c423159` |

The SQGM checkout did not include a top-level license file at the recorded
commit; no license for it is asserted and its source is not repackaged.
The small Qsyn patch is retained as an explicit study adapter. It adjusts
build compatibility and logs permutations; it does not change the routing
algorithm. Software installed by the requirements files retains its own
license. Publication identifiers and implementation caveats are in
`experiments/published_baselines_v1/baseline_registry.json`.

Calibration attribution and the distinction between upstream data and our
derived outputs are documented in `DATA_LICENSE.md`.
