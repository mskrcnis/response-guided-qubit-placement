"""Dependency-free check of the published mean-TVD reduction denominators."""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / 'paper/data'
summary = json.loads((data / 'method_summary.json').read_text())
pairs = json.loads((data / 'paired_comparisons.json').read_text())
lookup = {(r['stratum'], r['method']): r for r in summary}
for result in pairs:
    baseline = lookup[result['stratum'], result['baseline']]
    ours = lookup[result['stratum'], 'swarm_unique_56']
    assert baseline['cases'] == ours['cases'] == result['cases']
    gain = baseline['mean_tvd'] - ours['mean_tvd']
    assert math.isclose(gain, result['mean_gain'], abs_tol=1e-12)
    assert math.isclose(100 * gain / baseline['mean_tvd'],
                        result['relative_mean_error_reduction_pct'], abs_tol=1e-10)
    assert sum(result['case_outcomes'].values()) == result['cases']
assert len(summary) == 12 and len(pairs) == 10
assert sum(r['cases'] for r in summary) == 5940
print('Checked 12 group/method means and all 10 relative-error-reduction denominators.')
