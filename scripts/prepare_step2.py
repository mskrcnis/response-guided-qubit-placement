"""Build conservative event filters and daily, past-only calibration scenarios.

No imputation, symmetric-edge synthesis, or external source execution.
"""
from pathlib import Path
import collections
import datetime as dt
import hashlib
import json
import os
import shutil
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/processed/step2_v1'
SOURCE = ROOT/'data/external/step1/calibration_drift/source/data/train-00000-of-00001.parquet'
UTC = dt.timezone.utc
PROPERTIES = ['sx_gate_error','x_gate_error','rz_gate_error','cz_gate_error',
              'prob_meas0_prep1','prob_meas1_prep0']

def dump(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False)+'\n')
    temp.replace(path)

def split(t):
    if t < dt.datetime(2026,8,1,tzinfo=UTC): return 'train'
    if t < dt.datetime(2026,9,1,tzinfo=UTC): return 'validation'
    return 'test'

def key(r):
    return (r['property'], r['qubit_a'], r['qubit_b'])

def usable(r, time):
    return (r['quality_reason']=='accepted'
            and 0 <= (time-r['calibrated_time']).total_seconds() <= 72*3600
            and 0 <= (time-r['observed_time']).total_seconds() <= 72*3600)

def main():
    if OUT.exists():
        raise SystemExit(f'{OUT} exists; preserve the frozen preparation. Use a new version for changes.')
    stage=OUT.with_name(OUT.name+'.building')
    if stage.exists():
        raise SystemExit(f'Unfinished staging folder exists: {stage}; inspect before recovery.')
    stage.mkdir(parents=True)
    c=duckdb.connect()
    c.execute("SET TimeZone='UTC'")
    c.execute("SET threads=2")
    c.read_parquet(str(SOURCE)).create_view('raw')
    prop_sql=','.join(repr(p) for p in PROPERTIES)
    # Keep rejected latest events available to reconstruction: no fallback to
    # an older healthy value after a later invalid/failure event.
    c.execute(f"""create table events as select backend,property,qubit_a,
      cast(qubit_b as integer) qubit_b,value,unit,scope,is_failure_ceiling,
      observed_time,calibrated_time,snapshot_update_time,is_new_measurement,
      case when unit is null or unit!='' then 'missing_or_unexpected_unit'
       when observed_time is null or calibrated_time is null then 'missing_timestamp'
       when calibrated_time>observed_time then 'future_calibration'
       when snapshot_update_time>observed_time then 'future_snapshot'
       when not isfinite(value) or value is null then 'nonfinite_value'
       when value<0 or value>1 then 'invalid_probability'
       when is_failure_ceiling then 'failure_ceiling'
       when value>0.5 then 'pilot_high_error_exclusion'
       when qubit_a is null or qubit_a<0 or qubit_a!=floor(qubit_a)
        or (qubit_b is not null and (qubit_b<0 or qubit_b!=floor(qubit_b))) then 'invalid_qubit'
       when property='cz_gate_error' and (qubit_b is null or qubit_a=qubit_b)
        then 'invalid_pair'
       when property!='cz_gate_error' and qubit_b is not null then 'unexpected_pair'
       else 'accepted' end quality_reason
      from raw where property in ({prop_sql})""")
    for filename,where in [('calibration_events',''),('clean_events',"where quality_reason='accepted'"),
                           ('quarantined_events',"where quality_reason!='accepted'")]:
        c.execute(f"COPY (select * from events {where}) TO '{stage/filename}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")
    def rows(sql):
        cur=c.execute(sql); names=[d[0] for d in cur.description]
        return [dict(zip(names,row)) for row in cur.fetchall()]
    quality=rows('select backend,quality_reason,count(*) row_count from events group by all order by backend,quality_reason')
    anchor=dt.datetime(2026,6,1,tzinfo=UTC)
    anchor_rows=rows("""select * from events where backend='ibm_fez'
      and observed_time<=TIMESTAMPTZ '2026-06-01 00:00:00+00'
      qualify row_number() over(partition by property,qubit_a,qubit_b
        order by observed_time desc,calibrated_time desc)=1""")
    state={key(r):r for r in anchor_rows}
    single=[p for p in PROPERTIES if p!='cz_gate_error']
    nodes=sorted({r['qubit_a'] for r in anchor_rows if all(
        (p,r['qubit_a'],None) in state and usable(state[(p,r['qubit_a'],None)],anchor) for p in single)})
    graph={n:set() for n in nodes}
    for a in nodes:
        for b in nodes:
            if a!=b and all(('cz_gate_error',x,y) in state and usable(state[('cz_gate_error',x,y)],anchor)
                           for x,y in [(a,b),(b,a)]): graph[a].add(b)
    selected=[]
    for start in nodes:
        queue=[start]; visited=[]
        while queue and len(visited)<6:
            n=queue.pop(0)
            if n in visited: continue
            visited.append(n)
            queue.extend(x for x in sorted(graph[n]) if x not in visited)
        if len(visited)==6:
            selected=sorted(visited); break
    assert len(selected)==6, 'No six-node valid connected anchor patch; do not silently relax policy.'
    edges=sorted((a,b) for a in selected for b in graph[a] if b in selected)
    required=sorted([(p,n,None) for p in single for n in selected]+[('cz_gate_error',a,b) for a,b in edges],key=str)
    all_rows=rows(f"""select * from events where backend='ibm_fez'
      and qubit_a in ({','.join(map(str,selected))})
      order by observed_time,calibrated_time,property,qubit_a,qubit_b""")
    daily=[]; ptr=0; latest={}
    for day in range((dt.datetime(2026,10,11,tzinfo=UTC)-anchor).days):
        cutoff=anchor+dt.timedelta(days=day)
        while ptr<len(all_rows) and all_rows[ptr]['observed_time']<=cutoff:
            r=all_rows[ptr]; latest[key(r)]=r;ptr+=1
        reasons=[]; evidence=[]
        for k in required:
            r=latest.get(k)
            if r is None: reasons.append({'key':k,'reason':'never_observed'})
            elif not usable(r,cutoff):
                reasons.append({'key':k,'reason':r['quality_reason'] if r['quality_reason']!='accepted' else 'stale_over_72_hours'})
            if r is not None: evidence.append(r)
        daily.append({'scenario_id':'fez_'+cutoff.strftime('%Y%m%d'),
          'backend':'ibm_fez','cutoff_utc':cutoff,'temporal_split':split(cutoff),
          'status':'ready' if not reasons else 'blocked','blocking_reasons':reasons,
          'parameters':evidence})
    config={'version':'step2_v1','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'source_revision':'d7f844495732bd25f721792e56248a8cc338e9dd',
      'properties':PROPERTIES,'backend':'ibm_fez','physical_qubits':selected,
      'directed_cz_edges':edges,'max_measurement_and_observation_age_hours':72,
      'anchor_utc':anchor,'patch_selection':'First six-node BFS by ascending qubit ID with complete bidirectional anchor calibration; no outcome-based selection',
      'chronological_splits':{'train':'2026-06-01 through 2026-07-31','validation':'2026-08-01 through 2026-08-31','test':'2026-09-01 through 2026-10-10'},
      'pairing':'Consecutive calendar days within the same split; both must be ready; no gap bridging',
      'circuit_family_splits':{'train':['ghz','product'],'validation':['qft'],'test':['qaoa']},
      'strict_training_rule':'Use only train-time AND train-family rows to train. Validation/test Cartesian strata remain separate.',
      'pilot_rule':'First three eligible train-time day-pairs; train families only; seeds 11,22,33. No validation/test outcomes opened.',
      'noise_model':'Independent gate depolarization using average gate infidelity r and lambda=d*r/(d-1), plus asymmetric assignment error; excludes relaxation, crosstalk, leakage and idle noise',
      'basis_gates':['rz','sx','x','cz'],'routing_method':'basic','optimization_level':1,
      'swarm':{'particles':8,'iterations':6,'inertia':0.5,'cognitive':0.5,'social':0.7,'mutation_probability':0.2},
      'feature_policy':'Only circuit descriptors, previous compiled circuit descriptors, and past/current calibration changes; no fresh compilation score/cost/outcome as inference input',
      'label_policy':'delta_tvd = reuse_TVD - fresh_TVD; positive means fresh improves. No cost-aware binary label until an explicit cost tradeoff is set.'}
    dump(stage/'config.json',config);dump(stage/'daily_scenarios.json',daily)
    pairs=[]
    for prev,now in zip(daily,daily[1:]):
        if prev['status']==now['status']=='ready' and prev['temporal_split']==now['temporal_split']:
            pairs.append({'pair_id':prev['scenario_id']+'__'+now['scenario_id'],
              'old_scenario':prev['scenario_id'],'current_scenario':now['scenario_id'],'temporal_split':now['temporal_split']})
    dump(stage/'pair_manifest.json',pairs)
    summary={'quality_counts':quality,'physical_qubits':selected,'directed_edges':edges,
      'daily_status_counts':dict(collections.Counter(x['status'] for x in daily)),
      'pair_counts':dict(collections.Counter(x['temporal_split'] for x in pairs)),
      'blocking_reason_counts':dict(collections.Counter(z['reason'] for x in daily for z in x['blocking_reasons'])),
      'event_rows':sum(x['row_count'] for x in quality),'source_untouched':True}
    dump(stage/'preparation_summary.json',summary)
    assert pairs and all(x['cutoff_utc']>=anchor for x in daily)
    stage.replace(OUT)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
