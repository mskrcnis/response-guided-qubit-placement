"""Past-only receipt-gated scenarios; no noisy evaluation or automatic promotion."""
import datetime,json
from pathlib import Path
import duckdb
from run_coverage_gain import ROOT,load,dump,sha
OUT=ROOT/'experiments/swarm_confirmation_v1'
UTC=datetime.timezone.utc
def parse(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).astimezone(UTC)
def assess(p,receipts,cutoff):
    required=[(name,q,None) for q in p['physical_qubits'] for name in p['properties'] if name!='cz_gate_error']+[('cz_gate_error',a,b) for a,b in p['directed_cz_edges']]
    db=duckdb.connect();db.execute("SET TimeZone='UTC'");db.execute('SET threads=1');visible=[];seen=set()
    for r in sorted(receipts,key=lambda r:r['received_at_utc']):
        if parse(r['received_at_utc'])>cutoff:continue
        a=r['regional'];identity=a['sha256']
        if identity in seen:continue
        seen.add(identity);f=ROOT/a['path'];assert sha(f)==identity;name='p'+str(len(visible));db.read_parquet(str(f)).create_view(name);visible.append(f"select *, TIMESTAMPTZ '{parse(r['received_at_utc']).isoformat()}' as receipt_at, '{identity}' as receipt_hash from {name}")
    if not visible:return dict(status='blocked',blocking_reasons=[dict(reason='no_receipt_available_at_cutoff')],parameters=[],visible_revisions=0)
    db.execute('create view all_rows as '+' union all '.join(visible));freeze=parse(p['frozen_at_utc'])
    # Conflicting values for the same calibrated event are quarantined once visible.
    cur=db.execute('''with eligible as (select * from all_rows where observed_time<=? or observed_time is null), conflicts as (select backend,property,qubit_a,qubit_b,calibrated_time,count(distinct struct_pack(v:=value,u:=unit,f:=is_failure_ceiling))>1 has_conflict from eligible group by all), joined as(select e.*,c.has_conflict from eligible e join conflicts c on e.backend=c.backend and e.property=c.property and e.qubit_a=c.qubit_a and e.qubit_b is not distinct from c.qubit_b and e.calibrated_time is not distinct from c.calibrated_time) select *, min(receipt_at) over(partition by backend,property,qubit_a,qubit_b,observed_time,calibrated_time,value,unit,snapshot_update_time,is_failure_ceiling) as first_seen_at, arg_min(receipt_hash,receipt_at) over(partition by backend,property,qubit_a,qubit_b,observed_time,calibrated_time,value,unit,snapshot_update_time,is_failure_ceiling) as first_seen_hash from joined qualify row_number() over(partition by backend,property,qubit_a,qubit_b order by coalesce(observed_time,receipt_at) desc,calibrated_time desc nulls last,receipt_at desc)=1''',[cutoff]);rows=[dict(zip([x[0] for x in cur.description],r)) for r in cur.fetchall()]
    import math
    def missing(x):return x is None or isinstance(x,float) and math.isnan(x)
    state={};invalid_extra=[]
    for r in rows:
        a=r['qubit_a'];b=None if missing(r['qubit_b']) else r['qubit_b']
        if missing(a) or a!=int(a) or (b is not None and b!=int(b)):invalid_extra.append(dict(reason='invalid_qubit'));continue
        state[r['property'],int(a),None if b is None else int(b)]=r
    blocked=invalid_extra;params=[]
    for key in required:
        r=state.get(key);reasons=[]
        if r is None:blocked.append(dict(key=list(key),reasons=['missing_parameter']));continue
        value=r['value'];obs=r['observed_time'];cal=r['calibrated_time'];update=r['snapshot_update_time'];receipt=r['receipt_at']
        if r['unit']!='':reasons.append('invalid_unit')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=.5:reasons.append('invalid_probability_or_high_error')
        if r['is_failure_ceiling'] is True:reasons.append('failure_ceiling')
        if r['has_conflict']:reasons.append('conflicting_revision')
        if missing(obs) or missing(cal):reasons.append('missing_timestamp')
        else:
            if cal>obs or obs>receipt or (not missing(update) and update>obs):reasons.append('future_timestamp')
            if cal<freeze:reasons.append('pre_freeze_measurement')
            if not 0<=(cutoff-cal).total_seconds()<=72*3600 or not 0<=(cutoff-obs).total_seconds()<=72*3600:reasons.append('stale_over72h')
        if reasons:blocked.append(dict(key=list(key),reasons=reasons))
        else:params.append(dict(backend=r['backend'],property=key[0],qubit_a=key[1],qubit_b=key[2],value=float(value),unit='',observed_time=obs.isoformat(),calibrated_time=cal.isoformat(),snapshot_update_time=None if missing(update) else update.isoformat(),first_received_at_utc=r['first_seen_at'].isoformat(),receipt_hash=r['first_seen_hash']))
    return dict(status='blocked' if blocked else 'ready',blocking_reasons=blocked,parameters=params if not blocked else [],visible_revisions=len(visible))
def main():
    p=load(OUT/'protocol.json')
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    now=datetime.datetime.now(UTC);receipts=[]
    for f in sorted((OUT/'collection/receipts').glob('*.json')):
        r=load(f);assert r['protocol_sha256'] in [sha(OUT/'protocol.json')]+p.get('compatible_receipt_protocols',[]);receipts.append(r)
    rows=[];previous_ready=False
    for day in [p['initial_date']]+p['dates']:
        cutoff=parse(day+'T00:00:00+00:00');r=assess(p,receipts,cutoff) if cutoff<=now else dict(status='future',parameters=[],blocking_reasons=[dict(reason='cutoff_not_reached')]);is_ready=r['status']=='ready';rows.append(dict(date=day,cutoff_utc=cutoff.isoformat(),episode_initialization=is_ready and not previous_ready,paired_observation=is_ready and previous_ready and day in p['dates'],**r));previous_ready=is_ready
    # Freeze strictly after old source max measurement -> disjoint event identities.
    audit=load(OUT/'exposure_audit.json');assert parse(p['frozen_at_utc'])>parse(audit['old_source']['max_calibrated'])
    ready=sum(r['paired_observation'] for r in rows);closed=now>=parse(p['dates'][-1]+'T00:00:00+00:00');result=dict(status='ready_for_confirmation' if closed and ready>=28 else 'inconclusive_source_coverage' if closed else 'awaiting_future_calibration',as_of_utc=now.isoformat(),calendar_closed=closed,ready_observation_dates=ready,required_ready_dates=28,source_receipts=len(receipts),scenarios=rows,protocol_sha256=sha(OUT/'protocol.json'),noisy_outcomes_generated=False,qualified_controller=False)
    dump(OUT/'readiness.json',result);print(json.dumps({k:v for k,v in result.items() if k!='scenarios'}))
if __name__=='__main__':main()
