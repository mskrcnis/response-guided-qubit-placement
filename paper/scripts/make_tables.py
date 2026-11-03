from pathlib import Path
import json,re
R=Path(__file__).resolve().parents[1];D=R/'data';M=R/'main.tex'
def write_generated_table(name,text):
 (R/'tables').mkdir(exist_ok=True)
 (R/'tables'/(name+'.tex')).write_text(text)
def read(n):return json.loads((D/n).read_text())
strata=['known_instance_new_period','new_instance_new_period']
methods=['swarm_unique_56','mapomatic','lightsabre','sqgm','qmap','duostra']
names={'swarm_unique_56':'Unique swarm','swarm':'Original swarm','random_fixed_56':'Random search','mapomatic':'Mapomatic','lightsabre':'LightSABRE','sqgm':'SQGM','qmap':'MQT QMAP','duostra':'Duostra'}
s=read('method_summary.json');pc=read('paired_comparisons.json');time=read('serial_timing.json')['summary']
def table(name,caption,label,cols,head,rows):
 text='\\begin{table}[t]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\n\\begin{tabular}{@{}'+cols+'@{}}\\toprule\n'+head+' \\\\\\midrule\n'+'\n'.join(x+' \\\\' for x in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n';write_generated_table(name,text)
rows=[]
for m in methods:
 vals=[next(r['mean_tvd'] for r in s if r['stratum']==h and r['method']==m) for h in strata]
 gains=['--','--'] if m==methods[0] else [f"{next(r['relative_mean_error_reduction_pct'] for r in pc if r['stratum']==h and r['baseline']==m):.2f}" for h in strata]
 rows.append(names[m]+' & '+f'{vals[0]:.6f} & {vals[1]:.6f} & '+gains[0]+' & '+gains[1])
table('table_primary','Mean TVD and reduction relative to each baseline.','tab:primary','lrrrr','Method & Familiar TVD & New TVD & Familiar (\\%) & New (\\%)',rows)
rows=[]
for m in methods:
 vals=[]
 for h in strata:
  rr=next(r for r in s if r['stratum']==h and r['method']==m);tt=next(r for r in time if r['stratum']==h and r['method']==m)
  vals.extend([f"{rr['mean_cz']:.2f}",f"{rr['mean_depth']:.2f}",f"{tt['median_seconds']:.4f}"])
 rows.append(names[m]+' & '+' & '.join(vals))
table('table_cost','Native structure and cold serial compilation cost.','tab:cost','lrrrrrr',' & \\multicolumn{3}{c}{Familiar} & \\multicolumn{3}{c}{New} \\\\ \nMethod & CZ & Depth & Time (s) & CZ & Depth & Time (s)',rows)
rows=[]
for m in methods[1:]:
 vals=[]
 for h in strata:
  r=next(r for r in pc if r['stratum']==h and r['baseline']==m)['case_outcomes'];vals.append('/'.join(str(r[k]) for k in ['our_better','little_difference','our_worse']))
 rows.append(names[m]+' & '+' & '.join(vals))
table('table_cases','Paired cases favoring the swarm, near ties, and losses.','tab:cases','lcc','Baseline & Familiar: better/little/worse & New: better/little/worse',rows)
ins=read('internal_method_summary.json');rows=[]
for h,lab in zip(strata,['Familiar','New']):
 for m in ['swarm','random_fixed_56','swarm_unique_56']:
  rows0=[next(r for r in ins if r['stratum']==h and r['method']==m and r['policy']==p) for p in ['always_adopt','retain_unless_better','never_recompile']]
  pr=rows0[1]
  rows.append(lab+' & '+names[m]+' & '+' & '.join(f"{r['mean_selected_tvd']:.6f}" for r in rows0)+f" & {pr['actions']} & {pr['accepted_outcomes']['harm']}")
table('table_internal','Internal search and retention comparisons.','tab:internal','llrrrrr','Group & Search & Daily TVD & Protected & Initial & Accepted & Harmful',rows)
sens=read('sensitivity.json');rows=[]
for h,lab in zip(strata,['Familiar','New']):
 rr=[next(r for r in sens if r['stratum']==h and r['baseline']=='mapomatic' and r['scope']==scope) for scope in ['fixed','resampled','changing_calibration_only']]
 for r,lab2 in zip(rr,['Fixed seeds','Resampled seeds','Changed values only']):rows.append(f"{lab} & {lab2} & {r['cases']} & {r['mean_gain']:.6f}")
table('table_sensitivity','Sensitivity of the paired gain against Mapomatic.','tab:sensitivity','llrr','Group & Restriction & Cases & Mean TVD gain',rows)
print('Wrote five data-derived tables under paper/tables/.')
