"""Frozen factorial coverage comparison and extrapolating exposure reference."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import collections,datetime,hashlib,json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/coverage_gain_v1';LABEL=ROOT/'experiments/training_coverage_labels_v1'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def exposure(x,cols):
 d=dict(zip(cols,x));fail=lambda z:float(-np.expm1(-z));pc,po=d['reuse_proxy_current'],d['reuse_proxy_old'];sc=sum(d[f'q{i}_single_gate_loss_current']for i in range(6));so=sum(d[f'q{i}_single_gate_loss_old']for i in range(6));rc=sum(d[f'q{i}_readout_loss_current']for i in range(6));ro=sum(d[f'q{i}_readout_loss_old']for i in range(6));ce=sum(d[k]for k in cols if k.startswith('edge')and k.endswith('_cz_exposure'));read=(d['prob_meas0_prep1_mean_current']+d['prob_meas1_prep0_mean_current'])/2
 return np.array([fail(pc),fail(po),fail(pc)-fail(po),rc,ro,rc-ro,sc,so,sc-so,ce,d['logical_two_qubit_gates']*d['cz_gate_error_mean_current'],d['num_qubits']*read,d['logical_depth']*d['sx_gate_error_mean_current'],d['native_cz_amplification']*d['cz_gate_error_mean_current'],d['num_qubits'],d['logical_two_qubit_gates']],dtype=float)
def targets(G):return np.c_[G.mean(1),np.maximum(-G,0).mean(1),(G<-1e-4).mean(1),G.std(1,ddof=1)]
def sampleweights(rows):
 counts=collections.Counter((r['date'],r['circuit_id'])for r in rows);w=np.array([1/counts[(r['date'],r['circuit_id'])]for r in rows]);return w/w.mean()
def weightargs(w):return {}if np.ptp(w)<1e-15 else dict(sample_weight=w)
def leafweights(f,X,V,w):
 a=f.apply(X);b=f.apply(V);out=np.zeros((len(V),len(X)))
 for i in range(len(V)):
  for j in range(a.shape[1]):
   mask=a[:,j]==b[i,j];out[i,mask]+=w[mask]/(a.shape[1]*w[mask].sum())
 return out

def prepare():
 assert not OUT.exists();OUT.mkdir();old=load(ROOT/'experiments/distribution_models_v1/protocol.json');files=[Path(__file__),LABEL/'protocol.json',LABEL/'contexts.json',ROOT/'experiments/training_failure_diagnosis_v1/training_only.json',ROOT/'experiments/training_failure_diagnosis_v1/folds.json',ROOT/'experiments/training_failure_diagnosis_v1/protocol.json',ROOT/'experiments/rich_features_v1/feature_schema.json']
 p=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(f.relative_to(ROOT)):sha(f)for f in files},scope='Exploratory strictly past-only comparison inside original training partition; same 72 prediction contexts per compiler as prior diagnostic. No late-August or future confirmation.',coverage_levels=['sparse','dates','dates_and_states'],coverage='Sparse=original 8 sampled dates/latest retained seed11 state; dates=all16 original fit dates/latest seed11; dates_and_states=all16 dates/earliest+latest available seeds11+22. Always fit dates strictly earlier than prediction date. Equal total weight to each date/circuit group within compiler.',models=['forest','exposure_ridge','exposure_residual_forest'],forest=old['forest'],ridge=dict(alpha=1.0,fit_intercept=True,solver='svd'),reference='Standardized 16 pre-search exposure descriptors from retained negative-log success proxy, current/old gate and readout exposure and logical sizes; fit weighted ridge to empirical mean gain. Features are surrogates, not measured fidelity or guaranteed physical bounds.',residual='Fit ridge on outer-fold training contexts; form all eight outcome residuals relative to its in-fit prediction; fit original four standardized residual moments with full168-feature RF. Predict shifted weighted empirical residual distribution. In-fit residuals are part of the declared algorithm; outer prediction states/dates remain excluded.',marginal_residual='For exposure_ridge, equally weighted date/circuit groups supply global empirical residual distribution. For residual forest, condition weights on forest leaves. Clip simulated gain support to physical interval [-1,1]; do not clip to earlier target range.',inputs='Only existing pre-search retained-circuit/calibration features; no noisy simulator output, new winner, seed identity or current realized search time enters prediction.',fold_dates=['2026-08-16','2026-08-18','2026-08-21','2026-08-23'],methods=old['methods'],policy=old['policy'],metrics='Mean-gain MAE primary prediction diagnostic, downside MAE, harm-fraction MAE, outcome Brier, signed mean bias. Six unchanged decision settings per model/coverage/compiler; same observed quality, harm and support criteria. All outcomes and settings retained.',cost='Fold-fitted log search-time RF with prior measurements only. Reused/new four-worker timing sessions are heterogeneous, so this is a sensitivity proxy, not deployment or inclusive runtime validation.',selection='No hyperparameter tuning. A descriptive screen pass is only a candidate for a separately frozen own-state sequential/cost experiment. No controller is qualified or fresh validation activated by this task.')
 dump(OUT/'protocol.json',p)
def verify():
 p=load(OUT/'protocol.json')
 for f,h in p['files'].items():assert sha(ROOT/f)==h,f
 return p

def run():
 p=verify();rows=load(LABEL/'dataset.json');lookup={r['uid']:r for r in rows};sparse={r['uid']for r in load(ROOT/'experiments/training_failure_diagnosis_v1/training_only.json')};assert sparse<=set(lookup);latest={r['uid']for r in load(LABEL/'fresh_contexts.json')};folds=load(ROOT/'experiments/training_failure_diagnosis_v1/folds.json');cols=load(ROOT/'experiments/rich_features_v1/feature_schema.json')['circuit_and_local_noise'];pred=[];fits=[];(OUT/'models').mkdir(exist_ok=True)
 for fold in folds:
  test=[lookup[u]for u in fold['predict_uids']];V=np.array([r['x']for r in test]);ZV=np.array([exposure(r['x'],cols)for r in test]);actual=targets(np.array([r['gains']for r in test]))
  for coverage in p['coverage_levels']:
   train=[r for r in rows if r['method']==fold['method']and r['date']<fold['date']and(coverage=='dates_and_states'or r['uid']in(sparse if coverage=='sparse'else latest))];X=np.array([r['x']for r in train]);Z=np.array([exposure(r['x'],cols)for r in train]);G=np.array([r['gains']for r in train]);T=targets(G);w=sampleweights(train);kw=weightargs(w);key=fold['key']+'__'+coverage
   scaler=StandardScaler().fit(T,**kw);forest=RandomForestRegressor(**p['forest']).fit(X,scaler.transform(T),**kw);fw=leafweights(forest,X,V,w)
   zs=StandardScaler().fit(Z,**kw);ridge=Ridge(**p['ridge']).fit(zs.transform(Z),T[:,0],**kw);reftrain=ridge.predict(zs.transform(Z));ref=ridge.predict(zs.transform(ZV));res=G-reftrain[:,None];RT=targets(res);rs=StandardScaler().fit(RT,**kw);resforest=RandomForestRegressor(**p['forest']).fit(X,rs.transform(RT),**kw);rw=leafweights(resforest,X,V,w)
   cost=RandomForestRegressor(**p['forest']).fit(X,np.log([np.median(r['search_seconds'])for r in train]),**kw);cp=np.exp(cost.predict(V));pack=dict(forest=forest,target_scaler=scaler,ridge=ridge,exposure_scaler=zs,residual_forest=resforest,residual_scaler=rs,cost=cost,fit_uids=[r['uid']for r in train],sample_weights=w);joblib.dump(pack,OUT/'models'/(key+'.joblib'));fits.append(dict(key=key,method=fold['method'],date=fold['date'],coverage=coverage,fit_contexts=len(train),fit_dates=sorted({r['date']for r in train}),fit_uids=pack['fit_uids'],predict_uids=fold['predict_uids']))
   np.savez_compressed(OUT/'models'/(key+'_support.npz'),forest_weights=fw,residual_weights=rw,train_reference=reftrain,prediction_reference=ref)
   for i,r in enumerate(test):
    choices=[('forest',fw[i],G),('exposure_ridge',w/w.sum(),res+ref[i]),('exposure_residual_forest',rw[i],res+ref[i])]
    for model,cw,support in choices:
     clipped=np.clip(support,-1,1);mu=float(cw@clipped.mean(1));down=float(cw@np.maximum(-clipped,0).mean(1));harm=float(cw@(clipped<-1e-4).mean(1));pred.append(dict(uid=r['uid'],method=r['method'],date=r['date'],coverage=coverage,model=model,mean=mu,downside=down,harm=harm,predicted_search_seconds=float(cp[i]),observed_mean=float(actual[i,0]),observed_downside=float(actual[i,1]),observed_harm=float(actual[i,2]),reference_mean=float(ref[i]),prior_mean_min=float(T[:,0].min()),prior_mean_max=float(T[:,0].max()),support_values_clipped=int(np.sum(clipped!=support))))
   print('Fitted',key,len(train),'contexts',flush=True)
 dump(OUT/'fits.json',fits);dump(OUT/'predictions.json',pred);dump(OUT/'data_sources.json',{str(f.relative_to(ROOT)):sha(f)for f in [LABEL/'dataset.json',LABEL/'artifact_manifest.json']});summarize()
def summarize():
 p=verify();pr=load(OUT/'predictions.json');rows={r['uid']:r for r in load(LABEL/'dataset.json')};comparisons=[];policies=[];bydate=[]
 for method in p['methods']:
  for coverage in p['coverage_levels']:
   for model in p['models']:
    rr=[r for r in pr if r['method']==method and r['coverage']==coverage and r['model']==model];assert len(rr)==72;G=np.array([rows[r['uid']]['gains']for r in rr]);gm=G.mean(1);mu=np.array([r['mean']for r in rr]);down=np.array([r['downside']for r in rr]);hp=np.array([r['harm']for r in rr]);metrics=dict(mean_mae=float(np.mean(abs(mu-gm))),mean_bias=float(np.mean(mu-gm)),downside_mae=float(np.mean(abs(down-np.maximum(-G,0).mean(1)))),harm_mae=float(np.mean(abs(hp-(G<-1e-4).mean(1)))),outcome_brier=float(np.mean((hp[:,None]-(G<-1e-4))**2)),predictions_outside_prior_mean_range=sum(r['mean']<r['prior_mean_min']-1e-12 or r['mean']>r['prior_mean_max']+1e-12 for r in rr),support_values_clipped=sum(r['support_values_clipped']for r in rr));comparisons.append(dict(method=method,coverage=coverage,model=model,contexts=72,**metrics))
    for day in p['fold_dates']:
     indices=[i for i,r in enumerate(rr)if r['date']==day];bydate.append(dict(method=method,coverage=coverage,model=model,date=day,contexts=18,mean_mae=float(np.mean(abs(mu[indices]-gm[indices])))))
    for beta in [0.,1.]:
     for lam in [0.,.001,.01]:
      a=mu-beta*down>lam*np.array([r['predicted_search_seconds']for r in rr]);n=int(a.sum());h=int((G[a]<-1e-4).sum());risk=h/(n*8)if n else None;delta=float(G[~a].sum()/G.size);dates=len({r['date']for r,b in zip(rr,a)if b});screen=p['policy']['screen'];passed=bool(n>=screen['min_actions']and n/72>=screen['min_coverage']and dates>=screen['min_action_dates']and risk is not None and risk<=screen['max_observed_harm_fraction']and delta<=screen['max_mean_delta_vs_always']);policies.append(dict(method=method,coverage=coverage,model=model,beta=beta,cost_weight=lam,actions=n,contexts=72,action_dates=dates,harmful_action_outcomes=h,action_outcomes=n*8,harm_fraction=risk,mean_delta_vs_always=delta,mean_delta_vs_never=float(-G[a].sum()/G.size),missed_beneficial_contexts=int(np.sum((~a)&(gm>1e-4))),selected_harmful_mean_contexts=int(np.sum(a&(gm<-1e-4))),passes_descriptive_screen=passed,qualified_controller=False))
 dump(OUT/'comparison.json',comparisons);dump(OUT/'policy_screen.json',policies);dump(OUT/'by_date.json',bydate);dump(OUT/'summary.json',dict(training_contexts_available=2160,unique_prediction_contexts=144,prediction_rows=len(pr),fold_coverage_fits=24,models_per_bundle='3 forests plus 1 ridge; 96 independent fits',total_policy_settings=len(policies),passing_descriptive_settings=sum(r['passes_descriptive_screen']for r in policies),best_by_mean_mae={m:min([r for r in comparisons if r['method']==m],key=lambda r:r['mean_mae'])for m in p['methods']},qualified_controller=False,later_partition_used=False,inclusive_runtime_validated=False))
 dump(OUT/'artifact_manifest.json',{str(f.relative_to(ROOT)):sha(f)for f in sorted(OUT.rglob('*'))if f.is_file()and f.name not in ['artifact_manifest.json','validation.json','checkpoint.json']});dump(OUT/'checkpoint.json',dict(status='generated_validation_pending'))
if __name__=='__main__':prepare()if sys.argv[-1]=='prepare'else run()
