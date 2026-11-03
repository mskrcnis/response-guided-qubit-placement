"""Matched hybrid-only experiment using preserved trajectory machinery."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import sys,datetime,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import response_full_search as original
from duplicate_aware_swarm import Owner
from run_coverage_gain import ROOT,load,dump,sha
OUT=ROOT/'experiments/duplicate_swarm_v1';REPORT=ROOT/'reports/duplicate_swarm_v1'
METHODS=['swarm','swarm_unique_56','random_fixed_56'];P=None

def prepare():
    assert not OUT.exists();old=original.verify();p={k:old[k] for k in ['bank','initial_date','dates','replicates','seed_rules','environment','score_epsilon','gain_epsilon']}
    files=dict(old['files'])
    for f in [ROOT/'scripts'/n for n in ['duplicate_aware_swarm.py','duplicate_swarm_experiment.py','validate_duplicate_swarm.py','time_duplicate_swarm.py','validate_search_loss_diagnosis.py','validate_response_search.py','validate_directional_full_search.py','validate_circuit_objective.py','time_response_search.py']]+[ROOT/'experiments/response_full_search_v1'/n for n in ['protocol.json','generation_manifest.json','validation.json']]+[ROOT/'experiments/search_loss_diagnosis_v1'/n for n in ['generation_manifest.json','validation.json']]+[ROOT/'reports/search_loss_diagnosis_v1/delivery_validation.json']:files[str(f.relative_to(ROOT))]=sha(f)
    p.update(version='duplicate_swarm_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,methods=METHODS,objectives=['hybrid'],proposal_budget=56,trajectories=90,searches=810,workers=4,
      intervention='Same8-particle6-iteration synchronous PSO. Identity and original seeded full-permutation initialization; duplicates at initialization or later replaced. Independent RNG(seed XOR0xD09A56) shuffles lexicographic injective placements once per search, pop from end until unseen. Replace position with chosen logical prefix and ascending unused tail. Reset that particle velocity after replacement; keep personal best unless strict score improvement. All scored proposals unique; replacement draws consume no PSO RNG and no extra scores. Pool preparation and rejection work included in total time.',
      controls='Rerun original swarm and random_fixed_56 with unchanged hybrid score, historical seeds and compiler. Check every control signature, trace, score, candidates and own-state outcome against saved response_full_search_v1. No coefficient cache shared across trajectory/method/replicate/rule. Exactly56 scored proposals per date; equal proposals and variant/random unique counts, not equal wall time.',
      scope='Development intervention on previously inspected5 templates, ibm_fez_r0 qubits[0,1,2,3,4,16], August15 initialization and16-23 observations,3replicates, fixed/resampled seeds. Not fresh confirmation, no general swarm-superiority claim.',
      labels=old['labels'],ordering=old['ordering'],cache=old['cache'],
      metrics='240 update-date fresh paired comparisons per method pair, gain=controlTVD-variantTVD. Better/little/worse at1e-4. Raw memberships plus deduplicated circuit/date/seed counts. Strata by template/date/rule and leave-one-template/date-out. Own protected-state help/harm/little, rejected help, selectedTVD and original update-screen gates:mean selected-fresh<=0,harm<=10%,>=30actions,>=10%coverage,>=4dates. No tuning after results, no IID tests or controller qualification.',
      validation='Independent seeded proposal reconstruction including replacement RNG and velocity reset. Recompile each distinct logical placement; verify saved native signature, ideal equivalence, hashes and56budget. Independently compute every candidate/date hybrid score using uncached density-matrix tangent propagation and every saved noisy output with explicit channels, verify own-state transitions/cache accounting. Check controls against immutable source.',
      timing='After generation and validation stop, one pinned serial run of every5template x3method x2rule replicate0 trajectory =30trajectories/270searches. Empty owner cache per trajectory, retained across9dates. Rotate method order by(template_index+rule_index) modulo3. One timing run, no selected repeats. Include owner initialization, parameter conversion, proposal generation/population construction, compilation, coefficient construction/cache operations and protected incumbent decision. Exclude input loading, artifact writes, validation checks and noisy labels. Separate cold first-date and later totals, report memory cache payload. Shared-generation times are not primary timing.',
      decision='Descriptive pilot only. Improvement requires assessing gains against BOTH controls, harm and runtime; no significance gate introduced. If no consistent useful benefit, retain random as primary quality baseline and do not force swarm into the contribution. Independent future dates/templates required before promotion.',qualified_controller=False,research_complete=False,independent_confirmation=False)
    OUT.mkdir();dump(OUT/'protocol.json',p);dump(OUT/'checkpoint.json',dict(status='protocol_frozen',completed_jobs=0,total_jobs=90,next_action='Generate90 trajectories, validate, serial timing, summarize and report.'))
def verify():
    p=load(OUT/'protocol.json')
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p
def init():
    global P
    original.OUT=OUT;original.METHODS=METHODS;original.OBJECTIVES=['hybrid'];original.Owner=Owner;original.init();P=original.P
def jobs(p):return [(c,m,'hybrid',rep,rule) for c in p['bank'] for rep in p['replicates'] for rule in p['seed_rules'] for m in METHODS]
def execute(job):return original.execute(job)
def run():
    p=verify();(OUT/'jobs').mkdir(exist_ok=True);jj=jobs(p);pending=[j for j in jj if not (OUT/'jobs'/(original.key(j)+'.json')).exists()];done=len(jj)-len(pending)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        for f in as_completed([pool.submit(execute,j) for j in pending]):
            k=f.result();done+=1;dump(OUT/'checkpoint.json',dict(status='generating',completed_jobs=done,total_jobs=90,last_job=k,next_action='Finish generation then independent validation.'));print(done,'/90',k,flush=True)
    dump(OUT/'generation_manifest.json',{str(f.relative_to(ROOT)):sha(f) for sub in ['jobs','native'] for f in sorted((OUT/sub).rglob('*')) if f.is_file()});dump(OUT/'checkpoint.json',dict(status='generation_complete_validation_pending',completed_jobs=90,total_jobs=90,next_action='Run validate_duplicate_swarm.py then time_duplicate_swarm.py.'))
if __name__=='__main__':globals()[sys.argv[1]]()
