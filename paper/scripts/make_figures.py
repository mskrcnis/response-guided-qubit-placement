"""Regenerate seven numerical plots; preserve the supplied architecture figure."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';F=ROOT/'figures'
for ext in ('pdf','png'):
 if not (F/('workflow.'+ext)).is_file():
  raise FileNotFoundError('Restore the supplied workflow.'+ext+' before regenerating plots.')
def read(n):return json.loads((D/n).read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
navy='#17324D';teal='#007F82';gold='#CE8A26';red='#AF4050';grey='#667581'
colors=[teal,navy,'#5286B3',gold,'#89619D',red]
methods=['swarm_unique_56','mapomatic','lightsabre','sqgm','qmap','duostra']
names={'swarm_unique_56':'Unique swarm','swarm':'Original swarm','random_fixed_56':'Random search','mapomatic':'Mapomatic','lightsabre':'LightSABRE','sqgm':'SQGM','qmap':'MQT QMAP','duostra':'Duostra'}
strata=['known_instance_new_period','new_instance_new_period']; snames=['Familiar instances','New instances']
def save(fig,n):
 fig.savefig(F/(n+'.pdf'),bbox_inches='tight');fig.savefig(F/(n+'.png'),dpi=190,bbox_inches='tight');plt.close(fig)
def label(ax,s):ax.text(-.12,1.04,s,transform=ax.transAxes,fontweight='bold',color=navy)

sc=read('scenarios.json');fig,axes=plt.subplots(1,2,figsize=(6.4,3.05),gridspec_kw={'width_ratios':[1,2.1]})
ax=axes[0];pos={0:(0,0),1:(1,0),2:(2,0),3:(3,0),4:(4,.8),16:(4,-.8)}
for a,b in [(0,1),(1,2),(2,3),(3,4),(3,16)]:ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color=grey,lw=2,zorder=1)
for v,(x,y) in pos.items():ax.scatter(x,y,s=420,color=navy,zorder=2);ax.text(x,y,str(v),ha='center',va='center',color='white',fontsize=10)
ax.set(xlim=(-.6,4.6),ylim=(-1.6,1.6),aspect='equal');ax.axis('off');ax.text(2,-1.4,'Physical qubit IDs',ha='center',fontsize=8);label(ax,'(a)')
ax=axes[1];dates=[s['date'][8:] for s in sc];series={}
for kind,lab in [('sx_gate_error','SX gate error'),('cz_gate_error','CZ gate error'),('prob_meas1_prep0','Readout 0→1'),('prob_meas0_prep1','Readout 1→0')]:series[lab]=[np.mean([p['value'] for p in s['parameters'] if p['property']==kind]) for s in sc]
for (lab,ys),col,mark in zip(series.items(),[navy,teal,gold,red],['o','s','^','D']):ax.plot(range(12),ys,marker=mark,ms=3.5,lw=1.2,color=col,label=lab)
for i in [5,11]:ax.axvspan(i-.27,i+.27,color='#DEE4E8',alpha=.7,zorder=0)
ax.set(xticks=range(12),xticklabels=dates,xlabel='October 2026 cutoff (UTC)',ylabel='Mean recorded probability',ylim=(0,None));ax.tick_params(axis='x',labelsize=7);ax.legend(frameon=False,ncol=2,fontsize=6.8,loc='upper left',bbox_to_anchor=(0,1.22));ax.grid(axis='y',alpha=.2);label(ax,'(b)');fig.tight_layout(w_pad=2);save(fig,'calibration_patch')

summary=read('method_summary.json');fig,axes=plt.subplots(1,2,figsize=(6.4,3.0),sharex=True,sharey=True)
for i,(ax,s) in enumerate(zip(axes,strata)):
 vals=[next(r['mean_tvd'] for r in summary if r['stratum']==s and r['method']==m) for m in methods]
 ax.barh(np.arange(6),vals,color=colors,height=.65);ax.set(yticks=np.arange(6),yticklabels=[names[m] for m in methods],xlabel='Mean simulated TVD',xlim=(0,.17));ax.set_ylim(5.6,-.6);ax.grid(axis='x',alpha=.18)
 for j,v in enumerate(vals):ax.text(v+.002,j,f'{v:.4f}',va='center',fontsize=7)
 ax.text(.5,1.04,snames[i],ha='center',va='bottom',transform=ax.transAxes,fontsize=8);label(ax,'(a)' if i==0 else '(b)')
fig.tight_layout(w_pad=1.2);save(fig,'quality')

bc=read('by_circuit.json');p=read('protocol.json');circuits=list(p['bank']);short=['GHZ 4','Product 4','Line 4','Ring 4','Star 5','Rotated GHZ 4','Rotated GHZ 5','Product 4*','Product 5*','Line 4*','Line 5*','Ring 4*','Ring 5*','Star 4*','Star 5*']
mat=np.array([[next(r['mean_gain'] for r in bc if r['circuit_id']==c and r['baseline']==b) for b in methods[1:]] for c in circuits])
fig,ax=plt.subplots(figsize=(6.4,4.65));im=ax.imshow(mat,cmap='RdBu',norm=TwoSlopeNorm(vmin=min(mat.min(),-.005),vcenter=0,vmax=mat.max()),aspect='auto')
ax.set(yticks=range(15),yticklabels=short,xticks=range(5),xticklabels=[names[m] for m in methods[1:]]);ax.axhline(4.5,lw=2,color='white')
for y in range(15):
 for x in range(5):ax.text(x,y,f'{mat[y,x]:+.3f}',ha='center',va='center',fontsize=7,color='white' if mat[y,x]>.060 or mat[y,x]<-.015 else '#202C39')
cb=fig.colorbar(im,ax=ax,shrink=.8,pad=.025);cb.set_label('Baseline TVD − unique-swarm TVD');fig.tight_layout();save(fig,'circuit_gains')

bd=read('by_date.json');fig,axes=plt.subplots(2,1,figsize=(6.4,4.1),sharex=True)
for i,(ax,s) in enumerate(zip(axes,strata)):
 for b,col,mark in zip(methods[1:],colors[1:],['o','s','^','D','v']):
  rows=[r for r in bd if r['stratum']==s and r['baseline']==b];ax.plot(range(11),[r['mean_gain'] for r in rows],label=names[b],color=col,marker=mark,ms=3,lw=1.1)
 ax.axhline(0,color='#333333',lw=.7);ax.set(ylabel='Mean TVD difference');ax.text(.99,.93,snames[i],transform=ax.transAxes,ha='right',va='top',fontsize=8);ax.grid(axis='y',alpha=.18);label(ax,'(a)' if i==0 else '(b)')
axes[0].legend(ncol=3,frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(0,1.35));axes[1].set(xticks=range(11),xticklabels=[s['date'][8:] for s in sc[1:]],xlabel='October 2026 cutoff (UTC)');fig.tight_layout(h_pad=1.3);save(fig,'daily_gains')

internal=read('internal_method_summary.json');work=read('internal_computational_work.json');fig,axes=plt.subplots(1,2,figsize=(6.4,3.15))
imethods=['swarm','random_fixed_56','swarm_unique_56'];cols=[grey,gold,teal];x=np.arange(2);w=.24
for j,(m,col) in enumerate(zip(imethods,cols)):
 vals=[next(r['mean_selected_tvd'] for r in internal if r['stratum']==s and r['method']==m and r['policy']=='always_adopt') for s in strata];axes[0].bar(x+(j-1)*w,vals,width=w,color=col,label=names[m])
 counts=[np.mean([r['mean_unique_layouts'] for r in work if r['stratum']==s and r['method']==m]) for s in strata];axes[1].bar(x+(j-1)*w,counts,width=w,color=col)
axes[0].set(ylabel='Mean simulated TVD',ylim=(0,.09));axes[1].set(ylabel='Distinct placements per search',ylim=(0,62));axes[1].axhline(56,color=navy,ls='--',lw=.8)
for i,ax in enumerate(axes):ax.set(xticks=x,xticklabels=['Familiar','New']);ax.grid(axis='y',alpha=.18);label(ax,'(a)' if i==0 else '(b)')
axes[0].legend(frameon=False,fontsize=7,ncol=3,bbox_to_anchor=(-.08,1.22),loc='upper left');fig.tight_layout(w_pad=2);save(fig,'search_coverage')

fig,axes=plt.subplots(1,2,figsize=(6.4,3.1));policies=['never_recompile','always_adopt','retain_unless_better'];labs=['Retain initial','Daily replacement','Score-protected'];x=np.arange(2);w=.24
for j,(pol,col) in enumerate(zip(policies,[grey,gold,teal])):
 vals=[next(r['mean_selected_tvd'] for r in internal if r['stratum']==s and r['method']=='swarm_unique_56' and r['policy']==pol) for s in strata];axes[0].bar(x+(j-1)*w,vals,width=w,color=col,label=labs[j])
axes[0].set(xticks=x,xticklabels=['Familiar','New'],ylabel='Mean simulated TVD',ylim=(0,.095));axes[0].legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.35));label(axes[0],'(a)')
out=np.array([[173,86,71],[373,156,131]],float);left=np.zeros(2)
for j,(lab,col) in enumerate(zip(['Helpful','Little difference','Harmful'],[teal,'#C4CDD3',red])):
 vals=100*out[:,j]/out.sum(axis=1);axes[1].barh(x,vals,left=left,color=col,label=lab,height=.45)
 for k in range(2):axes[1].text(left[k]+vals[k]/2,k,str(int(out[k,j])),ha='center',va='center',fontsize=8,color='white' if j!=1 else navy)
 left+=vals
axes[1].set(yticks=x,yticklabels=['Familiar','New'],xlabel='Daily replacement cases (%)',xlim=(0,100));axes[1].legend(frameon=False,fontsize=7,bbox_to_anchor=(.5,1.25),loc='upper center');label(axes[1],'(b)');fig.tight_layout(w_pad=1.5);save(fig,'reuse_policies')

timing=read('serial_timing.json')['summary'];fig,axes=plt.subplots(1,2,figsize=(6.4,3.3))
for i,(s,marker) in enumerate(zip(strata,['o','s'])):
 vals=[next(r['median_seconds'] for r in timing if r['stratum']==s and r['method']==m) for m in methods];axes[0].plot(vals,np.arange(6)+(i-.5)*.18,marker=marker,ls='',ms=5,label=snames[i],color=navy if i==0 else teal)
axes[0].set(xscale='log',yticks=range(6),yticklabels=[names[m] for m in methods],xlabel='Median serial compilation time (s)');axes[0].invert_yaxis();axes[0].grid(axis='x',which='major',alpha=.2);axes[0].legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(-.05,1.22));label(axes[0],'(a)')
for j,m in enumerate(methods):
 v=next(r for r in summary if r['stratum']==strata[1] and r['method']==m);t=next(r['median_seconds'] for r in timing if r['stratum']==strata[1] and r['method']==m);axes[1].scatter(t,v['mean_tvd'],color=colors[j],s=35);axes[1].annotate(names[m],(t,v['mean_tvd']),xytext=(4,4 if m not in ['sqgm','swarm_unique_56'] else -10),textcoords='offset points',fontsize=7)
axes[1].set(xscale='log',xlim=(.007,25),ylim=(.052,.115),xlabel='Median serial compilation time (s)',ylabel='Mean simulated TVD (new instances)');axes[1].grid(alpha=.2);label(axes[1],'(b)');fig.tight_layout(w_pad=2);save(fig,'cost_quality')
print('Generated 7 numerical plots and PNG previews; preserved the supplied architecture PDF and PNG.')
