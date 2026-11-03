"""Publication-style descriptive figures for the frozen baseline comparison."""
from published_baseline_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
LABELS={'swarm_unique_56':'Our modified swarm','lightsabre':'LightSABRE','mapomatic':'Mapomatic','sqgm':'SQGM','qmap':'MQT QMAP','duostra':'Duostra'}
ORDER=['swarm_unique_56','mapomatic','lightsabre','sqgm','qmap','duostra']
COLORS={'swarm_unique_56':'#176B77','mapomatic':'#B65B2B','lightsabre':'#3E6280','sqgm':'#738C50','qmap':'#8B6FA8','duostra':'#78838F'}
def save(fig,name):
    fig.savefig(REPORT/(name+'.png'),dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(REPORT/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.titleweight':'bold','axes.labelcolor':'#253443','text.color':'#253443','xtick.color':'#435262','ytick.color':'#435262','svg.fonttype':'none'})
    summary=load(REPORT/'method_summary.json');comparisons=load(REPORT/'paired_comparisons.json')
    strata=['known_instance_new_period','new_instance_new_period'];titles=['5 known circuit instances','10 new instances of related families']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,st,title in zip(axes,strata,titles):
        rs={r['method']:r for r in summary if r['stratum']==st};values=[rs[m]['mean_tvd'] for m in ORDER]
        ax.barh(range(len(ORDER)),values,color=[COLORS[m] for m in ORDER],height=.64)
        ax.set_yticks(range(len(ORDER)),[LABELS[m] for m in ORDER]);ax.set_ylim(len(ORDER)-.5,-.5);ax.set_xlabel('Mean simulated output error (TVD) ↓');ax.set_title(title,pad=15,loc='left')
        ax.set_xlim(0,max(r['mean_tvd'] for r in summary)*1.24);ax.xaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(v+max(values)*.018,i,f'{v:.5f}',va='center',fontsize=9)
    fig.suptitle('Direct comparison on the same calibration period',x=.04,ha='left',fontsize=16,fontweight='bold',y=1.03)
    fig.text(.04,-.03,'11 daily comparisons · 4–5 logical qubits · one 6-qubit patch · exact noisy simulation\nNative search budgets differ. Means are descriptive; no hardware or statistical-superiority claim.',fontsize=9,color='#536270')
    fig.tight_layout(w_pad=2);save(fig,'quality_comparison')
    daily=load(REPORT/'by_date.json');fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharex=True)
    dates=sorted({r['date'] for r in daily})
    for ax,st,title in zip(axes,strata,titles):
        for m in ORDER[1:]:
            ix={r['date']:r for r in daily if r['stratum']==st and r['baseline']==m}
            ax.plot(range(len(dates)),[ix[d]['mean_gain'] for d in dates],label=LABELS[m],color=COLORS[m],marker='o',ms=3,lw=1.6)
        ax.axhline(0,color='#253443',lw=1);ax.set_title(title,loc='left');ax.set_xticks(range(0,len(dates),2),[d[8:] for d in dates[::2]]);ax.set_xlabel('October 2026 cutoff date');ax.set_ylabel('Baseline TVD − our TVD');ax.grid(alpha=.14);ax.yaxis.set_major_locator(MaxNLocator(5))
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.55,.98),ncol=5)
    fig.suptitle('Daily paired quality differences',x=.04,ha='left',fontsize=16,fontweight='bold',y=1.03)
    fig.text(.04,-.03,'Positive values favor our swarm; negative values favor the published baseline.\nDates and repeated seeds are correlated. Two dates have unchanged calibration values.',fontsize=9,color='#536270')
    fig.tight_layout(w_pad=2,rect=(0,0,1,.87));save(fig,'daily_paired_differences')
    if (REPORT/'serial_timing.json').exists():
        timing=load(REPORT/'serial_timing.json')['summary'];fig,ax=plt.subplots(figsize=(8,4.4));offsets=[-.17,.17]
        for st,off,color,title in zip(strata,offsets,['#244A73','#E6A15A'],titles):
            rs={r['method']:r for r in timing if r['stratum']==st}
            ax.barh(np.arange(len(ORDER))+off,[rs[m]['median_seconds'] for m in ORDER],height=.3,label=title,color=color)
        ax.set_yticks(range(len(ORDER)),[LABELS[m] for m in ORDER]);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Median compilation seconds, logarithmic scale ↓');ax.legend(frameon=False,fontsize=8,loc='best');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_title('Serial cost of the tested pipelines',loc='left',fontsize=15,pad=18)
        fig.text(.02,-.06,'3 technical repeats per circuit and method · cold compiler/cache objects · same pinned CPU\nNative budgets and implementation languages differ; this is not a matched-time quality comparison.',fontsize=9,color='#536270')
        fig.tight_layout();save(fig,'serial_compilation_cost')
if __name__=='__main__':main()
