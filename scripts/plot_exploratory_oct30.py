"""Publication-style descriptive plots; no unsupported inferential error bars."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from run_exploratory_oct30 import OUT, REPORT, load

COLORS={'swarm':'#4361A6','swarm_unique_56':'#087F8C','random_fixed_56':'#B15D35'}
LABELS={'swarm':'Original swarm','swarm_unique_56':'Modified swarm','random_fixed_56':'Random search'}
STRATA=['known_instance_new_period','new_instance_new_period']
TITLES={'known_instance_new_period':'5 known circuit instances','new_instance_new_period':'10 new circuit instances'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'figure.dpi':160,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#B6BEC8','axes.labelcolor':'#253449','text.color':'#253449',
        'xtick.color':'#526174','ytick.color':'#526174','grid.color':'#DFE5EB','grid.linewidth':.6,
        'svg.fonttype':'none'})

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(REPORT/(name+'.'+ext),facecolor='white')
    plt.close(fig)

def main():
    style();p=load(OUT/'plan.json');cal=load(REPORT/'calibration_audit.json')
    rows=load(REPORT/'by_date.json');ds=p['paired_dates'];x=np.arange(len(ds)); tick=[d[-2:] for d in ds]
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    fig.subplots_adjust(top=.80,bottom=.14,left=.08,right=.97,wspace=.15,hspace=.26)
    for col,st in enumerate(STRATA):
        for row,policy in enumerate(['always_adopt','retain_unless_better']):
            ax=axes[row,col]
            for method in p['methods']:
                rr={r['date']:r for r in rows if (r['stratum'],r['method'],r['policy'])==(st,method,policy)}
                y=[rr[d]['mean_selected_tvd'] for d in ds]
                ax.plot(x,y,color=COLORS[method],marker='o',markersize=3.5,linewidth=1.8,label=LABELS[method])
            for date in cal['unchanged_value_transition_dates']:
                i=ds.index(date);ax.axvspan(i-.35,i+.35,color='#8C96A8',alpha=.12,lw=0)
            ax.grid(axis='y');ax.set_axisbelow(True)
            ax.set_title(TITLES[st]+'\n'+('Daily recompilation' if row==0 else 'Score-protected adoption'),loc='left',pad=10)
            ax.set_xticks(x,tick)
            if col==0:ax.set_ylabel('Mean output TVD ↓')
            if row==1:ax.set_xlabel('October 2026 · daily cutoff')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle('Output quality across the available calibration period',x=.08,y=.97,ha='left',fontsize=17,fontweight='bold')
    fig.text(.08,.925,'11 paired dates · 3 search seeds · fixed and resampled seeds · 56 proposals per search',fontsize=10,color='#526174')
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.07,.90),ncol=3,frameon=False)
    fig.text(.08,.055,'Lower TVD is better. Shading marks unchanged calibration values from the preceding cutoff.\nEqual circuit/seed/rule weighting; correlated exploratory simulations, not physical-device trials.',fontsize=9,color='#526174')
    save(fig,'quality_by_date')

    rows=load(REPORT/'by_circuit.json');lookup={(r['circuit_id'],r['method'],r['policy']):r for r in rows}
    names={
        'ghz_4':'Known · GHZ 4q','product_4':'Known · product 4q',
        'rotcx_line_n4_l6':'Known · line 4q','rotcx_ring_n4_l6':'Known · ring 4q','rotcx_star_n5_l6':'Known · star 5q'}
    for c in p['bank']:
        if c not in names:
            family=p['bank'][c]['family'];family='rotated GHZ' if family=='ghz_rotated' else family.replace('_',' ')
            n=p['bank'][c]['qubits'];names[c]=f'New · {family} {n}q'
    cs=list(p['bank']);y=np.arange(len(cs));fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True,sharex=True)
    fig.subplots_adjust(top=.84,bottom=.13,left=.20,right=.97,wspace=.12)
    for ax,control in zip(axes,['swarm','random_fixed_56']):
        gains=[lookup[c,control,'always_adopt']['mean_selected_tvd']-lookup[c,'swarm_unique_56','always_adopt']['mean_selected_tvd'] for c in cs]
        ax.barh(y,gains,color=['#087F8C' if g>1e-4 else '#BD4F51' if g< -1e-4 else '#909CAA' for g in gains],height=.65)
        ax.axvline(0,color='#253449',lw=.8);ax.axhline(4.5,color='#B6BEC8',lw=.7,ls='--')
        ax.grid(axis='x');ax.set_axisbelow(True);ax.set_yticks(y,[names[c] for c in cs]);ax.set_title('Against '+LABELS[control].lower(),loc='left',pad=10)
        ax.set_xlabel('Control TVD − modified swarm TVD →')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].invert_yaxis()
    maximum=max(abs(lookup[c,control,'always_adopt']['mean_selected_tvd']-lookup[c,'swarm_unique_56','always_adopt']['mean_selected_tvd']) for c in cs for control in ['swarm','random_fixed_56'])
    axes[0].set_xlim(-maximum*1.12,maximum*1.12)
    fig.suptitle('Does the modified swarm help every circuit?',x=.20,y=.965,ha='left',fontsize=17,fontweight='bold')
    fig.text(.20,.91,'Fresh recompilation · mean over all 11 paired dates and 6 seed/rule cases per date',fontsize=10,color='#526174')
    fig.text(.20,.045,'Right / teal: modified swarm improves quality. Left / red: quality worsens. Gray: |difference| ≤ 0.0001.\nAbsolute TVD differences; related circuit families and repeated dates are not independent trials.',fontsize=9,color='#526174')
    save(fig,'gain_by_circuit')

    rows=load(REPORT/'method_summary.json');lookup={(r['stratum'],r['method'],r['policy']):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharex=True,sharey=True)
    fig.subplots_adjust(top=.72,bottom=.22,left=.16,right=.98,wspace=.15)
    colors=['#087F8C','#CBD3DC','#BD4F51']
    for ax,st in zip(axes,STRATA):
        yy=np.arange(3);left=np.zeros(3)
        for outcome,color in zip(['help','little','harm'],colors):
            values=np.array([lookup[st,m,'always_adopt']['recompile_outcomes'][outcome]/lookup[st,m,'always_adopt']['cases']*100 for m in p['methods']])
            ax.barh(yy,values,left=left,color=color,height=.6,label={'help':'Helpful','little':'Little difference','harm':'Harmful'}[outcome])
            for i,(a,v) in enumerate(zip(left,values)):
                if v>=7:ax.text(a+v/2,i,f'{v:.1f}%',ha='center',va='center',color='#253449' if outcome=='little' else 'white',fontsize=9)
            left+=values
        ax.set_xlim(0,100);ax.set_yticks(yy,[LABELS[m] for m in p['methods']]);ax.set_title(TITLES[st],loc='left',pad=10);ax.set_xlabel('Share of paired cases (%)')
    axes[0].invert_yaxis()
    fig.suptitle('Recompilation can help, harm, or change little',x=.16,y=.97,ha='left',fontsize=17,fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.15,.89),ncol=3,frameon=False)
    fig.text(.16,.075,'Compared with each method’s preceding daily winner; threshold ±0.0001 TVD.\n330 cases/method for known instances; 660 for new instances. Cases are correlated.',fontsize=9,color='#526174')
    save(fig,'recompilation_outcomes')
    print('Saved three figures as PNG and editable SVG.')

if __name__=='__main__':main()
