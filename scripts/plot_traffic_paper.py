"""Render the development figure and, optionally, a verified HPC report directory."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES={'pems03':'PEMS03','pems04':'PEMS04','sh':'Shanghai Metro','hz':'Hangzhou Metro',
       'bike':'NYC bike','taxi':'NYC taxi','bj':'Beijing Metro','pems08':'PEMS08'}


def development(out):
    with (ROOT/'results/development/window_scores.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    fig, axes=plt.subplots(1,2,figsize=(8.5,3.35),sharey=True,layout='constrained')
    objectives={'suffix_mae':('Ordinary suffix','#455668','o'),'prefix_mae':('Ordinary prefix','#bd682b','s'),
                'verification_margin':('Verification margin','#157777','^')}
    for ax,d in zip(axes,('pems08','bj')):
        ax.axhline(0,color='#888',lw=.8)
        for arm,(label,color,marker) in objectives.items():
            diff=[]
            for window in range(1,5):
                chosen=[r for r in rows if r['dataset']==d and r['objective']==arm and int(r['window'])==window]
                l=mean(float(r['local_mae']) for r in chosen)
                a=mean(float(r['deployed_mae']) for r in chosen)
                diff.append(100*(a/l-1))
            ax.plot(range(1,5),diff,label=label,linestyle='-',color=color,marker=marker,alpha=.9)
        ax.set(xticks=[1,2,3,4],xlabel='Chronological development window',title=NAMES[d])
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Change in MAE vs local (%)')
    axes[0].text(2.5,-3.2,'All three rules coincide at zero',ha='center',fontsize=8)
    axes[1].legend(fontsize=8,frameon=False)
    fig.savefig(out/'development_windows.svg',metadata={'Date':None},bbox_inches='tight')
    plt.close(fig)


def heldout(report,out):
    values=json.loads((report/'blend_comparison.json').read_text())
    fig,ax=plt.subplots(figsize=(8.2,4.5),layout='constrained')
    controls={'local':'Local','verification_margin/half':'Equal blend','prefix_mae/prefix_blend':'Matched prefix objective',
              'B03/prefix_blend':'LLM-UTP adaptation','B04/prefix_blend':'Qwen2.5'}
    for j,(key,title) in enumerate(controls.items()):
        x=values[key]
        if x['equal_domain_effect'] is None:
            continue
        point=x['equal_domain_effect']*100
        ci=x['descriptive_95_intervals']['144']
        if ci is not None:
            ax.hlines(j,ci[0]*100,ci[1]*100,color='#157777',lw=2)
        ax.plot(point,j,'o',color='#193f55')
    ax.axvline(0,color='#888',lw=.8)
    ax.set(yticks=range(len(controls)),yticklabels=list(controls.values()),
           xlabel='Weighted margin forecast minus control (percentage points of local MAE)')
    ax.invert_yaxis()
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'heldout_comparisons.svg',metadata={'Date':None},bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'paper/figures')
    args=p.parse_args()
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'ectt-publication','font.family':'DejaVu Sans'})
    args.output.mkdir(parents=True,exist_ok=True)
    development(args.output)
    if args.report:
        heldout(args.report,args.output)
