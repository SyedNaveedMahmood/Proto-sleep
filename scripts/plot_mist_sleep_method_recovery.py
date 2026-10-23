"""Four aggregate-only method figures; never reads waveforms or checkpoints."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/"reports/mist_sleep/method_recovery_v1"
OUT=ROOT/"figures"
OUT.mkdir(exist_ok=True)
RESULT=json.loads((ROOT/"robustness_analysis.json").read_text())
PROBE=json.loads((ROOT/"stage_information_probe.json").read_text())
ARMS=("A0","M0","M1","M2","M3")
COLORS={"A0":"#5b6770","M0":"#267fa5","M1":"#ba7a19","M2":"#22905d","M3":"#834ca7"}

fig,ax=plt.subplots(figsize=(11,3.3))
ax.set_xlim(0,11);ax.set_ylim(0,3);ax.axis("off")
blocks=[(.25,"30 s EEG\nFpz-Cz, 100 Hz"),(2.4,"29 overlap\n2 s / 1 s tokens"),(4.55,"7/31/101-sample\nmulti-scale filters"),(6.7,"64 shared\nprototypes, τ=.1"),(8.85,"within-epoch GRU\n5 stage logits")]
for x,label in blocks:
 ax.add_patch(plt.Rectangle((x,.95),1.85,1.1,facecolor="#e8f0f6",edgecolor="#224e69",linewidth=1.3))
 ax.text(x+.925,1.5,label,ha="center",va="center",fontsize=9)
for x,_ in blocks[:-1]:ax.annotate("",xy=(x+2.12,1.5),xytext=(x+1.85,1.5),arrowprops={"arrowstyle":"->","color":"#224e69"})
ax.text(5.5,.5,"Two aligned views: stage CE for every arm; MPI embedding L2 + assignment KL for M2/M3; SupCon for M1/M3",ha="center",fontsize=8)
fig.tight_layout();fig.savefig(OUT/"01_architecture.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,view in zip(axes,("clean","bounded")):
 x=np.arange(len(ARMS))
 for j,seed in enumerate((123,456)):
  values=[RESULT["results"][f"{arm}_{seed}"][view]["stage"]["mean_participant_macro_f1"] for arm in ARMS]
  ax.bar(x+(.2 if j else -.2),values,width=.36,label=f"seed {seed}" if ax is axes[0] else None,
         color=[COLORS[a] for a in ARMS],alpha=.55 if j else 1)
 ax.set_xticks(x,ARMS);ax.set_ylim(0,.8);ax.set_title(f"{view.title()} SC development")
 ax.set_ylabel("Mean participant fixed-five Macro-F1")
 ax.grid(axis="y",alpha=.2)
axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"02_clean_and_bounded_macro_f1.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for ax,metric,title in zip(axes,("embedding_L2","prototype_KL"),("Embedding L2 drift","Prototype assignment KL")):
 for j,seed in enumerate((123,456)):
  arms=("M0","M1","M2","M3")
  ax.plot(arms,[RESULT["results"][f"{arm}_{seed}"]["bounded"][metric] for arm in arms],
          marker="o",label=f"seed {seed}")
 ax.set_title(title);ax.grid(alpha=.2)
axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"03_representation_stability.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.8))
stages=("W","N1","N2","N3","REM")
for ax,seed in zip(axes,(123,456)):
 for arm in ("M0","M1","M2","M3"):
  ax.plot(stages,RESULT["results"][f"{arm}_{seed}"]["clean"]["stage"]["per_stage_recall"],
          marker="o",label=arm,color=COLORS[arm])
 ax.set_title(f"Clean stage recall: seed {seed}");ax.set_ylim(0,1);ax.grid(alpha=.2)
axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"04_stage_recall.png",dpi=180);plt.close(fig)
print("Created four aggregate-only figures")
