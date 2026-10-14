import sys; sys.path.insert(0,'Reports')
import numpy as np
from stage8_labels import *
d=np.load("/private/tmp/claude-501/-Users-mallareddykadthala-Desktop-Unity-VisionRL/4703ee71-7c23-48bf-b683-536d8c474d5d/scratchpad/diag_data.npz"); obs=d['obs'].astype(np.float32); act=d['act']; rew=d['rew']; step=d['step']
ok=0;bad=0;none=0;lab=[]
for i in range(len(obs)):
    c=cells_from_frame(obs[i])
    if c is None: none+=1; lab.append(None); continue
    rob,ex=c
    if bl(*rob) or rob==ex: bad+=1; lab.append(None); continue
    e,_,_,_=expected_reward(rob,act[i],ex,step[i])
    if abs(e-rew[i])<2e-3: ok+=1; lab.append((rob,ex))
    else: bad+=1; lab.append(None)
n=len(obs); print(f"transitions {n}: pixel-derived labels reproduce the REAL reward exactly for {ok} ({100*ok/n:.1f}%); mismatches {bad}; undetected {none}")
import pickle; pickle.dump(lab,open("/private/tmp/claude-501/-Users-mallareddykadthala-Desktop-Unity-VisionRL/4703ee71-7c23-48bf-b683-536d8c474d5d/scratchpad/diag_labels.pkl","wb"))
# reward signal analysis from the real rewards
prog=[];away=[];bump=[];same=[]
for i,l in enumerate(lab):
    if l is None or d['term'][i]: continue
    rob,ex=l; e,b,new,_=expected_reward(rob,act[i],ex,step[i]); D=DIST[ex]
    (bump if b else prog if D[new]<D[rob] else away).append(rew[i])
for nm,v in (("progress (closer)",prog),("blocked (wall/obstacle)",bump),("farther away",away)):
    print(f"  real reward, {nm:24s}: n={len(v):5d} mean {np.mean(v):+.4f} min {np.min(v):+.4f} max {np.max(v):+.4f}")
fr=len(prog)/(len(prog)+len(bump)+len(away)); print(f"  under a uniform random policy {100*fr:.0f}% of steps make progress; per-step reward gap progress vs others = {np.mean(prog)-np.mean(bump+away):+.3f}")
exs=[l[1] for l in lab if l]; import collections
side=collections.Counter(('N' if e[1]==7 else 'S' if e[1]==0 else 'W' if e[0]==0 else 'E') for e in exs)
print("  exit sides seen in collected frames:",dict(side),"| distinct exits:",len(set(exs)),"of 27 | held-out exits:",sum(1 for e in set(exs) if e in held))
