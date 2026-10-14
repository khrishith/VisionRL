# Measures whether the camera frame shows the robot where the game actually has it.
# Truth = replaying the executed actions through the game rules from the first frame of each episode (validated against real rewards).
import sys; sys.path.insert(0,'Reports')
import numpy as np, collections
from stage8_labels import *
path=sys.argv[1]
d=np.load(path); obs=d['obs'].astype(np.float32); act=d['act']; rew=d['rew']; term=d['term']; step=d['step']
n=len(obs); c=[cells_from_frame(o) for o in obs]
eps=[];s=0
for i in range(n):
    if term[i]: eps.append((s,i)); s=i+1
rew_ok=rew_tot=0; agree=np.zeros(50); tot=np.zeros(50); changes=0
for (a,b) in eps:
    pos,ex=c[a]
    for k,i in enumerate(range(a,b+1)):
        e,_,new,_=expected_reward(pos,act[i],ex,step[i]); rew_tot+=1; rew_ok+=abs(e-rew[i])<2e-3
        agree[k]+=(c[i][0]==pos); tot[k]+=1; pos=new
frames=len(obs); distinct=len({o.tobytes() for o in d['obs']})
cc=sum(1 for i in range(1,n) if c[i][0]!=c[i-1][0])
print(f"{path.split('/')[-1]}: {n} transitions, {len(eps)} episodes")
print(f"  replay-from-first-frame reproduces the real reward on {100*rew_ok/rew_tot:.1f}% of steps")
print(f"  frame shows the TRUE robot cell: overall {100*agree.sum()/tot.sum():.1f}%  | step 1: {100*agree[0]/tot[0]:.0f}%, steps 2-10: {100*agree[1:10].sum()/tot[1:10].sum():.1f}%, steps 11-48: {100*agree[10:48].sum()/max(tot[10:48].sum(),1):.1f}%")
print(f"  derived robot cell changed {cc} times over {n} frames; distinct frames {distinct}")

# ---- acceptance checks (Stage 8 render fix) ----
moved_total=moved_changed=bump_total=bump_same=0; exit_const=exit_total=0
for (a,b) in eps:
    pos,ex=c[a]
    for i in range(a,b):
        e,bump,new,_=expected_reward(pos,act[i],ex,step[i])
        if i+1<=b:
            if bump: bump_total+=1; bump_same+= (c[i+1][0]==c[i][0])
            else: moved_total+=1; moved_changed+= (c[i+1][0]!=c[i][0])
            exit_total+=1; exit_const+= (c[i+1][1]==c[i][1])
        pos=new
agree_pct=100*agree.sum()/tot.sum(); rew_pct=100*rew_ok/rew_tot
print(f"  frame changes when the robot truly moved: {100*moved_changed/max(moved_total,1):.1f}% of {moved_total} moves")
print(f"  robot position in frame unchanged on blocked moves: {100*bump_same/max(bump_total,1):.1f}% of {bump_total} bumps")
print(f"  exit position stable within episodes: {100*exit_const/max(exit_total,1):.1f}%")
ok = agree_pct>=99 and rew_pct>=99 and moved_changed/max(moved_total,1)>=.99 and bump_same/max(bump_total,1)>=.99
print(f"  ACCEPTANCE (>=99% on position agreement, reward sync, frame-follows-move, frame-still-on-bump): {'PASS' if ok else 'FAIL'}")
