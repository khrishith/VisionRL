# Python port of RoomWorld rewards (read-only audit; not used by training).
from collections import deque
L=[".....GG.","..##....","......#.","##.#..#.","...#....",".#...#..",".#...#..","........"]
N=8; blocked=set(); goal=set()
for r,row in enumerate(L):
    for x,c in enumerate(row):
        z=N-1-r
        if c=='#': blocked.add((x,z))
        if c=='G': goal.add((x,z))
def bl(x,z): return x<0 or x>=N or z<0 or z>=N or (x,z) in blocked
D={g:0 for g in goal}; q=deque(goal)
while q:
    x,z=q.popleft()
    for dx,dz in((0,1),(0,-1),(-1,0),(1,0)):
        n=(x+dx,z+dz)
        if not bl(*n) and n not in D: D[n]=D[(x,z)]+1; q.append(n)
STEP,BUMP,GOAL,TIMEOUT,SCALE,G,H=-.01,-.02,1.,-1.,.05,.99,128
ACT=[(0,1),(0,-1),(-1,0),(1,0)]
def step(s,a,t):
    x,z=s; dx,dz=ACT[a]; prev=D[s]; b=bl(x+dx,z+dz)
    if not b: x,z=x+dx,z+dz
    t+=1; succ=(x,z) in goal; done=succ or t>=H
    pot=-SCALE*prev; nxt=0 if done else -SCALE*D[(x,z)]
    return (x,z),t,done,succ,STEP+(BUMP if b else 0)+(GOAL if succ else TIMEOUT if done else 0)+G*nxt-pot,b
def run(s,acts):
    t=0;tot=0;
    for a in acts:
        s,t,done,succ,r,b=step(s,a,t); tot+=r
        if done: break
    return tot,t,succ
free=[c for c in D if c not in goal]
print("cells",len(free),"max path distance",max(D.values()))
# cross-check against Unity test outputs
print("oscillation 40 steps from (0,0):",round(run((0,0),[0,1]*20)[0],3),"(Unity test: -0.170)")
far=max(free,key=lambda c:D[c]); s=far; acts=[]
while s not in goal:
    for a in range(4):
        n=(s[0]+ACT[a][0],s[1]+ACT[a][1])
        if not bl(*n) and D[n]==D[s]-1: acts.append(a); s=n; break
print("shortest route from",far,len(acts),"steps return:",round(run(far,acts)[0],3),"(Unity test: 1.513)")
# audit
print("\n-- per-step reward of ANY non-terminal no-progress step: -0.01 + 0.0005*d (d=path distance)")
print("   worst case (d=12): %.4f ; bump worst case: %.4f -> always negative" % (STEP+0.0005*12, STEP+BUMP+0.0005*12))
best=max(STEP+(G-1)*(-SCALE*D[c]) for c in free); print("   max over all cells of idle-step reward: %.4f"%best)
print("\n-- 128 steps pure bumping from each free cell (undiscounted return): min/max")
vals=[]
for c in free:
    bump=None
    for a in range(4):
        dx,dz=ACT[a]
        if bl(c[0]+dx,c[1]+dz): bump=a;break
    if bump is not None: vals.append((run(c,[bump]*H)[0],c))
print("   %.3f .. %.3f"%(min(vals)[0],max(vals)[0]))
print("\n-- 128-step oscillation (fwd/back) between two open cells, best over all cells:")
bo=-9
for c in free:
    for a in range(4):
        n=(c[0]+ACT[a][0],c[1]+ACT[a][1])
        if not bl(*n) and n not in goal:
            bo=max(bo,run(c,[a,a^1]*64)[0])
print("   best undiscounted return %.3f (a timeout; success returns ~+1)"%bo)
print("\n-- return of an optimal episode vs timeout, by start distance")
for d in (1,6,12):
    c=[c for c in free if D[c]==d][0]; s=c; ac=[]
    while s not in goal:
        for a in range(4):
            n=(s[0]+ACT[a][0],s[1]+ACT[a][1])
            if not bl(*n) and D[n]==D[s]-1: ac.append(a); s=n; break
    print("   d=%2d optimal %.3f | timeout by bumping %.3f"%(d,run(c,ac)[0],run(c,[ [a for a in range(4) if bl(c[0]+ACT[a][0],c[1]+ACT[a][1])][0] ]*H)[0] if any(bl(c[0]+ACT[a][0],c[1]+ACT[a][1]) for a in range(4)) else float('nan')))
# discounted sanity: with gamma .99, discounted value of a loop of k steps returning to same state
print("\n-- discounted (gamma=.99) return of a 2-step loop from d=12 cell vs starting value change: shaping is policy-invariant;")
c=far; n=(c[0]+ACT[0][0],c[1]+ACT[0][1])
print("   discounted 200-step oscillation from",c,"=%.3f"%sum((G**i)*r for i,r in enumerate([step(s,a,0)[4] for s,a in [(c,0),(n,1)]]*100)))
import random
random.seed(1)
tot=[];succ=0;L_=[]
for _ in range(20000):
    s=random.choice(free); t=0;R=0
    while True:
        s,t,done,sc,r,b=step(s,random.randrange(4),t); R+=r
        if done: break
    tot.append(R); succ+=sc; L_.append(t)
print("\nPython-port random policy (20000 eps): success %.3f mean return %.3f mean length %.1f"%(succ/20000,sum(tot)/20000,sum(L_)/20000))
