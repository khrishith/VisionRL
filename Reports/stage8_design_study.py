# Design study for the randomized-exit EscapeRoom (Python port; read-only, not used by training).
import random
from collections import deque
L=["........","..##....","......#.","##.#..#.","...#....",".#...#..",".#...#..","........"]
N=8; blocked=set()
for r,row in enumerate(L):
    for x,c in enumerate(row):
        if c=='#': blocked.add((x,N-1-r))
def bl(x,z): return x<0 or x>=N or z<0 or z>=N or (x,z) in blocked
free=[(x,z) for x in range(N) for z in range(N) if (x,z) not in blocked]
def bfs(g):
    D={g:0};q=deque([g])
    while q:
        x,z=q.popleft()
        for dx,dz in((0,1),(0,-1),(-1,0),(1,0)):
            n=(x+dx,z+dz)
            if not bl(*n) and n not in D: D[n]=D[(x,z)]+1;q.append(n)
    return D
cands=sorted([c for c in free if c[0] in(0,N-1) or c[1] in(0,N-1)])
dist={c:bfs(c) for c in cands}
assert all(len(dist[c])==len(free) for c in cands),"unreachable cells"
held=[c for i,c in enumerate(cands) if i%4==3]; train=[c for i,c in enumerate(cands) if i%4!=3]
print("border exit candidates",len(cands),"| train",len(train),"| held-out",len(held),"| held-out cells",held)
print("max path distance over all (exit,cell):",max(max(d.values()) for d in dist.values()))
ACT=[(0,1),(0,-1),(-1,0),(1,0)]
MIN=6
def episode(policy,H,pool,rng):
    g=rng.choice(pool); D=dist[g]
    s=rng.choice([c for c in free if c!=g and D[c]>=MIN]); t=0
    while True:
        dx,dz=ACT[policy(rng)]; n=(s[0]+dx,s[1]+dz)
        if not bl(*n): s=n
        t+=1
        if s==g: return True
        if t>=H: return False
pol={ 'uniform random':lambda r:r.randrange(4),
      'Stage7 blind [.48,.02,.12,.38]':lambda r:r.choices(range(4),[.4838,.0195,.1214,.3753])[0],
      'always forward':lambda r:0,'always right':lambda r:3,
      'fwd/right 50/50':lambda r:r.choice([0,3]),'back/left 50/50':lambda r:r.choice([1,2])}
rng=random.Random(5)
Hs=(32,48,64,96,128)
print("\nblind success by step limit (all exits, 4000 eps each):")
print("%-34s"%"policy"+"".join("%8d"%H for H in Hs))
for name,p in pol.items():
    print("%-34s"%name+"".join("%7.1f%%"%(100*sum(episode(p,H,cands,rng) for _ in range(4000))/4000) for H in Hs))
ds=[dist[g][s] for g in cands for s in free if s!=g and dist[g][s]>=MIN]
print("\noptimal steps (start>=%d from exit): mean %.1f max %d min %d"%(MIN,sum(ds)/len(ds),max(ds),min(ds)))
print("starts available per exit (min):",min(sum(1 for s in free if s!=g and dist[g][s]>=MIN) for g in cands))
md=max(max(d.values()) for d in dist.values())
print("idle-step reward bound: -0.01 + 0.0005*maxdist = %.4f (must be < 0)"%(-0.01+0.0005*md))
