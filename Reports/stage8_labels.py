# Shared helpers: pixel -> (robot cell, exit cell), port of the room rules, label validation.
import numpy as np
from collections import deque
L=["........","..##....","......#.","##.#..#.","...#....",".#...#..",".#...#..","........"]
N=8; blocked={(x,N-1-r) for r,row in enumerate(L) for x,c in enumerate(row) if c=='#'}
bl=lambda x,z: x<0 or x>=N or z<0 or z>=N or (x,z) in blocked
free=[(x,z) for x in range(N) for z in range(N) if (x,z) not in blocked]
cands=sorted([c for c in free if c[0] in(0,N-1) or c[1] in(0,N-1)])
held={c for i,c in enumerate(cands) if i%4==3}
def bfs(g):
    D={g:0};q=deque([g])
    while q:
        x,z=q.popleft()
        for dx,dz in((0,1),(0,-1),(-1,0),(1,0)):
            n=(x+dx,z+dz)
            if not bl(*n) and n not in D: D[n]=D[(x,z)]+1;q.append(n)
    return D
DIST={c:bfs(c) for c in cands}
ACT=[(0,1),(0,-1),(-1,0),(1,0)]
S=84/9.2
def cells_from_frame(f):             # f: (3,84,84) float 0..1
    r,g,b=f[0],f[1],f[2]
    blue=(b>.7)&(r<.4)&(g<.5); green=(g>.7)&(r<.4)&(b<.4)
    if blue.sum()<3 or green.sum()<10: return None
    ry,rx=np.nonzero(blue); gy,gx=np.nonzero(green)
    rob=(int(round((rx.mean()-42)/S+3.5)),int(round(3.5-(ry.mean()-42)/S)))
    gwx=(gx.mean()-42)/S+3.5; gwz=3.5-(gy.mean()-42)/S
    ex=min(cands,key=lambda c:(c[0]-gwx)**2+(c[1]-gwz)**2)
    return rob,ex
def expected_reward(rob,act,ex,step,H=48):
    dx,dz=ACT[act]; n=(rob[0]+dx,rob[1]+dz); bump=bl(*n); new=rob if bump else n
    D=DIST[ex]; succ=new==ex; done=succ or step>=H
    nxt=0 if done else -.05*D[new]
    return -.01+(-.02 if bump else 0)+(1. if succ else -1. if done else 0)+.99*nxt+.05*D[rob], bump, new, succ
def optimal_actions(rob,ex):
    D=DIST[ex]; return [a for a in range(4) if not bl(rob[0]+ACT[a][0],rob[1]+ACT[a][1]) and D[(rob[0]+ACT[a][0],rob[1]+ACT[a][1])]<D[rob]]
