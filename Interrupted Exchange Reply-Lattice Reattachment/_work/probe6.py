import numpy as np, pandas as pd, itertools
d=dict(np.load("_work/feats_train.npz"))
y=pd.read_csv("train_labels.csv").drop(columns=["id"]).values.reshape(-1,4,4); gi=y.argmax(2)
out,rel=d["out"],d["rel"]
def s(a,lo,hi): return set(int(x) for x in a if lo<=x<hi)
res={"w":[[],[]],"n":[[],[]]}
for r in range(len(out)):
    for a in range(4):
        for b in range(4):
            if rel[r,a,b,0]:  # a.last == b.prev  => a precedes b
                ja,jb=gi[r,a],gi[r,b]
                for t,(lo,hi) in (("w",(2,122)),("n",(122,378))):
                    A=s(out[r,ja],lo,hi)
                    for j in range(4):
                        if j==ja: continue
                        res[t][int(j==jb)].append(len(A & s(out[r,j],lo,hi)))
for t in res: print(t, "succ-outcome overlap %.3f vs other %.3f"%(np.mean(res[t][1]),np.mean(res[t][0])), len(res[t][1]))
