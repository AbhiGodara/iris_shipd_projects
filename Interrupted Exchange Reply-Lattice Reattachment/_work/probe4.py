import numpy as np, pandas as pd, itertools, collections
from sklearn.model_selection import GroupKFold
d = dict(np.load("_work/feats_train.npz"))
tr = pd.read_csv("train.csv"); lab = pd.read_csv("train_labels.csv")
y = lab.drop(columns=["id"]).values.reshape(-1,4,4); gi = y.argmax(2)
P = np.array(list(itertools.permutations(range(4))))
gold = np.array([np.where((P==g).all(1))[0][0] for g in gi])
ctx,out,succ=d["ctx"],d["out"],d["succ"]
def ckey(a): return tuple(int(x) for x in a if 2<=x<122)
def fkey(a): return tuple(int(x) for x in a if x!=0)
tri,vai=next(GroupKFold(4).split(tr,groups=tr.group))
for name,kf in (("content",ckey),("full",fkey)):
    pair=collections.Counter(); lastc=collections.Counter(); prevpair=collections.Counter(); outc=collections.Counter()
    for r in tri:
        for i in range(4):
            j=gi[r,i]; pair[(kf(ctx[r,i,1]),kf(out[r,j]))]+=1; lastc[kf(ctx[r,i,1])]+=1
            prevpair[(kf(ctx[r,i,0]),kf(ctx[r,i,1]),kf(out[r,j]))]+=1
        for j in range(4): outc[kf(out[r,j])]+=1
    anyhit=0; acc=0; tp=0; fp=0
    for r in vai:
        S=np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                S[i,j]=pair[(kf(ctx[r,i,1]),kf(out[r,j]))] + 3*prevpair[(kf(ctx[r,i,0]),kf(ctx[r,i,1]),kf(out[r,j]))]
                if S[i,j]>0: 
                    if gi[r,i]==j: tp+=1
                    else: fp+=1
        if S.sum()>0: anyhit+=1
        k=(S[np.arange(4),P].sum(1)).argmax(); acc+=(k==gold[r])
    print(name, "rows w/ any pair hit", anyhit/len(vai), "cells hit true", tp/(4*len(vai)), "false", fp/(12*len(vai)), "perm acc (ties->first)", acc/len(vai))
# distribution of content key lengths among hits
lens=collections.Counter(len(ckey(ctx[r,i,1])) for r in vai for i in range(4))
print(sorted(lens.items())[:14])
