import numpy as np, pandas as pd, itertools, collections
from sklearn.model_selection import GroupKFold
d = dict(np.load("_work/feats_train.npz"))
tr = pd.read_csv("train.csv"); lab = pd.read_csv("train_labels.csv"); te=pd.read_csv("test.csv")
y = lab.drop(columns=["id"]).values.reshape(-1,4,4); gi = y.argmax(2)
ctx,out,succ,rel,kind=d["ctx"],d["out"],d["succ"],d["rel"],d["kind"]
print("rel type freq per row (sum over i,k):", rel.sum((1,2)).mean(0))
print("n contexts with prev:", (ctx[:,:,0,0]!=0).mean(), "with succ:", (succ[:,:,0]!=0).mean())
# relation-pattern vs gold: for chain i->k (i.last==k.prev), is kind of outcome(i) related?
# duplicates across groups
def ckey(a): return tuple(int(x) for x in a if 2<=x<122)
tri,vai=next(GroupKFold(4).split(tr,groups=tr.group))
trainlast=collections.defaultdict(set); trainout=set()
for r in tri:
    for i in range(4): trainlast[ckey(ctx[r,i,1])].add(ckey(out[r,gi[r,i]]))
    for j in range(4): trainout.add(ckey(out[r,j]))
hit=0; outhit=0; tot=0
for r in vai:
    for i in range(4):
        k=ckey(ctx[r,i,1]); tot+=1
        if k in trainlast: hit+=1
        if ckey(out[r,gi[r,i]]) in trainout: outhit+=1
print("val last-seg content seen in train:", hit/tot, " val true outcome content seen:", outhit/tot)
# full-token (with filler) uniqueness
full=collections.Counter(tuple(ctx[r,i,1]) for r in range(len(tr)) for i in range(4))
print("distinct last segs", len(full), "of", 4*len(tr))
# content length relation
cl=(ctx[:,:,1]>=2)&(ctx[:,:,1]<122); ol=(out>=2)&(out<122)
cn=cl.sum(-1); on=ol.sum(-1)
pos=[(cn[r,i],on[r,gi[r,i]]) for r in range(3000) for i in range(4)]
print("corr ctx content len vs true outcome len", np.corrcoef(np.array(pos).T)[0,1])
# kind vs whether context has succ / prev
hs=(succ[:,:,0]!=0); hp=(ctx[:,:,0,0]!=0)
kt=np.array([[kind[r,gi[r,i]] for i in range(4)] for r in range(len(tr))])
print("P(event|has_succ)",kt[hs].mean(),"P(event|no succ)",kt[~hs].mean(),"P(event|has_prev)",kt[hp].mean(),"P(event|no prev)",kt[~hp].mean())
# position of token in outcome: is n_ filler at fixed positions?
fpos=((out>=122)).mean((0,1)); print("filler rate by position", np.round(fpos,2))
cpos=((ctx[:,:,1]>=122)).mean((0,1)); print("ctx last filler by pos", np.round(cpos,2))
# test structure
