import numpy as np, pandas as pd, itertools, collections
from sklearn.model_selection import GroupKFold
d=dict(np.load("_work/feats_train.npz")); tr=pd.read_csv("train.csv")
y=pd.read_csv("train_labels.csv").drop(columns=["id"]).values.reshape(-1,4,4); gi=y.argmax(2)
P=np.array(list(itertools.permutations(range(4)))); gold=np.array([np.where((P==g).all(1))[0][0] for g in gi])
ctx,out=d["ctx"],d["out"]
tri,vai=next(GroupKFold(4).split(tr,groups=tr.group))
def uni(a): return [int(x) for x in a if 2<=x<122]
def bi(a):
    c=uni(a); return [c[k]*1000+c[k+1] for k in range(len(c)-1)]
def bi_shuf(a, r):
    c=uni(a); rng=np.random.RandomState(r); c=list(rng.permutation(c)) if c else c; return [c[k]*1000+c[k+1] for k in range(len(c)-1)]
def run(fc, fo, name):
    pos=collections.Counter(); neg=collections.Counter(); np_=0; nn_=0
    for r in tri:
        for i in range(4):
            a=fc(ctx[r,i,1],r*4+i)
            for j in range(4):
                b=fo(out[r,j],r*4+j)
                for x in a:
                    for z in b:
                        if gi[r,i]==j: pos[(x,z)]+=1; np_+=1
                        else: neg[(x,z)]+=1; nn_+=1
    acc=0
    for r in vai:
        S=np.zeros((4,4))
        for i in range(4):
            a=fc(ctx[r,i,1],r*4+i)
            for j in range(4):
                b=fo(out[r,j],r*4+j)
                S[i,j]=sum(np.log((pos[(x,z)]+0.5)/np_)-np.log((neg[(x,z)]+0.5)/nn_) for x in a for z in b)
        acc+=S[np.arange(4),P].sum(1).argmax()==gold[r]
    print(f"{name:32s} acc={acc/len(vai):.4f}",flush=True)
run(lambda a,r: bi(a), lambda a,r: bi(a), "ctx bigram x out bigram (ordered)")
run(lambda a,r: bi_shuf(a,r), lambda a,r: bi_shuf(a,r+7), "ctx bigram x out bigram (shuffled)")
