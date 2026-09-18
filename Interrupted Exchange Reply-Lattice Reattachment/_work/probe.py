import numpy as np, pandas as pd, itertools
from sklearn.model_selection import GroupKFold
d = dict(np.load("_work/feats_train.npz"))
tr = pd.read_csv("train.csv"); lab = pd.read_csv("train_labels.csv")
y = lab.drop(columns=["id"]).values.reshape(-1,4,4); gi = y.argmax(2)
P = np.array(list(itertools.permutations(range(4))))
gold = np.array([np.where((P==g).all(1))[0][0] for g in gi])
V=378
ctx, out, succ = d["ctx"], d["out"], d["succ"]
def isw(a): return (a>=2)&(a<98)
def isv(a): return (a>=98)&(a<122)
def content(a): return isw(a)|isv(a)
tri, vai = next(GroupKFold(4).split(tr, groups=tr.group))
def pairs_feats(A, B):
    # A: [N,4,L] ctx tokens, B: [N,4,L] outcome tokens -> list per (r,i,j) of co-occur (a,b)
    return A, B
def eval_table(sel_c, sel_o, name, alpha=1.0):
    # sel_c(r,i)-> array of ctx tokens, sel_o(r,j)-> outcome tokens. Build log-odds table on tri, score vai.
    pos = np.full((V,V), alpha); neg = np.full((V,V), alpha)
    for r in tri:
        for i in range(4):
            a = sel_c(r,i)
            if len(a)==0: continue
            for j in range(4):
                b = sel_o(r,j)
                if len(b)==0: continue
                aa, bb = np.meshgrid(a,b, indexing="ij")
                np.add.at(pos if gi[r,i]==j else neg, (aa.ravel(), bb.ravel()), 1)
    lo = np.log(pos/pos.sum()) - np.log(neg/neg.sum())
    acc=0; cell=0
    for r in vai:
        S=np.zeros((4,4))
        for i in range(4):
            a=sel_c(r,i)
            for j in range(4):
                b=sel_o(r,j)
                if len(a) and len(b): S[i,j]=lo[np.ix_(a,b)].sum()
        ps = S[np.arange(4),P].sum(1); k=ps.argmax()
        acc += k==gold[r]; cell += (P[k]==gi[r]).mean()
    print(f"{name:40s} perm_acc={acc/len(vai):.4f} cell_acc={cell/len(vai):.4f}", flush=True)
def toks(a, f): return a[f(a)]
eval_table(lambda r,i: toks(ctx[r,i,1],content), lambda r,j: toks(out[r,j],content), "last-seg content x outcome content")
eval_table(lambda r,i: toks(ctx[r,i,0],content), lambda r,j: toks(out[r,j],content), "prev-seg content x outcome content")
eval_table(lambda r,i: toks(succ[r,i],content), lambda r,j: toks(out[r,j],content), "succ-seg content x outcome content")
eval_table(lambda r,i: toks(ctx[r,i,1],lambda a: a>=122), lambda r,j: toks(out[r,j],lambda a: a>=122), "last-seg filler x outcome filler")
eval_table(lambda r,i: toks(ctx[r,i,1],content)[-3:], lambda r,j: toks(out[r,j],content)[:3], "last3 content x first3 outcome content")
eval_table(lambda r,i: toks(ctx[r,i,1],content)[:3], lambda r,j: toks(out[r,j],content)[:3], "first3 ctx content x first3 outcome")
