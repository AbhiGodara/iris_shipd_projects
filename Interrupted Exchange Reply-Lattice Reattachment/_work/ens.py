import numpy as np, pandas as pd, itertools, sys
sys.path.insert(0,"_work")
from exp2 import decode, metric, PERMS_NP, PERM_MATS_NP
from sklearn.model_selection import GroupKFold
tr=pd.read_csv("train.csv"); y=pd.read_csv("train_labels.csv").drop(columns=["id"]).values.reshape(-1,4,4).astype(np.float32)
gi=y.argmax(2); gold=np.array([np.where((PERMS_NP==g).all(1))[0][0] for g in gi])
folds=[va for _,va in GroupKFold(4).split(tr,groups=tr.group)]
o1=np.load("_work/oof_e1_full.npy"); o3=np.load("_work/oof_e3_knn.npy")
def z(o): o=o-o.mean(1,keepdims=True); return o/o.std()
def rep(name,o):
    best=None
    for T in (0.25,0.15,0.1,0.05,0.02):
        m,p=decode(o,T); fs=[metric(m[v],y[v]) for v in folds]
        s=(np.mean(fs),T,[round(f,4) for f in fs])
        best=max(best,s) if best else s
    print(f"{name:14s} acc={(o.argmax(1)==gold).mean():.4f} best mean={best[0]:.4f} T={best[1]} folds={best[2]}")
rep("e1",o1); rep("e3",o3)
for w in (0.2,0.3,0.4):
    rep(f"e3+{w}e1", (1-w)*z(o3)+w*z(o1))
# Bayes decision: calibrated posterior (T fitted by NLL), choose per-row sharpening maximizing expected clipped score
