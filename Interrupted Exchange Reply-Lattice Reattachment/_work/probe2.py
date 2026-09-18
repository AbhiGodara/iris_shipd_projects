import os; os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import numpy as np, pandas as pd, itertools, torch, torch.nn.functional as F, sys, time
from sklearn.model_selection import GroupKFold
torch.manual_seed(0)
d = dict(np.load("_work/feats_train.npz"))
tr = pd.read_csv("train.csv"); lab = pd.read_csv("train_labels.csv")
y = lab.drop(columns=["id"]).values.reshape(-1,4,4).astype(np.float32); gi = y.argmax(2)
P = np.array(list(itertools.permutations(range(4)))); PM=np.zeros((24,4,4),np.float32)
for k,p in enumerate(P): PM[k,np.arange(4),p]=1
gold = np.array([np.where((P==g).all(1))[0][0] for g in gi])
dev="cuda"; V=378
def bag(ids):
    t=torch.as_tensor(ids,device=dev); b=torch.zeros(*t.shape[:-1],V,device=dev)
    b.scatter_add_(-1,t,torch.ones_like(t,dtype=torch.float)); b[...,0]=0; return b
bl=bag(d["ctx"][:,:,1]); bp=bag(d["ctx"][:,:,0]); bs=bag(d["succ"]); bo=bag(d["out"])
kind=torch.as_tensor(d["kind"],device=dev).float()
Pt=torch.as_tensor(P,device=dev); goldt=torch.as_tensor(gold,device=dev)
def metric(pred,t):
    corr=((pred-t)**2).mean((1,2)); rm=((pred.sum(2)-1)**2).mean(1); cm=((pred.sum(1)-1)**2).mean(1)
    return np.clip(1-(0.8*corr+0.2*(rm+cm)/18)/0.05,0,1).mean()
def scores(W, idx):
    S = torch.einsum("biv,vu,bju->bij", bl[idx], W[0], bo[idx]) + torch.einsum("biv,vu,bju->bij", bp[idx], W[1], bo[idx]) \
      + torch.einsum("biv,vu,bju->bij", bs[idx], W[2], bo[idx]) + torch.einsum("biv,vu,bju->bij", bl[idx], W[3], bo[idx])*kind[idx].unsqueeze(1)
    return S
def ps(S): return S[:,torch.arange(4,device=dev).unsqueeze(0),Pt].sum(-1)
def fit(tri, lam, iters=300):
    W=torch.zeros(4,V,V,device=dev,requires_grad=True)
    opt=torch.optim.Adam([W],lr=0.02)
    idx=torch.as_tensor(tri,device=dev)
    for it in range(iters):
        loss=F.cross_entropy(ps(scores(W,idx)),goldt[idx])+lam*(W**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), float(loss)
def ev(W, idx, T=1.0):
    with torch.no_grad():
        s=ps(scores(W,torch.as_tensor(idx,device=dev)))
        p=torch.softmax(s/T,-1).cpu().numpy()
    m=np.einsum("bk,kij->bij",p,PM)
    return metric(m,y[idx]), (p.argmax(1)==gold[idx]).mean(), -np.log(p[np.arange(len(idx)),gold[idx]]+1e-12).mean(), s.cpu().numpy()
folds=list(GroupKFold(4).split(tr,groups=tr.group))
tri,vai=folds[0]
for lam in [1e-4,3e-5,1e-5,3e-6]:
    t0=time.time(); W,l=fit(tri,lam)
    a=ev(W,vai); b=ev(W,tri[:2500])
    bestT=max([(ev(W,vai,T)[0],T) for T in (1,0.5,0.3,0.2,0.1)])
    print(f"lam={lam:g} val score={a[0]:.4f} acc={a[1]:.4f} nll={a[2]:.3f} | train score={b[0]:.4f} acc={b[1]:.4f} | bestT {bestT} {time.time()-t0:.0f}s",flush=True)
