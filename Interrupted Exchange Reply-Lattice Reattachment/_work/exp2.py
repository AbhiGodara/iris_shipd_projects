"""Exp2: bilinear permutation CRF + MLP over group-safe retrieval-memory features."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, time, itertools, random, collections
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

SEED = 20260918
PERMS_NP = np.array(list(itertools.permutations(range(4))), dtype=np.int64)
PERM_MATS_NP = np.zeros((24, 4, 4), np.float32)
for _k, _p in enumerate(PERMS_NP):
    PERM_MATS_NP[_k, np.arange(4), _p] = 1.0
V = 378


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def metric(pred, target):
    corr = ((pred - target) ** 2).mean(axis=(1, 2))
    rowm = ((pred.sum(axis=2) - 1.0) ** 2).mean(axis=1)
    colm = ((pred.sum(axis=1) - 1.0) ** 2).mean(axis=1)
    loss = 0.8 * corr + 0.2 * (rowm + colm) / 18.0
    return float(np.mean(np.clip(1.0 - loss / 0.05, 0.0, 1.0)))


def np_bag(ids):
    n = ids.reshape(-1, ids.shape[-1])
    b = np.zeros((n.shape[0], V), np.float32)
    np.add.at(b, (np.repeat(np.arange(n.shape[0]), n.shape[1]), n.ravel()), 1.0)
    b[:, 0] = 0
    return b.reshape(*ids.shape[:-1], V)


class Keys:
    """Integer ids for exact segment keys (full tokens, and content-only)."""
    def __init__(self, d):
        self.map = {}
        f = lambda a: self.map.setdefault(("f",) + tuple(int(x) for x in a if x != 0), len(self.map))
        c = lambda a: self.map.setdefault(("c",) + tuple(int(x) for x in a if 2 <= x < 122), len(self.map))
        ctx, out = d["ctx"], d["out"]
        n = len(ctx)
        self.last_f = np.array([[f(ctx[r, i, 1]) for i in range(4)] for r in range(n)])
        self.prev_f = np.array([[f(ctx[r, i, 0]) for i in range(4)] for r in range(n)])
        self.last_c = np.array([[c(ctx[r, i, 1]) for i in range(4)] for r in range(n)])
        self.out_f = np.array([[f(out[r, j]) for j in range(4)] for r in range(n)])
        self.out_c = np.array([[c(out[r, j]) for j in range(4)] for r in range(n)])


def build_memory(K, gi, rows, bag_last_c, bag_out_c):
    m = dict(pf=collections.Counter(), pc=collections.Counter(), tr=collections.Counter(),
             nf=collections.Counter(), lc=collections.Counter(), oc=collections.Counter(),
             fol={}, pre={})
    for r in rows:
        for i in range(4):
            j = gi[r, i]
            m["pf"][(K.last_f[r, i], K.out_f[r, j])] += 1
            m["pc"][(K.last_c[r, i], K.out_c[r, j])] += 1
            m["tr"][(K.prev_f[r, i], K.last_f[r, i], K.out_f[r, j])] += 1
            m["lc"][K.last_f[r, i]] += 1
            m["oc"][K.out_f[r, j]] += 1
            for jj in range(4):
                if jj != j:
                    m["nf"][(K.last_f[r, i], K.out_f[r, jj])] += 1
            a = m["fol"].get(K.last_c[r, i])
            m["fol"][K.last_c[r, i]] = bag_out_c[r, j].copy() if a is None else a + bag_out_c[r, j]
            a = m["pre"].get(K.out_c[r, j])
            m["pre"][K.out_c[r, j]] = bag_last_c[r, i].copy() if a is None else a + bag_last_c[r, i]
    return m


def cos(a, b):
    return float(a @ b / (np.sqrt(a @ a) * np.sqrt(b @ b) + 1e-6))


N_RET = 12


def retrieval_feats(K, m, rows, bag_last_c, bag_out_c):
    f = np.zeros((len(rows), 4, 4, N_RET), np.float32)
    for n, r in enumerate(rows):
        for i in range(4):
            lf, lcn, pf_ = K.last_f[r, i], K.last_c[r, i], K.prev_f[r, i]
            fol = m["fol"].get(lcn)
            cnt_l = m["lc"][lf]
            for j in range(4):
                of, ocn = K.out_f[r, j], K.out_c[r, j]
                pre = m["pre"].get(ocn)
                pfc = m["pf"][(lf, of)]; pcc = m["pc"][(lcn, ocn)]; trc = m["tr"][(pf_, lf, of)]
                nfc = m["nf"][(lf, of)]; cnt_o = m["oc"][of]
                f[n, i, j] = [np.log1p(pfc), np.log1p(pcc), np.log1p(trc), np.log1p(nfc),
                              np.log1p(cnt_l), np.log1p(cnt_o),
                              pfc / (cnt_l + 1.0), pfc / (cnt_o + 1.0),
                              0.0 if fol is None else cos(fol, bag_out_c[r, j]),
                              0.0 if pre is None else cos(pre, bag_last_c[r, i]),
                              float(fol is not None), float(pre is not None)]
    return f


def oof_retrieval(K, gi, rows, groups, bag_last_c, bag_out_c, n_inner=4):
    """Features for labeled rows, each computed from a memory that excludes its own group-fold."""
    f = np.zeros((len(rows), 4, 4, N_RET), np.float32)
    pos = {r: n for n, r in enumerate(rows)}
    for a, b in GroupKFold(n_inner).split(rows, groups=groups[rows]):
        m = build_memory(K, gi, rows[a], bag_last_c, bag_out_c)
        f[b] = retrieval_feats(K, m, rows[b], bag_last_c, bag_out_c)
    return f


def static_feats(d):
    ctx, out, succ, kind = d["ctx"], d["out"], d["succ"], d["kind"]
    n = len(ctx)
    last, prev = ctx[:, :, 1], ctx[:, :, 0]
    has_prev = (prev[:, :, 0] != 0).astype(np.float32)
    has_succ = (succ[:, :, 0] != 0).astype(np.float32)
    isc = lambda a: ((a >= 2) & (a < 122))
    lc = isc(last).sum(-1) / 12.0; pc = isc(prev).sum(-1) / 12.0; oc = isc(out).sum(-1) / 12.0
    ev = kind.astype(np.float32)
    f = []
    C = lambda x: np.repeat(x[:, :, None], 4, 2)
    O = lambda x: np.repeat(x[:, None, :], 4, 1)
    f += [C(has_prev), C(has_succ), C(lc), C(pc), O(ev), O(oc), C(has_prev) * O(ev), C(has_succ) * O(ev),
          C(lc) * O(oc), C(d["rel"][..., 2].sum(-1)), C(d["rel"][..., 1].sum(-1))]
    # exact overlap by type and segment
    bl, bp, bs, bo = [np_bag(x) > 0 for x in (last, prev, succ, out)]
    for sb in (bl, bp, bs):
        for lo, hi in ((2, 98), (98, 122), (122, V)):
            inter = np.einsum("riv,rjv->rij", sb[..., lo:hi].astype(np.float32), bo[..., lo:hi].astype(np.float32))
            f.append(np.log1p(inter))
    return np.stack(f, -1).astype(np.float32)


class Hybrid(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.1, use_bil=True, use_mlp=True):
        super().__init__()
        self.use_bil, self.use_mlp = use_bil, use_mlp
        self.W = nn.Parameter(torch.zeros(4, V, V))
        self.mlp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, bl, bp, bs, bo, ev, x):
        s = 0.0
        if self.use_bil:
            W = self.W
            s = (torch.einsum("biv,vu,bju->bij", bl, W[0], bo) + torch.einsum("biv,vu,bju->bij", bp, W[1], bo)
                 + torch.einsum("biv,vu,bju->bij", bs, W[2], bo)
                 + torch.einsum("biv,vu,bju->bij", bl, W[3], bo) * ev.unsqueeze(1))
        if self.use_mlp:
            s = s + self.mlp(x).squeeze(-1)
        return s


def perm_scores(lg, perms):
    return lg[:, torch.arange(4, device=lg.device).unsqueeze(0), perms].sum(-1)


def decode(scores, T=1.0):
    s = scores / T
    s = s - s.max(1, keepdims=True)
    p = np.exp(s); p /= p.sum(1, keepdims=True)
    return np.einsum("bk,kij->bij", p, PERM_MATS_NP), p


def fit_predict(cfg, T_bags, T_x, T_gold, P_bags, P_x, dev, eval_cb=None):
    seed_everything(cfg["seed"])
    model = Hybrid(T_x.shape[-1], cfg["hidden"], cfg["dropout"], cfg["bil"], cfg["mlp"]).to(dev)
    opt = torch.optim.Adam([{"params": [model.W], "lr": cfg["lr_w"]},
                            {"params": list(model.mlp.parameters()), "lr": cfg["lr_m"], "weight_decay": 0.0}])
    perms = torch.as_tensor(PERMS_NP, device=dev)
    tb = [torch.as_tensor(a, device=dev) for a in T_bags]
    tx = torch.as_tensor(T_x, device=dev); tg = torch.as_tensor(T_gold, device=dev)
    for it in range(cfg["iters"]):
        model.train()
        lg = model(*tb, tx)
        loss = F.cross_entropy(perm_scores(lg, perms), tg) + cfg["lam"] * (model.W ** 2).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if eval_cb is not None and (it + 1) % 50 == 0:
            eval_cb(it + 1, model)
    return model


def predict(model, bags, x, dev):
    model.eval()
    perms = torch.as_tensor(PERMS_NP, device=dev)
    with torch.no_grad():
        lg = model(*[torch.as_tensor(a, device=dev) for a in bags], torch.as_tensor(x, device=dev))
        return perm_scores(lg, perms).cpu().numpy().astype(np.float64)


def main():
    cfg = dict(hidden=64, dropout=0.1, bil=True, mlp=True, ret=True, lr_w=0.02, lr_m=0.005,
               lam=1e-4, iters=300, seed=SEED)
    for a in sys.argv[2:]:
        k, v = a.split("=")
        cfg[k] = (v in ("1", "True", "true")) if isinstance(cfg[k], bool) else type(cfg[k])(v)
    tag = sys.argv[1]
    dev = torch.device("cuda")
    t0 = time.time()
    tr = pd.read_csv("train.csv"); lab = pd.read_csv("train_labels.csv")
    y = lab.drop(columns=["id"]).values.reshape(-1, 4, 4).astype(np.float32); gi = y.argmax(2)
    gold = np.array([np.where((PERMS_NP == g).all(1))[0][0] for g in gi])
    d = dict(np.load("_work/feats_train.npz"))
    K = Keys(d)
    groups = tr.group.astype(str).values
    bl = np_bag(d["ctx"][:, :, 1]); bp = np_bag(d["ctx"][:, :, 0]); bs = np_bag(d["succ"]); bo = np_bag(d["out"])
    cmask = np.zeros(V, np.float32); cmask[2:122] = 1
    ev = d["kind"].astype(np.float32)
    sf = static_feats(d)
    print(f"prep {time.time()-t0:.1f}s static feats {sf.shape}", flush=True)
    folds = list(GroupKFold(4).split(tr, groups=groups))
    oof = np.zeros((len(tr), 24)); trfit = []
    curves = collections.defaultdict(list)
    for f, (tri, vai) in enumerate(folds):
        if cfg["ret"]:
            r_tr = oof_retrieval(K, gi, tri, groups, bl * cmask, bo * cmask)
            mem = build_memory(K, gi, tri, bl * cmask, bo * cmask)
            r_va = retrieval_feats(K, mem, vai, bl * cmask, bo * cmask)
            x_tr = np.concatenate([sf[tri], r_tr], -1); x_va = np.concatenate([sf[vai], r_va], -1)
        else:
            x_tr, x_va = sf[tri], sf[vai]
        mu, sd = x_tr.reshape(-1, x_tr.shape[-1]).mean(0), x_tr.reshape(-1, x_tr.shape[-1]).std(0) + 1e-6
        x_tr = (x_tr - mu) / sd; x_va = (x_va - mu) / sd
        Tb = [bl[tri], bp[tri], bs[tri], bo[tri], ev[tri]]; Vb = [bl[vai], bp[vai], bs[vai], bo[vai], ev[vai]]

        def cb(it, model):
            sc = predict(model, Vb, x_va, dev)
            m, p = decode(sc, 0.25)
            curves[it].append(((p.argmax(1) == gold[vai]).mean(), metric(m, y[vai])))
        model = fit_predict(cfg, Tb, x_tr, gold[tri], Vb, x_va, dev, eval_cb=cb)
        oof[vai] = predict(model, Vb, x_va, dev)
        sub = np.arange(min(2500, len(tri)))
        m_tr, _ = decode(predict(model, [a[sub] for a in Tb], x_tr[sub], dev), 0.25)
        trfit.append(metric(m_tr, y[tri][sub]))
        print(f"fold {f} done {time.time()-t0:.0f}s acc={(oof[vai].argmax(1)==gold[vai]).mean():.4f}", flush=True)
    np.save(f"_work/oof_{tag}.npy", oof)
    res = {"tag": tag, "cfg": cfg, "time_s": round(time.time() - t0, 1), "train_fit_T0.25": [round(x, 4) for x in trfit],
           "curve_iter_acc_score": {int(k): [round(float(np.mean([a for a, _ in v])), 4), round(float(np.mean([s for _, s in v])), 4)] for k, v in curves.items()}}
    for T in (1.0, 0.5, 0.35, 0.25, 0.15):
        m, p = decode(oof, T)
        fs = [metric(m[va], y[va]) for _, va in folds]
        res[f"T={T}"] = {"folds": [round(x, 4) for x in fs], "mean": round(float(np.mean(fs)), 4),
                          "std": round(float(np.std(fs)), 4), "worst": round(float(np.min(fs)), 4)}
    _, p = decode(oof)
    res["perm_acc"] = round(float((p.argmax(1) == gold).mean()), 4)
    res["cell_acc"] = round(float((PERMS_NP[p.argmax(1)] == gi).mean()), 4)
    res["nll_T1"] = round(float(-np.mean(np.log(p[np.arange(len(p)), gold] + 1e-12))), 4)
    print(json.dumps(res, indent=1), flush=True)
    Path(f"_work/res_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
