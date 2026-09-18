"""Exp3: exp2 + soft kNN pair-retrieval features (similarity to known true / false pairs in other groups)."""
import sys, time, json, collections
import numpy as np, torch
sys.path.insert(0, "_work")
import exp2
from exp2 import *

N_KNN = 8


def knn_feats(q_rows, m_rows, gi, bl, bp, bo, idf, dev, chunk=48):
    """For query rows, similarity of each (ctx_i, out_j) to true and false pairs of memory rows."""
    def vec(b):
        v = torch.as_tensor(b, device=dev) * idf
        return v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    qc, qp, qo = vec(bl[q_rows]), vec(bp[q_rows]), vec(bo[q_rows])            # [Q,4,V]
    mr = torch.as_tensor(m_rows, device=dev)
    g = torch.as_tensor(gi[m_rows], device=dev)                                  # [M,4]
    Mc, Mp = vec(bl[m_rows]).reshape(-1, V), vec(bp[m_rows]).reshape(-1, V)      # true ctx i
    Mo_all = vec(bo[m_rows])                                                     # [M,4,V]
    Mo = torch.gather(Mo_all, 1, g.unsqueeze(-1).expand(-1, -1, V)).reshape(-1, V)  # true outcome for ctx i
    # negatives: ctx i with the outcome of its row-neighbour (i+1 mod 4 in gold order) -> 3 negatives per ctx
    negs = [torch.gather(Mo_all, 1, torch.roll(g, s, 1).unsqueeze(-1).expand(-1, -1, V)).reshape(-1, V) for s in (1, 2, 3)]
    out = torch.zeros(len(q_rows), 4, 4, N_KNN, device=dev)
    for s in range(0, len(q_rows), chunk):
        c, p, o = qc[s:s+chunk], qp[s:s+chunk], qo[s:s+chunk]
        sc = torch.einsum("biv,nv->bin", c, Mc).clamp(min=0)   # [b,4,N]
        sp = torch.einsum("biv,nv->bin", p, Mp).clamp(min=0)
        so = torch.einsum("bjv,nv->bjn", o, Mo).clamp(min=0)
        pos = sc.unsqueeze(2) * so.unsqueeze(1)                 # [b,4,4,N]
        posp = pos * (0.5 + 0.5 * sp.unsqueeze(2))
        f = [pos.max(-1).values, (pos ** 8).sum(-1).clamp(min=1e-12).log() / 8, posp.max(-1).values,
             (pos > 0.8).float().sum(-1).log1p()]
        nm = torch.zeros_like(pos[..., 0]); ns = torch.zeros_like(nm)
        for Mn in negs:
            so_n = torch.einsum("bjv,nv->bjn", o, Mn).clamp(min=0)
            neg = sc.unsqueeze(2) * so_n.unsqueeze(1)
            nm = torch.maximum(nm, neg.max(-1).values); ns = ns + (neg ** 8).sum(-1)
        f += [nm, ns.clamp(min=1e-12).log() / 8]
        # outcome-side only: best similarity of out_j to any true outcome whose context resembles ctx_i (already pos);
        # ctx-side coverage: how well ctx_i / out_j are covered by memory at all
        f += [sc.max(-1).values.unsqueeze(2).expand(-1, -1, 4), so.max(-1).values.unsqueeze(1).expand(-1, 4, -1)]
        out[s:s+chunk] = torch.stack(f, -1)
    return out.cpu().numpy()


def idf_of(rows, bl, bo, dev):
    b = np.concatenate([bl[rows].reshape(-1, V), bo[rows].reshape(-1, V)]) > 0
    df_ = b.mean(0)
    w = np.log(1.0 / (df_ + 1e-4)).astype(np.float32); w[0] = 0
    return torch.as_tensor(w, device=dev)


def knn_oof(rows, groups, gi, bl, bp, bo, dev, n_inner=4):
    f = np.zeros((len(rows), 4, 4, N_KNN), np.float32)
    for a, b in GroupKFold(n_inner).split(rows, groups=groups[rows]):
        idf = idf_of(rows[a], bl, bo, dev)
        f[b] = knn_feats(rows[b], rows[a], gi, bl, bp, bo, idf, dev)
    return f


def main():
    cfg = dict(hidden=64, dropout=0.1, bil=True, mlp=True, ret=True, lr_w=0.02, lr_m=0.005,
               lam=1e-4, iters=300, seed=SEED)
    for a in sys.argv[2:]:
        k, v = a.split("=")
        cfg[k] = (v in ("1", "True", "true")) if isinstance(cfg[k], bool) else type(cfg[k])(v)
    tag = sys.argv[1]
    dev = torch.device("cuda"); t0 = time.time()
    tr = pd.read_csv("train.csv"); lab = pd.read_csv("train_labels.csv")
    y = lab.drop(columns=["id"]).values.reshape(-1, 4, 4).astype(np.float32); gi = y.argmax(2)
    gold = np.array([np.where((PERMS_NP == g).all(1))[0][0] for g in gi])
    d = dict(np.load("_work/feats_train.npz")); K = Keys(d)
    groups = tr.group.astype(str).values
    bl = np_bag(d["ctx"][:, :, 1]); bp = np_bag(d["ctx"][:, :, 0]); bs = np_bag(d["succ"]); bo = np_bag(d["out"])
    cmask = np.zeros(V, np.float32); cmask[2:122] = 1
    ev = d["kind"].astype(np.float32); sf = static_feats(d)
    folds = list(GroupKFold(4).split(tr, groups=groups))
    oof = np.zeros((len(tr), 24)); trfit = []
    for f, (tri, vai) in enumerate(folds):
        parts_tr, parts_va = [sf[tri]], [sf[vai]]
        if cfg["ret"]:
            parts_tr.append(oof_retrieval(K, gi, tri, groups, bl * cmask, bo * cmask))
            mem = build_memory(K, gi, tri, bl * cmask, bo * cmask)
            parts_va.append(retrieval_feats(K, mem, vai, bl * cmask, bo * cmask))
        parts_tr.append(knn_oof(tri, groups, gi, bl, bp, bo, dev))
        parts_va.append(knn_feats(vai, tri, gi, bl, bp, bo, idf_of(tri, bl, bo, dev), dev))
        x_tr = np.concatenate(parts_tr, -1); x_va = np.concatenate(parts_va, -1)
        mu, sd = x_tr.reshape(-1, x_tr.shape[-1]).mean(0), x_tr.reshape(-1, x_tr.shape[-1]).std(0) + 1e-6
        x_tr = (x_tr - mu) / sd; x_va = (x_va - mu) / sd
        Tb = [bl[tri], bp[tri], bs[tri], bo[tri], ev[tri]]; Vb = [bl[vai], bp[vai], bs[vai], bo[vai], ev[vai]]
        model = fit_predict(cfg, Tb, x_tr, gold[tri], Vb, x_va, dev)
        oof[vai] = predict(model, Vb, x_va, dev)
        sub = np.arange(2500)
        m_tr, _ = decode(predict(model, [a[sub] for a in Tb], x_tr[sub], dev), 0.25)
        trfit.append(metric(m_tr, y[tri][sub]))
        print(f"fold {f} done {time.time()-t0:.0f}s acc={(oof[vai].argmax(1)==gold[vai]).mean():.4f}", flush=True)
    np.save(f"_work/oof_{tag}.npy", oof)
    res = {"tag": tag, "cfg": cfg, "time_s": round(time.time() - t0, 1), "train_fit_T0.25": [round(x, 4) for x in trfit]}
    for T in (1.0, 0.5, 0.35, 0.25, 0.15, 0.1):
        m, p = decode(oof, T)
        fs = [metric(m[va], y[va]) for _, va in folds]
        res[f"T={T}"] = {"folds": [round(x, 4) for x in fs], "mean": round(float(np.mean(fs)), 4),
                          "std": round(float(np.std(fs)), 4), "worst": round(float(np.min(fs)), 4)}
    _, p = decode(oof)
    res["perm_acc"] = round(float((p.argmax(1) == gold).mean()), 4)
    res["cell_acc"] = round(float((PERMS_NP[p.argmax(1)] == gi).mean()), 4)
    res["nll_T1"] = round(float(-np.mean(np.log(p[np.arange(len(p)), gold] + 1e-12))), 4)
    print(json.dumps({k: v for k, v in res.items() if k != "cfg"}), flush=True)
    Path(f"_work/res_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
