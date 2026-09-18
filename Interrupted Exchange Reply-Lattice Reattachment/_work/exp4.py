"""Exp4: two-stage stacking. Stage 1 = bilinear permutation CRF, scores produced out-of-fold by group.
Stage 2 = permutation CRF with MLP over [OOF bilinear components, retrieval, kNN, structural] features."""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import GroupKFold
import solution as S

ITERS1, ITERS2 = 300, 400


class Bil(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(4, S.V, S.V))

    def comps(self, bl, bp, bs, bo, ev):
        W = self.W
        return torch.stack([torch.einsum("biv,vu,bju->bij", bl, W[0], bo),
                            torch.einsum("biv,vu,bju->bij", bp, W[1], bo),
                            torch.einsum("biv,vu,bju->bij", bs, W[2], bo),
                            torch.einsum("biv,vu,bju->bij", bl, W[3], bo) * ev.unsqueeze(1)], -1)


def fit_bil(bags, gold, dev, lam=S.LAMBDA_W):
    S.seed_everything()
    m = Bil().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=S.LR_W)
    perms = torch.as_tensor(S.PERMS_NP, device=dev)
    tb = [torch.as_tensor(a, device=dev) for a in bags]; tg = torch.as_tensor(gold, device=dev)
    for _ in range(ITERS1):
        loss = F.cross_entropy(S.perm_scores(m.comps(*tb).sum(-1), perms), tg) + lam * (m.W ** 2).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


def bil_comps(m, bags, dev):
    with torch.no_grad():
        return m.comps(*[torch.as_tensor(a, device=dev) for a in bags]).cpu().numpy()


def centered(x):
    # row / column relative versions of a [n,4,4] score: margin vs best competitor in row and column
    r = np.sort(x, 2); c = np.sort(x, 1)
    row_best_other = np.where(x >= r[:, :, -1:], r[:, :, -2:-1], r[:, :, -1:])
    col_best_other = np.where(x >= c[:, -1:, :], c[:, -2:-1, :], c[:, -1:, :])
    return np.stack([x - x.mean(2, keepdims=True), x - x.mean(1, keepdims=True),
                     x - row_best_other, x - col_best_other], -1)


def stack_feats(base, comps):
    tot = comps.sum(-1)
    extra = [comps, tot[..., None], centered(tot), centered(base[..., 20])]  # base[...,20] = log1p exact pair count
    return np.concatenate([base] + extra, -1).astype(np.float32)


class Stage2(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def fit_stage2(x, gold, dev, wd=1e-4):
    S.seed_everything()
    m = Stage2(x.shape[-1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=0.005, weight_decay=wd)
    perms = torch.as_tensor(S.PERMS_NP, device=dev)
    tx = torch.as_tensor(x, device=dev); tg = torch.as_tensor(gold, device=dev)
    for _ in range(ITERS2):
        m.train()
        loss = F.cross_entropy(S.perm_scores(m(tx), perms), tg)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


def pred_stage2(m, x, dev):
    m.eval()
    perms = torch.as_tensor(S.PERMS_NP, device=dev)
    with torch.no_grad():
        return S.perm_scores(m(torch.as_tensor(x, device=dev)), perms).cpu().numpy().astype(np.float64)


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    S.seed_everything()
    _, train, y, gi, gold = S.load(".")
    groups = train["group"].astype(str).values
    D = S.Data(S.featurize(train), {})
    folds = list(GroupKFold(4).split(train, groups=groups))
    oof2 = np.zeros((len(train), 24)); oof1 = np.zeros((len(train), 24)); trfit = []
    for f, (tri, vai) in enumerate(folds):
        base_tr = S.labelled_features(D, gi, tri, groups, dev)
        base_va = S.query_features(D, vai, D, gi, tri, dev)
        comps_tr = np.zeros((len(tri), 4, 4, 4), np.float32)
        for a, b in GroupKFold(S.N_INNER).split(tri, groups=groups[tri]):
            comps_tr[b] = bil_comps(fit_bil(D.bags(tri[a]), gold[tri[a]], dev), D.bags(tri[b]), dev)
        full = fit_bil(D.bags(tri), gold[tri], dev)
        comps_va = bil_comps(full, D.bags(vai), dev)
        perms = torch.as_tensor(S.PERMS_NP)
        oof1[vai] = S.perm_scores(torch.as_tensor(comps_va.sum(-1)), perms).numpy()
        x_tr, x_va = S.standardize(stack_feats(base_tr, comps_tr), stack_feats(base_va, comps_va))
        m2 = fit_stage2(x_tr, gold[tri], dev)
        oof2[vai] = pred_stage2(m2, x_va, dev)
        tf = S.metric(S.map_matrix(pred_stage2(m2, x_tr, dev)), y[tri]); trfit.append(tf)
        print(f"fold {f} stage1={S.metric(S.map_matrix(oof1[vai]), y[vai]):.4f} "
              f"stage2={S.metric(S.map_matrix(oof2[vai]), y[vai]):.4f} train_fit={tf:.4f} t={time.time()-t0:.0f}s", flush=True)
    np.save("_work/oof_e4_stage2.npy", oof2); np.save("_work/oof_e4_stage1.npy", oof1)
    res = {}
    for name, o in (("stage1", oof1), ("stage2", oof2)):
        fs = [S.metric(S.map_matrix(o[v]), y[v]) for _, v in folds]
        res[name] = {"folds": [round(x, 4) for x in fs], "mean": round(float(np.mean(fs)), 4),
                     "std": round(float(np.std(fs)), 4), "worst": round(float(np.min(fs)), 4)}
    res["train_fit"] = [round(x, 4) for x in trfit]; res["time_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
    Path("_work/res_e4.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
