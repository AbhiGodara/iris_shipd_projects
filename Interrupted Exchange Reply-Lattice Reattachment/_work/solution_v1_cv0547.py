#!/usr/bin/env python3
"""
Structured four-by-four correspondence: bilinear permutation CRF + group-safe memory features.

Usage:
  python solution.py <public_dir> <submission.csv>
  python solution.py <public_dir> <cv.json> --cv

From-scratch ML only (no pretrained weights, no external data):
- sketch tokens parsed into closed bucket vocabularies (96 word, 24 value, 256 filler buckets)
- context split into previous / last segment; successor segment recovered from the context lattice
- pair score = learned full-rank bucket x bucket compatibility (last, previous, successor segment,
  event-specific) + MLP over structural, overlap, exact-memory and soft-kNN pair features
- memory / kNN features for labelled rows are computed out-of-fold by exchange group, so a row
  never sees its own exchange; test rows use a memory built from the training rows only
- exact 24-permutation distribution trained with permutation cross-entropy
- decoding: the published row score is clip(1 - ||P - Y||_F^2, 0, 1) for doubly-stochastic P and
  distinct permutation matrices are >= 2 apart, so at most one permutation can ever score > 0; the
  Bayes-optimal output is therefore the MAP permutation matrix
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import json
import time
import random
import hashlib
import itertools
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold


SEED = 20260918
L = 16
HIDDEN = 64
DROPOUT = 0.1
LR_W = 0.02
LR_MLP = 0.005
LAMBDA_W = 1e-3
ITERS = 300
N_INNER = 4

PERMS_NP = np.array(list(itertools.permutations(range(4))), dtype=np.int64)
PERM_MATS_NP = np.zeros((24, 4, 4), np.float32)
for _k, _p in enumerate(PERMS_NP):
    PERM_MATS_NP[_k, np.arange(4), _p] = 1.0


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------- parsing
def build_vocab():
    # Closed vocabulary from the published bucket sizes; nothing is fitted on any split.
    toks = ["<PAD>", "<UNK>"]
    toks += [f"w_{i:02x}" for i in range(96)]
    toks += [f"v_{i:02x}" for i in range(24)]
    toks += [f"n_{i:02x}" for i in range(256)]
    return {t: i for i, t in enumerate(toks)}


VOCAB = build_vocab()
V = len(VOCAB)
W_LO, V_LO, N_LO = 2, 98, 122  # token-id ranges: word [2,98), value [98,122), filler [122,V)


def split_segments(s):
    segs, cur = [], []
    for t in str(s).split():
        if t == "segment_end":
            segs.append(cur)
            cur = []
        elif not t.startswith("role_a"):
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs


def enc(toks):
    ids = [VOCAB.get(t, 1) for t in toks[:L]]
    return ids + [0] * (L - len(ids))


def featurize(df):
    n = len(df)
    ctx = np.zeros((n, 4, 2, L), np.int64)    # slot 0 = previous segment (may be empty), slot 1 = last segment
    succ = np.zeros((n, 4, L), np.int64)      # segment following ctx i's last segment in the row's lattice
    rel = np.zeros((n, 4, 4, 4), np.float32)  # structural relations between contexts
    out = np.zeros((n, 4, L), np.int64)
    kind = np.zeros((n, 4), np.int64)
    for r, row in enumerate(df.itertuples(index=False)):
        segs = [split_segments(getattr(row, f"context_{i:02d}")) for i in range(4)]
        for i in range(4):
            ctx[r, i, 1] = enc(segs[i][-1])
            if len(segs[i]) > 1:
                ctx[r, i, 0] = enc(segs[i][-2])
        for i in range(4):
            last_i = tuple(segs[i][-1])
            prev_i = tuple(segs[i][-2]) if len(segs[i]) > 1 else None
            for k in range(4):
                if k == i:
                    continue
                last_k = tuple(segs[k][-1])
                prev_k = tuple(segs[k][-2]) if len(segs[k]) > 1 else None
                if prev_k is not None and last_i == prev_k:
                    rel[r, i, k, 0] = 1
                    succ[r, i] = enc(list(last_k))
                if prev_i is not None and prev_i == last_k:
                    rel[r, i, k, 1] = 1
                if last_i == last_k:
                    rel[r, i, k, 2] = 1
                if prev_i is not None and prev_i == prev_k:
                    rel[r, i, k, 3] = 1
        for j in range(4):
            toks = str(getattr(row, f"outcome_{j:02d}")).split()
            out[r, j] = enc([t for t in toks if not t.startswith("channel_")])
            kind[r, j] = 1 if getattr(row, f"outcome_kind_{j:02d}") == "event" else 0
    return dict(ctx=ctx, succ=succ, rel=rel, out=out, kind=kind)


def np_bag(ids):
    n = ids.reshape(-1, ids.shape[-1])
    b = np.zeros((n.shape[0], V), np.float32)
    np.add.at(b, (np.repeat(np.arange(n.shape[0]), n.shape[1]), n.ravel()), 1.0)
    b[:, 0] = 0
    return b.reshape(*ids.shape[:-1], V)


class Data:
    """Per-split arrays: bags, exact segment keys (string interning shared across splits), static feats."""

    def __init__(self, d, keymap):
        self.d = d
        f = lambda a: keymap.setdefault(("f",) + tuple(int(x) for x in a if x != 0), len(keymap))
        c = lambda a: keymap.setdefault(("c",) + tuple(int(x) for x in a if W_LO <= x < N_LO), len(keymap))
        ctx, out = d["ctx"], d["out"]
        n = len(ctx)
        self.n = n
        self.last_f = np.array([[f(ctx[r, i, 1]) for i in range(4)] for r in range(n)])
        self.prev_f = np.array([[f(ctx[r, i, 0]) for i in range(4)] for r in range(n)])
        self.last_c = np.array([[c(ctx[r, i, 1]) for i in range(4)] for r in range(n)])
        self.out_f = np.array([[f(out[r, j]) for j in range(4)] for r in range(n)])
        self.out_c = np.array([[c(out[r, j]) for j in range(4)] for r in range(n)])
        self.bl = np_bag(ctx[:, :, 1])
        self.bp = np_bag(ctx[:, :, 0])
        self.bs = np_bag(d["succ"])
        self.bo = np_bag(out)
        cmask = np.zeros(V, np.float32)
        cmask[W_LO:N_LO] = 1
        self.bl_c = self.bl * cmask
        self.bo_c = self.bo * cmask
        self.ev = d["kind"].astype(np.float32)
        self.static = static_feats(d, self)

    def bags(self, rows):
        return [self.bl[rows], self.bp[rows], self.bs[rows], self.bo[rows], self.ev[rows]]


def static_feats(d, D):
    ctx, out, succ, kind = d["ctx"], d["out"], d["succ"], d["kind"]
    last, prev = ctx[:, :, 1], ctx[:, :, 0]
    has_prev = (prev[:, :, 0] != 0).astype(np.float32)
    has_succ = (succ[:, :, 0] != 0).astype(np.float32)
    isc = lambda a: (a >= W_LO) & (a < N_LO)
    lc = isc(last).sum(-1) / 12.0
    pc = isc(prev).sum(-1) / 12.0
    oc = isc(out).sum(-1) / 12.0
    ev = kind.astype(np.float32)
    C = lambda x: np.repeat(x[:, :, None], 4, 2)
    O = lambda x: np.repeat(x[:, None, :], 4, 1)
    f = [C(has_prev), C(has_succ), C(lc), C(pc), O(ev), O(oc), C(has_prev) * O(ev), C(has_succ) * O(ev),
         C(lc) * O(oc), C(d["rel"][..., 2].sum(-1)), C(d["rel"][..., 1].sum(-1))]
    bo = (D.bo > 0).astype(np.float32)
    for sb in (D.bl, D.bp, D.bs):
        sb = (sb > 0).astype(np.float32)
        for lo, hi in ((W_LO, V_LO), (V_LO, N_LO), (N_LO, V)):
            f.append(np.log1p(np.einsum("riv,rjv->rij", sb[..., lo:hi], bo[..., lo:hi])))
    return np.stack(f, -1).astype(np.float32)


# ---------------------------------------------------------------- memory features (train rows only)
N_RET = 12
N_KNN = 8


def build_memory(D, gi, rows):
    m = dict(pf=collections.Counter(), pc=collections.Counter(), tr=collections.Counter(),
             nf=collections.Counter(), lc=collections.Counter(), oc=collections.Counter(), fol={}, pre={})
    for r in rows:
        for i in range(4):
            j = gi[r, i]
            m["pf"][(D.last_f[r, i], D.out_f[r, j])] += 1
            m["pc"][(D.last_c[r, i], D.out_c[r, j])] += 1
            m["tr"][(D.prev_f[r, i], D.last_f[r, i], D.out_f[r, j])] += 1
            m["lc"][D.last_f[r, i]] += 1
            m["oc"][D.out_f[r, j]] += 1
            for jj in range(4):
                if jj != j:
                    m["nf"][(D.last_f[r, i], D.out_f[r, jj])] += 1
            a = m["fol"].get(D.last_c[r, i])
            m["fol"][D.last_c[r, i]] = D.bo_c[r, j].copy() if a is None else a + D.bo_c[r, j]
            a = m["pre"].get(D.out_c[r, j])
            m["pre"][D.out_c[r, j]] = D.bl_c[r, i].copy() if a is None else a + D.bl_c[r, i]
    return m


def _cos(a, b):
    return float(a @ b / (np.sqrt(a @ a) * np.sqrt(b @ b) + 1e-6))


def retrieval_feats(Q, m, rows):
    """Exact-key memory statistics for each (context, outcome) pair of query rows."""
    f = np.zeros((len(rows), 4, 4, N_RET), np.float32)
    for n, r in enumerate(rows):
        for i in range(4):
            lf, lcn, pf_ = Q.last_f[r, i], Q.last_c[r, i], Q.prev_f[r, i]
            fol = m["fol"].get(lcn)
            cnt_l = m["lc"][lf]
            for j in range(4):
                of, ocn = Q.out_f[r, j], Q.out_c[r, j]
                pre = m["pre"].get(ocn)
                pfc = m["pf"][(lf, of)]
                cnt_o = m["oc"][of]
                f[n, i, j] = [np.log1p(pfc), np.log1p(m["pc"][(lcn, ocn)]), np.log1p(m["tr"][(pf_, lf, of)]),
                              np.log1p(m["nf"][(lf, of)]), np.log1p(cnt_l), np.log1p(cnt_o),
                              pfc / (cnt_l + 1.0), pfc / (cnt_o + 1.0),
                              0.0 if fol is None else _cos(fol, Q.bo_c[r, j]),
                              0.0 if pre is None else _cos(pre, Q.bl_c[r, i]),
                              float(fol is not None), float(pre is not None)]
    return f


def idf_of(M, rows, dev):
    b = np.concatenate([M.bl[rows].reshape(-1, V), M.bo[rows].reshape(-1, V)]) > 0
    w = np.log(1.0 / (b.mean(0) + 1e-4)).astype(np.float32)
    w[0] = 0
    return torch.as_tensor(w, device=dev)


def knn_feats(Q, q_rows, M, m_rows, gi, idf, dev, chunk=48):
    """Soft similarity of each query (context, outcome) pair to known true / false pairs in memory rows."""
    def vec(b):
        v = torch.as_tensor(b, device=dev) * idf
        return v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    qc, qp, qo = vec(Q.bl[q_rows]), vec(Q.bp[q_rows]), vec(Q.bo[q_rows])
    g = torch.as_tensor(gi[m_rows], device=dev)
    Mc, Mp = vec(M.bl[m_rows]).reshape(-1, V), vec(M.bp[m_rows]).reshape(-1, V)
    Mo_all = vec(M.bo[m_rows])
    Mo = torch.gather(Mo_all, 1, g.unsqueeze(-1).expand(-1, -1, V)).reshape(-1, V)
    negs = [torch.gather(Mo_all, 1, torch.roll(g, s, 1).unsqueeze(-1).expand(-1, -1, V)).reshape(-1, V)
            for s in (1, 2, 3)]
    res = torch.zeros(len(q_rows), 4, 4, N_KNN, device=dev)
    for s in range(0, len(q_rows), chunk):
        c, p, o = qc[s:s + chunk], qp[s:s + chunk], qo[s:s + chunk]
        sc = torch.einsum("biv,nv->bin", c, Mc).clamp(min=0)
        sp = torch.einsum("biv,nv->bin", p, Mp).clamp(min=0)
        so = torch.einsum("bjv,nv->bjn", o, Mo).clamp(min=0)
        pos = sc.unsqueeze(2) * so.unsqueeze(1)
        posp = pos * (0.5 + 0.5 * sp.unsqueeze(2))
        f = [pos.max(-1).values, (pos ** 8).sum(-1).clamp(min=1e-12).log() / 8, posp.max(-1).values,
             (pos > 0.8).float().sum(-1).log1p()]
        nm = torch.zeros_like(pos[..., 0])
        ns = torch.zeros_like(nm)
        for Mn in negs:
            neg = sc.unsqueeze(2) * torch.einsum("bjv,nv->bjn", o, Mn).clamp(min=0).unsqueeze(1)
            nm = torch.maximum(nm, neg.max(-1).values)
            ns = ns + (neg ** 8).sum(-1)
        f += [nm, ns.clamp(min=1e-12).log() / 8,
              sc.max(-1).values.unsqueeze(2).expand(-1, -1, 4), so.max(-1).values.unsqueeze(1).expand(-1, 4, -1)]
        res[s:s + chunk] = torch.stack(f, -1)
    return res.cpu().numpy()


def labelled_features(D, gi, rows, groups, dev):
    """Features for labelled rows; memory parts are out-of-fold by exchange group."""
    ret = np.zeros((len(rows), 4, 4, N_RET), np.float32)
    knn = np.zeros((len(rows), 4, 4, N_KNN), np.float32)
    for a, b in GroupKFold(N_INNER).split(rows, groups=groups[rows]):
        ret[b] = retrieval_feats(D, build_memory(D, gi, rows[a]), rows[b])
        knn[b] = knn_feats(D, rows[b], D, rows[a], gi, idf_of(D, rows[a], dev), dev)
    return np.concatenate([D.static[rows], ret, knn], -1)


def query_features(Q, q_rows, M, gi, m_rows, dev):
    """Features for query rows (validation or test) against a memory of labelled rows."""
    ret = retrieval_feats(Q, build_memory(M, gi, m_rows), q_rows)
    knn = knn_feats(Q, q_rows, M, m_rows, gi, idf_of(M, m_rows, dev), dev)
    return np.concatenate([Q.static[q_rows], ret, knn], -1)


# ---------------------------------------------------------------- model
class PermCRF(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        # bucket x bucket compatibility: last seg, previous seg, successor seg, last seg (event outcomes)
        self.W = nn.Parameter(torch.zeros(4, V, V))
        self.mlp = nn.Sequential(nn.Linear(n_feat, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, bl, bp, bs, bo, ev, x):
        W = self.W
        s = (torch.einsum("biv,vu,bju->bij", bl, W[0], bo)
             + torch.einsum("biv,vu,bju->bij", bp, W[1], bo)
             + torch.einsum("biv,vu,bju->bij", bs, W[2], bo)
             + torch.einsum("biv,vu,bju->bij", bl, W[3], bo) * ev.unsqueeze(1))
        return s + self.mlp(x).squeeze(-1)


def perm_scores(lg, perms):
    return lg[:, torch.arange(4, device=lg.device).unsqueeze(0), perms].sum(-1)


def fit_model(bags, x, gold, dev):
    seed_everything()
    model = PermCRF(x.shape[-1]).to(dev)
    opt = torch.optim.Adam([{"params": [model.W], "lr": LR_W},
                            {"params": list(model.mlp.parameters()), "lr": LR_MLP}])
    perms = torch.as_tensor(PERMS_NP, device=dev)
    tb = [torch.as_tensor(a, device=dev) for a in bags]
    tx = torch.as_tensor(x, device=dev)
    tg = torch.as_tensor(gold, device=dev)
    for _ in range(ITERS):
        model.train()
        loss = F.cross_entropy(perm_scores(model(*tb, tx), perms), tg) + LAMBDA_W * (model.W ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def predict_scores(model, bags, x, dev):
    model.eval()
    perms = torch.as_tensor(PERMS_NP, device=dev)
    with torch.no_grad():
        lg = model(*[torch.as_tensor(a, device=dev) for a in bags], torch.as_tensor(x, device=dev))
        return perm_scores(lg, perms).cpu().numpy().astype(np.float64)


def soft_matrix(scores, T):
    s = scores / T
    s = s - s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    return np.einsum("bk,kij->bij", p, PERM_MATS_NP)


def map_matrix(scores):
    # Bayes-optimal decision under the clipped structured-Brier row score (see module docstring).
    return PERM_MATS_NP[np.argmax(scores, axis=1)].astype(np.float64)


def standardize(x_fit, *others):
    flat = x_fit.reshape(-1, x_fit.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return [(a - mu) / sd for a in (x_fit,) + others]


# ---------------------------------------------------------------- metric
def metric_parts(pred, target):
    corr = ((pred - target) ** 2).mean(axis=(1, 2))
    rowm = ((pred.sum(axis=2) - 1.0) ** 2).mean(axis=1)
    colm = ((pred.sum(axis=1) - 1.0) ** 2).mean(axis=1)
    structure = (rowm + colm) / 18.0
    loss = 0.8 * corr + 0.2 * structure
    return np.clip(1.0 - loss / 0.05, 0.0, 1.0), corr, structure


def metric(pred, target):
    return float(np.mean(metric_parts(pred, target)[0]))


def load(public_dir):
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    labels = pd.read_csv(public_dir / "train_labels.csv")
    if not (train["id"].values == labels["id"].values).all():
        raise RuntimeError("train / label id mismatch")
    y = labels.drop(columns=["id"]).values.reshape(-1, 4, 4).astype(np.float32)
    gi = y.argmax(2)
    gold = np.array([np.where((PERMS_NP == g).all(1))[0][0] for g in gi], dtype=np.int64)
    return public_dir, train, y, gi, gold


# ---------------------------------------------------------------- entry points
def run_cv(public_dir, out_json, n_splits=4):
    t0 = time.time()
    seed_everything()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    public_dir, train, y, gi, gold = load(public_dir)
    groups = train["group"].astype(str).values
    D = Data(featurize(train), {})
    oof = np.zeros((len(train), 24))
    fold_rows, train_fit = [], []
    for fold, (tri, vai) in enumerate(GroupKFold(n_splits).split(train, groups=groups)):
        x_tr = labelled_features(D, gi, tri, groups, dev)
        x_va = query_features(D, vai, D, gi, tri, dev)
        x_tr, x_va = standardize(x_tr, x_va)
        model = fit_model(D.bags(tri), x_tr, gold[tri], dev)
        oof[vai] = predict_scores(model, D.bags(vai), x_va, dev)
        sub = np.arange(min(2500, len(tri)))
        tr_pred = map_matrix(predict_scores(model, [a[sub] for a in D.bags(tri)], x_tr[sub], dev))
        train_fit.append(metric(tr_pred, y[tri][sub]))
        fold_rows.append(vai)
        print(f"[fold {fold}] val={metric(map_matrix(oof[vai]), y[vai]):.5f} "
              f"train_fit={train_fit[-1]:.5f} t={time.time()-t0:.0f}s", flush=True)

    def summarize(pred):
        fs = [metric(pred[v], y[v]) for v in fold_rows]
        _, corr, struct = metric_parts(pred, y)
        return {"fold_scores": [round(float(s), 6) for s in fs], "mean": round(float(np.mean(fs)), 6),
                "std": round(float(np.std(fs)), 6), "worst": round(float(np.min(fs)), 6),
                "brier_component_mean": round(float(corr.mean()), 6),
                "structure_component_mean": round(float(struct.mean()), 6)}

    report = {
        "validation": dict(scheme=f"{n_splits}-fold GroupKFold by exchange group; exact published "
                           "mean_clipped_structured_brier_v1 metric; memory features out-of-fold by group",
                           **summarize(map_matrix(oof))),
        "decoding_comparison": {"map_permutation": summarize(map_matrix(oof))["mean"],
                                **{f"softmax_T={T}": summarize(soft_matrix(oof, T))["mean"] for T in (1.0, 0.5, 0.25, 0.1)}},
        "exact_permutation_accuracy": round(float((oof.argmax(1) == gold).mean()), 6),
        "cell_assignment_accuracy": round(float((PERMS_NP[oof.argmax(1)] == gi).mean()), 6),
        "train_fit_scores": [round(float(s), 6) for s in train_fit],
        "runtime_s": round(time.time() - t0, 1),
    }
    Path(out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_submission(public_dir, out_csv):
    t0 = time.time()
    seed_everything()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    public_dir, train, y, gi, gold = load(public_dir)
    test = pd.read_csv(public_dir / "test.csv")
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    groups = train["group"].astype(str).values
    keymap = {}
    D = Data(featurize(train), keymap)
    Q = Data(featurize(test), keymap)
    rows = np.arange(len(train))
    x_tr = labelled_features(D, gi, rows, groups, dev)
    x_te = query_features(Q, np.arange(len(test)), D, gi, rows, dev)
    x_tr, x_te = standardize(x_tr, x_te)  # statistics from training rows only
    model = fit_model(D.bags(rows), x_tr, gold, dev)
    pred = map_matrix(predict_scores(model, Q.bags(np.arange(len(test))), x_te, dev))

    out = pd.DataFrame({"id": test["id"].values})
    flat = pred.reshape(len(test), 16)
    for k, col in enumerate(sample.columns[1:]):
        out[col] = flat[:, k]
    out = out[sample.columns]

    vals = out.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        raise RuntimeError("non-finite prediction")
    if ((vals < 0) | (vals > 1)).any():
        raise RuntimeError("prediction outside [0,1]")
    if list(out.columns) != list(sample.columns):
        raise RuntimeError("submission columns differ from sample_submission.csv")
    if list(out["id"]) != list(test["id"]) or set(out["id"]) != set(sample["id"]):
        raise RuntimeError("submission ids differ from test.csv / sample_submission.csv")
    if out["id"].duplicated().any() or out.isna().any().any():
        raise RuntimeError("duplicate ids or missing values")

    out.to_csv(out_csv, index=False)
    print(json.dumps({
        "rows": int(len(out)),
        "validation_errors": 0,
        "device": str(dev),
        "runtime_s": round(time.time() - t0, 1),
        "sha256": hashlib.sha256(Path(out_csv).read_bytes()).hexdigest(),
    }, indent=2))


def main():
    args = sys.argv[1:]
    if "--cv" in args:
        args.remove("--cv")
        if len(args) != 2:
            raise SystemExit("Usage: python solution.py <public_dir> <cv.json> --cv")
        run_cv(args[0], args[1])
        return
    if len(args) != 2:
        raise SystemExit("Usage: python solution.py <public_dir> <submission.csv>")
    run_submission(args[0], args[1])


if __name__ == "__main__":
    main()
