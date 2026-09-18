"""Experiment harness: fast precomputed tensors, grouped 4-fold CV, OOF perm posteriors."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, time, itertools, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

SEED = 20260918
PERMS_NP = np.array(list(itertools.permutations(range(4))), dtype=np.int64)  # [24,4]
PERM_MATS_NP = np.zeros((24, 4, 4), np.float32)
for k, p in enumerate(PERMS_NP):
    PERM_MATS_NP[k, np.arange(4), p] = 1.0
L = 16


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_vocab():
    # Closed, data-independent vocabulary from the manifest's bucket sizes.
    v = ["<PAD>", "<UNK>"]
    v += [f"w_{i:02x}" for i in range(96)]
    v += [f"v_{i:02x}" for i in range(24)]
    v += [f"n_{i:02x}" for i in range(256)]
    return {t: i for i, t in enumerate(v)}


def split_segments(s):
    segs, cur = [], []
    for t in str(s).split():
        if t == "segment_end":
            segs.append(cur); cur = []
        elif not t.startswith("role_a"):
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs


def enc(toks, vocab):
    ids = [vocab.get(t, 1) for t in toks[:L]]
    return ids + [0] * (L - len(ids))


def featurize(df, vocab):
    n = len(df)
    ctx = np.zeros((n, 4, 2, L), np.int64)     # slot0 = previous segment (may be empty), slot1 = last segment
    succ = np.zeros((n, 4, L), np.int64)       # segment that follows ctx i's last segment (from another ctx)
    rel = np.zeros((n, 4, 4, 4), np.float32)   # structural relations between contexts i,k
    out = np.zeros((n, 4, L), np.int64)
    kind = np.zeros((n, 4), np.int64)
    for r, row in enumerate(df.itertuples(index=False)):
        segs = [split_segments(getattr(row, f"context_{i:02d}")) for i in range(4)]
        for i in range(4):
            s = segs[i]
            ctx[r, i, 1] = enc(s[-1], vocab)
            if len(s) > 1:
                ctx[r, i, 0] = enc(s[-2], vocab)
        for i in range(4):
            last_i = tuple(segs[i][-1]); prev_i = tuple(segs[i][-2]) if len(segs[i]) > 1 else None
            for k in range(4):
                if k == i:
                    continue
                last_k = tuple(segs[k][-1]); prev_k = tuple(segs[k][-2]) if len(segs[k]) > 1 else None
                if prev_k is not None and last_i == prev_k:
                    rel[r, i, k, 0] = 1; succ[r, i] = enc(list(last_k), vocab)
                if prev_i is not None and prev_i == last_k:
                    rel[r, i, k, 1] = 1
                if last_i == last_k:
                    rel[r, i, k, 2] = 1
                if prev_i is not None and prev_i == prev_k:
                    rel[r, i, k, 3] = 1
        for j in range(4):
            toks = str(getattr(row, f"outcome_{j:02d}")).split()
            out[r, j] = enc([t for t in toks if not t.startswith("channel_")], vocab)
            kind[r, j] = 1 if getattr(row, f"outcome_kind_{j:02d}") == "event" else 0
    return dict(ctx=ctx, succ=succ, rel=rel, out=out, kind=kind)


def metric_rows(pred, target):
    corr = ((pred - target) ** 2).mean(axis=(1, 2))
    rowm = ((pred.sum(axis=2) - 1.0) ** 2).mean(axis=1)
    colm = ((pred.sum(axis=1) - 1.0) ** 2).mean(axis=1)
    loss = 0.8 * corr + 0.2 * (rowm + colm) / 18.0
    return np.clip(1.0 - loss / 0.05, 0.0, 1.0)


def metric(pred, target):
    return float(np.mean(metric_rows(pred, target)))


# ---------------------------------------------------------------- model
class SegEncoder(nn.Module):
    """Permutation-invariant set encoder: multi-head attention pooling + mean pooling of token embeddings."""
    def __init__(self, emb, d, heads=4, dropout=0.1):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.gate = nn.Linear(d, heads)
        self.proj = nn.Sequential(nn.Linear(d * (heads + 1), d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))

    def forward(self, ids):
        sh = ids.shape[:-1]
        ids = ids.reshape(-1, ids.size(-1))
        h = self.emb(ids)
        m = ids.ne(0)
        lg = self.gate(h).masked_fill(~m.unsqueeze(-1), -1e4)
        w = torch.softmax(lg, dim=1)
        att = torch.einsum("blh,bld->bhd", w, h).reshape(h.size(0), -1)
        cnt = m.sum(1, keepdim=True).clamp(min=1).float()
        mean = (h * m.unsqueeze(-1)).sum(1) / cnt
        z = self.proj(torch.cat([att, mean], -1))
        empty = (~m.any(1)).float().unsqueeze(-1)
        z = z * (1 - empty)
        return z.reshape(*sh, -1)


class PairNet(nn.Module):
    def __init__(self, V, d=64, hidden=128, dropout=0.1, use_bilinear=True, use_axial=True, use_struct=True):
        super().__init__()
        self.V = V
        self.use_bilinear, self.use_axial, self.use_struct = use_bilinear, use_axial, use_struct
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.enc_c = SegEncoder(self.emb, d, dropout=dropout)
        self.enc_o = SegEncoder(self.emb, d, dropout=dropout)
        self.kind = nn.Embedding(2, 16)
        nb = 0
        if use_bilinear:
            # full bag-of-bucket compatibility matrices: last seg, prev seg, successor seg; kind-specific for last seg
            self.B = nn.Parameter(torch.zeros(4, V, V))
            nb = 4
        n_ov = 3 * 3 * 2  # (last, prev, succ) x (w, v, n) x (count, frac)
        n_struct = 8 if use_struct else 0
        in_dim = d * 3 + d * 2 + 16 + nb + n_ov + n_struct + (d if use_struct else 0)
        self.pair = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        if use_axial:
            self.row_att = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
            self.col_att = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
            self.n1 = nn.LayerNorm(hidden); self.n2 = nn.LayerNorm(hidden)
            self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
            self.n3 = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        # type masks for overlap features
        tmask = torch.zeros(3, V)
        tmask[0, 2:98] = 1; tmask[1, 98:122] = 1; tmask[2, 122:] = 1
        self.register_buffer("tmask", tmask)

    def bag(self, ids):
        b = torch.zeros(*ids.shape[:-1], self.V, device=ids.device)
        b.scatter_add_(-1, ids, torch.ones_like(ids, dtype=torch.float))
        b[..., 0] = 0
        return b

    def forward(self, ctx, succ, rel, out, kind):
        B_ = ctx.size(0)
        hc = self.enc_c(ctx)            # [B,4,2,d]
        hs = self.enc_c(succ)           # [B,4,d]
        ho = self.enc_o(out)            # [B,4,d]
        hprev, hlast = hc[:, :, 0], hc[:, :, 1]
        cvec = torch.cat([hprev, hlast, hprev * hlast], -1)  # [B,4,3d]
        C = cvec.unsqueeze(2).expand(-1, 4, 4, -1)
        O = ho.unsqueeze(1).expand(-1, 4, 4, -1)
        Hl = hlast.unsqueeze(2).expand(-1, 4, 4, -1)
        K = self.kind(kind).unsqueeze(1).expand(-1, 4, 4, -1)
        feats = [C, O, Hl * O, K]
        bc = self.bag(ctx)              # [B,4,2,V]
        bs = self.bag(succ)             # [B,4,V]
        bo = self.bag(out)              # [B,4,V]
        seg_bags = [bc[:, :, 1], bc[:, :, 0], bs]
        if self.use_bilinear:
            bil = []
            ev = kind.float().unsqueeze(1)  # [B,1,4]
            s_last = torch.einsum("biv,vu,bju->bij", seg_bags[0], self.B[0], bo)
            s_last_ev = torch.einsum("biv,vu,bju->bij", seg_bags[0], self.B[3], bo) * ev
            s_prev = torch.einsum("biv,vu,bju->bij", seg_bags[1], self.B[1], bo)
            s_succ = torch.einsum("biv,vu,bju->bij", seg_bags[2], self.B[2], bo)
            bil = torch.stack([s_last, s_prev, s_succ, s_last_ev], -1)
            feats.append(bil)
        ov = []
        bop = (bo > 0).float()
        for sb in seg_bags:
            sbp = (sb > 0).float()
            for t in range(3):
                inter = torch.einsum("biv,bjv->bij", sbp * self.tmask[t], bop)
                denom = (bop * self.tmask[t]).sum(-1).clamp(min=1).unsqueeze(1)
                ov += [inter, inter / denom]
        feats.append(torch.stack(ov, -1))
        if self.use_struct:
            has_prev = ctx[:, :, 0, 0].ne(0).float()
            has_succ = succ[:, :, 0].ne(0).float()
            st = torch.stack([has_prev, has_succ, rel[..., 0].sum(-1), rel[..., 1].sum(-1),
                              rel[..., 2].sum(-1), rel[..., 3].sum(-1),
                              (ctx[:, :, 1].ne(0).sum(-1).float() / L), (ctx[:, :, 0].ne(0).sum(-1).float() / L)], -1)
            feats.append(st.unsqueeze(2).expand(-1, 4, 4, -1))
            feats.append((hs.unsqueeze(2) * ho.unsqueeze(1)))
        z = self.pair(torch.cat(feats, -1))  # [B,4,4,H]
        if self.use_axial:
            H = z.size(-1)
            r = z.reshape(B_ * 4, 4, H)
            r = self.n1(r + self.row_att(r, r, r, need_weights=False)[0])
            c = r.reshape(B_, 4, 4, H).transpose(1, 2).reshape(B_ * 4, 4, H)
            c = self.n2(c + self.col_att(c, c, c, need_weights=False)[0])
            z = c.reshape(B_, 4, 4, H).transpose(1, 2)
            z = self.n3(z + self.ff(z))
        return self.head(z).squeeze(-1)  # [B,4,4]


def perm_scores(logits, perms):
    # logits [B,4,4], perms [24,4] -> [B,24]
    return logits[:, torch.arange(4, device=logits.device).unsqueeze(0), perms].sum(-1)


def to_dev(d, idx, dev):
    return [torch.as_tensor(d[k][idx]).to(dev) for k in ("ctx", "succ", "rel", "out", "kind")]


def predict_scores(model, data, idx, dev, perms, bs=512):
    model.eval()
    res = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            lg = model(*to_dev(data, b, dev))
            res.append(perm_scores(lg, perms).cpu().numpy())
    return np.concatenate(res).astype(np.float64)


def decode(scores, T=1.0):
    s = scores / T
    s = s - s.max(1, keepdims=True)
    p = np.exp(s); p /= p.sum(1, keepdims=True)
    return np.einsum("bk,kij->bij", p, PERM_MATS_NP), p


def train_fold(data, y, gold, tr, va, dev, cfg, log=print):
    seed_everything(cfg.get("seed", SEED))
    model = PairNet(cfg["V"], d=cfg["d"], hidden=cfg["hidden"], dropout=cfg["dropout"],
                    use_bilinear=cfg["bilinear"], use_axial=cfg["axial"], use_struct=cfg["struct"]).to(dev)
    bil_params = [p for n, p in model.named_parameters() if n == "B"]
    other = [p for n, p in model.named_parameters() if n != "B"]
    groups = [{"params": other, "weight_decay": cfg["wd"]}]
    if bil_params:
        groups.append({"params": bil_params, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    perms = torch.as_tensor(PERMS_NP, device=dev)
    epochs = cfg["epochs"]
    steps_per = (len(tr) + cfg["bs"] - 1) // cfg["bs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * steps_per, pct_start=0.15)
    rng = np.random.RandomState(cfg.get("seed", SEED))
    gold_t = torch.as_tensor(gold)
    best = (-1, None, 0)
    hist = []
    for ep in range(epochs):
        model.train()
        order = tr[rng.permutation(len(tr))]
        tl = 0.0
        for s in range(0, len(order), cfg["bs"]):
            b = order[s:s + cfg["bs"]]
            lg = model(*to_dev(data, b, dev))
            ps = perm_scores(lg, perms)
            loss = F.cross_entropy(ps, gold_t[b].to(dev))
            if cfg["l2B"] > 0 and bil_params:
                loss = loss + cfg["l2B"] * (bil_params[0] ** 2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tl += float(loss) * len(b)
        if va is not None:
            sc = predict_scores(model, data, va, dev, perms)
            m, _ = decode(sc)
            vs = metric(m, y[va])
            vnll = float(-np.mean(np.log(np.clip(_[np.arange(len(va)), gold[va]], 1e-12, 1))))
            hist.append((ep, tl / len(tr), vnll, vs))
            log(f"  ep{ep:02d} train_ce={tl/len(tr):.4f} val_nll={vnll:.4f} val_score(T=1)={vs:.4f}")
            if -vnll > best[0] if best[1] is not None else True:
                pass
    return model, hist


def main():
    cfg = dict(d=64, hidden=128, dropout=0.1, bilinear=True, axial=True, struct=True,
               lr=2e-3, wd=1e-4, l2B=1e-5, epochs=20, bs=128, seed=SEED)
    for a in sys.argv[2:]:
        k, v = a.split("=")
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], bool) else v in ("1", "True", "true")
    tag = sys.argv[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    train = pd.read_csv("train.csv"); lab = pd.read_csv("train_labels.csv")
    assert (train.id.values == lab.id.values).all()
    y = lab.drop(columns=["id"]).values.reshape(-1, 4, 4).astype(np.float32)
    gi = y.argmax(2)
    gold = np.array([np.where((PERMS_NP == g).all(1))[0][0] for g in gi])
    vocab = build_vocab(); cfg["V"] = len(vocab)
    cache = Path("_work/feats_train.npz")
    if cache.exists():
        data = dict(np.load(cache))
    else:
        data = featurize(train, vocab); np.savez_compressed(cache, **data)
    print(f"featurized {time.time()-t0:.1f}s", flush=True)
    groups = train.group.astype(str).values
    oof = np.zeros((len(train), 24))
    tr_scores = []
    for f, (tr, va) in enumerate(GroupKFold(4).split(train, groups=groups)):
        print(f"=== fold {f}", flush=True)
        model, hist = train_fold(data, y, gold, tr, va, dev, cfg)
        perms = torch.as_tensor(PERMS_NP, device=dev)
        oof[va] = predict_scores(model, data, va, dev, perms)
        sub = tr[:2500]
        m_tr, _ = decode(predict_scores(model, data, sub, dev, perms))
        tr_scores.append(metric(m_tr, y[sub]))
    np.save(f"_work/oof_{tag}.npy", oof)
    folds = [va for _, va in GroupKFold(4).split(train, groups=groups)]
    res = {"tag": tag, "cfg": {k: v for k, v in cfg.items()}, "time_s": round(time.time() - t0, 1),
           "train_fit_T1": [round(x, 4) for x in tr_scores]}
    for T in (1.0, 0.7, 0.5, 0.35, 0.25):
        m, p = decode(oof, T)
        fs = [metric(m[va], y[va]) for va in folds]
        res[f"T={T}"] = {"folds": [round(x, 4) for x in fs], "mean": round(float(np.mean(fs)), 4),
                          "std": round(float(np.std(fs)), 4), "worst": round(float(np.min(fs)), 4)}
    _, p = decode(oof)
    res["perm_acc"] = round(float((p.argmax(1) == gold).mean()), 4)
    res["cell_acc"] = round(float((PERMS_NP[p.argmax(1)] == gi).mean()), 4)
    res["nll"] = round(float(-np.mean(np.log(p[np.arange(len(p)), gold] + 1e-12))), 4)
    print(json.dumps(res, indent=1), flush=True)
    Path(f"_work/res_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
