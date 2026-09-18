#!/usr/bin/env python3
"""
Structured four-by-four correspondence starter.

Usage:
  python solution.py <public_dir> <submission.csv>
  python solution.py <public_dir> <cv.json> --cv

From-scratch ML only:
- permutation-invariant set encoder for opaque sketch tokens
- two-segment context representation
- outcome representation + outcome_kind
- learned 4x4 compatibility scores
- exact 24-permutation assignment distribution
- doubly-stochastic output probabilities by construction
- grouped CV by exchange `group`
- exact published structured-Brier metric
"""

import os
import sys
import json
import random
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold


SEED = 20260918
D_MODEL = 80
HIDDEN = 128
BATCH_SIZE = 128
EPOCHS = 8
LR = 2e-3
WEIGHT_DECAY = 2e-4
DROPOUT = 0.10
PATIENCE = 2
PAIR_LOSS_W = 0.25

PERMS = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_segments(s):
    toks = str(s).split()
    segs, cur = [], []
    for t in toks:
        if t == "segment_end":
            segs.append(cur)
            cur = []
        elif t not in ("role_a_0", "role_a_1"):
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs


def build_vocab(train):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    cols = [f"context_{i:02d}" for i in range(4)] + [f"outcome_{j:02d}" for j in range(4)]
    for col in cols:
        for s in train[col]:
            for tok in str(s).split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def encode_segments(segs, vocab, max_len=18):
    out = []
    for seg in segs[:2]:
        ids = [vocab.get(t, 1) for t in seg[:max_len]]
        if not ids:
            ids = [1]
        ids = ids + [0] * (max_len - len(ids))
        out.append(ids[:max_len])
    while len(out) < 2:
        out.append([0] * max_len)
    return np.asarray(out[:2], dtype=np.int64)


def encode_outcome(s, vocab, max_len=18):
    segs = split_segments(s)
    toks = segs[0] if segs else str(s).split()
    ids = [vocab.get(t, 1) for t in toks[:max_len]]
    if not ids:
        ids = [1]
    ids = ids + [0] * (max_len - len(ids))
    return np.asarray(ids[:max_len], dtype=np.int64)


def overlap_features(c_text, o_text):
    cs = set(str(c_text).split())
    os_ = set(str(o_text).split())
    cw = {x for x in cs if x.startswith("w_")}
    ow = {x for x in os_ if x.startswith("w_")}
    cv = {x for x in cs if x.startswith("v_")}
    ov = {x for x in os_ if x.startswith("v_")}
    cn = {x for x in cs if x.startswith("n_")}
    on = {x for x in os_ if x.startswith("n_")}
    inter = len(cs & os_)
    union = len(cs | os_)
    return np.asarray([
        inter,
        inter / max(1, union),
        inter / max(1, len(cs)),
        inter / max(1, len(os_)),
        len(cw & ow),
        len(cw & ow) / max(1, min(len(cw), len(ow))),
        len(cv & ov),
        len(cv & ov) / max(1, min(len(cv), len(ov))),
        len(cn & on),
        len(cn & on) / max(1, min(len(cn), len(on))),
        int("channel_event" in os_),
        int("channel_text" in os_),
        len(cs),
        len(os_),
    ], dtype=np.float32)


class EpisodeDataset(Dataset):
    def __init__(self, df, labels, vocab):
        self.df = df.reset_index(drop=True)
        self.labels = None if labels is None else labels.reshape(-1, 4, 4).astype(np.float32)
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ctx = np.stack([
            encode_segments(split_segments(r[f"context_{i:02d}"]), self.vocab)
            for i in range(4)
        ])
        out = np.stack([
            encode_outcome(r[f"outcome_{j:02d}"], self.vocab)
            for j in range(4)
        ])
        kinds = np.asarray([
            1 if r[f"outcome_kind_{j:02d}"] == "event" else 0
            for j in range(4)
        ], dtype=np.int64)
        ov = np.stack([
            overlap_features(r[f"context_{i:02d}"], r[f"outcome_{j:02d}"])
            for i in range(4) for j in range(4)
        ]).reshape(4, 4, -1)
        y = np.zeros((4, 4), dtype=np.float32) if self.labels is None else self.labels[idx]
        return (
            torch.from_numpy(ctx),
            torch.from_numpy(out),
            torch.from_numpy(kinds),
            torch.from_numpy(ov),
            torch.from_numpy(y),
        )


class SetEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, ns, l = x.shape
        h = self.emb(x)
        mask = x.ne(0)
        logits = self.gate(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        w = torch.softmax(logits, dim=-1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=-2)
        return self.proj(pooled)


class Matcher(nn.Module):
    def __init__(self, vocab_size, overlap_dim=14):
        super().__init__()
        self.encoder = SetEncoder(vocab_size)
        self.kind_emb = nn.Embedding(2, 8)
        pair_in = D_MODEL * 4 + overlap_dim + 8
        self.scorer = nn.Sequential(
            nn.Linear(pair_in, HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.GELU(),
            nn.Linear(HIDDEN // 2, 1),
        )

    def forward(self, ctx, out, kinds, overlap):
        b = ctx.size(0)
        ctx2 = ctx.view(b * 4, 2, ctx.size(-1))
        out2 = out.view(b * 4, out.size(-1))

        hc = self.encoder(ctx2)
        hc0, hc1 = hc[:, 0], hc[:, 1]
        hc = torch.cat([hc0, hc1, hc0 * hc1, torch.abs(hc0 - hc1)], dim=-1)
        hc = hc.view(b, 4, -1)

        ho = self.encoder(out2).squeeze(1).view(b, 4, -1)
        ke = self.kind_emb(kinds)

        c = hc.unsqueeze(2).expand(-1, 4, 4, -1)
        o = ho.unsqueeze(1).expand(-1, 4, 4, -1)
        k = ke.unsqueeze(1).expand(-1, 4, 4, -1)
        z = torch.cat([c, o, k, overlap], dim=-1)
        return self.scorer(z).squeeze(-1)


def permutation_distribution(pair_logits, temperature=1.0):
    p = PERMS.to(pair_logits.device)
    # [B, 24, 4] -> each row selects one candidate outcome.
    vals = []
    for k in range(24):
        vals.append(pair_logits[:, torch.arange(4, device=pair_logits.device), p[k]])
    perm_scores = torch.stack(vals, dim=1).sum(-1)
    perm_probs = torch.softmax(perm_scores / temperature, dim=-1)

    mat = torch.zeros(pair_logits.size(0), 4, 4, device=pair_logits.device)
    for k in range(24):
        mk = F.one_hot(p[k], num_classes=4).float()
        mat = mat + perm_probs[:, k].view(-1, 1, 1) * mk.view(1, 4, 4)
    return perm_probs, mat


def gold_perm_index(y):
    p = PERMS.numpy()
    gold = np.argmax(y, axis=2)
    out = []
    for row in gold:
        hit = np.where(np.all(p == row.reshape(1, 4), axis=1))[0]
        out.append(int(hit[0]) if len(hit) else 0)
    return np.asarray(out, dtype=np.int64)


def metric(pred, target):
    corr = ((pred - target) ** 2).mean(axis=(1, 2))
    rowm = ((pred.sum(axis=2) - 1.0) ** 2).mean(axis=1)
    colm = ((pred.sum(axis=1) - 1.0) ** 2).mean(axis=1)
    structure = (rowm + colm) / 18.0
    loss = 0.8 * corr + 0.2 * structure
    return float(np.mean(np.clip(1.0 - loss / 0.05, 0.0, 1.0)))


def predict_model(model, loader, device, temperature=1.0):
    model.eval()
    outs, ys = [], []
    with torch.no_grad():
        for ctx, out, kinds, overlap, y in loader:
            ctx = ctx.to(device)
            out = out.to(device)
            kinds = kinds.to(device)
            overlap = overlap.to(device)
            logits = model(ctx, out, kinds, overlap)
            _, mat = permutation_distribution(logits, temperature)
            outs.append(mat.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def train_one(model, train_loader, valid_loader, device, epochs=EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_score = None, -1.0
    bad = 0

    for ep in range(epochs):
        model.train()
        losses = []
        for ctx, out, kinds, overlap, y in train_loader:
            ctx = ctx.to(device)
            out = out.to(device)
            kinds = kinds.to(device)
            overlap = overlap.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(ctx, out, kinds, overlap)
            perm_probs, mat = permutation_distribution(logits, temperature=1.0)
            target_perm = torch.from_numpy(
                gold_perm_index(y.detach().cpu().numpy())
            ).to(device)

            brier = ((mat - y) ** 2).mean()
            ce = F.nll_loss(torch.log(torch.clamp(perm_probs, 1e-8, 1.0)), target_perm)
            pair_bce = F.binary_cross_entropy_with_logits(logits, y)
            loss = brier + PAIR_LOSS_W * ce + 0.10 * pair_bce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        pred_v, gold_v = predict_model(model, valid_loader, device)
        score = metric(pred_v, gold_v)
        print(
            f"[epoch {ep+1}] loss={np.mean(losses):.5f} val={score:.5f}",
            flush=True,
        )

        if score > best_score + 1e-5:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score


def make_loaders(train_df, y, vocab, tr_idx, va_idx):
    tr_ds = EpisodeDataset(train_df.iloc[tr_idx], y[tr_idx], vocab)
    va_ds = EpisodeDataset(train_df.iloc[va_idx], y[va_idx], vocab)
    tr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr, va


def run_cv(public_dir, out_json, n_splits=4):
    seed_everything()
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    labels_df = pd.read_csv(public_dir / "train_labels.csv")
    y = labels_df.drop(columns=["id"]).values.reshape(-1, 4, 4).astype(np.float32)

    vocab = build_vocab(train)
    groups = train["group"].astype(str).values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_scores = []
    for fold, (tri, vai) in enumerate(
        GroupKFold(n_splits).split(train, groups=groups)
    ):
        print(f"\n=== fold {fold} ===", flush=True)
        tr_loader, va_loader = make_loaders(train, y, vocab, tri, vai)
        model = Matcher(len(vocab)).to(device)
        best = train_one(model, tr_loader, va_loader, device)
        pred, gold = predict_model(model, va_loader, device)
        sc = metric(pred, gold)
        fold_scores.append(sc)
        print(
            f"[fold {fold}] best={best:.5f} final={sc:.5f}",
            flush=True,
        )

    report = {
        "validation": {
            "scheme": "4-fold GroupKFold by exchange group; exact published mean_clipped_structured_brier_v1 metric",
            "fold_scores": [round(float(x), 6) for x in fold_scores],
            "mean": round(float(np.mean(fold_scores)), 6),
            "std": round(float(np.std(fold_scores)), 6),
            "worst": round(float(np.min(fold_scores)), 6),
        },
        "model": {
            "architecture": "from-scratch permutation-invariant set encoder + 4x4 pair scorer + exact 24-permutation assignment distribution",
            "d_model": D_MODEL,
            "hidden": HIDDEN,
            "epochs_max": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "pair_loss_weight": PAIR_LOSS_W,
            "pretrained": False,
        },
        "constraints": {
            "external_data": False,
            "network": False,
            "package_install": False,
            "synthetic_training": False,
            "test_labels": False,
            "test_fit_statistics": False,
            "used_group_as_model_feature": False,
            "deterministic": True,
        },
        "final_submission": {
            "generated_by": "solution.py",
            "submission_schema": "sample_submission.csv exact columns/order",
        },
    }
    Path(out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_submission(public_dir, out_csv):
    seed_everything()
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    labels_df = pd.read_csv(public_dir / "train_labels.csv")
    test = pd.read_csv(public_dir / "test.csv")
    y = labels_df.drop(columns=["id"]).values.reshape(-1, 4, 4).astype(np.float32)

    vocab = build_vocab(train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a train-only grouped holdout for early stopping, then do a short
    # final fine-tune on all labeled rows. No test row/statistic is used in fit.
    groups = train["group"].astype(str).values
    tr_idx, va_idx = next(GroupKFold(5).split(train, groups=groups))
    tr_loader, va_loader = make_loaders(train, y, vocab, tr_idx, va_idx)

    model = Matcher(len(vocab)).to(device)
    train_one(model, tr_loader, va_loader, device)

    full_ds = EpisodeDataset(train, y, vocab)
    full_loader = DataLoader(
        full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR * 0.5, weight_decay=WEIGHT_DECAY
    )
    for ep in range(2):
        model.train()
        for ctx, out, kinds, overlap, yy in full_loader:
            ctx = ctx.to(device)
            out = out.to(device)
            kinds = kinds.to(device)
            overlap = overlap.to(device)
            yy = yy.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(ctx, out, kinds, overlap)
            probs, mat = permutation_distribution(logits, temperature=1.0)
            target_perm = torch.from_numpy(
                gold_perm_index(yy.detach().cpu().numpy())
            ).to(device)

            loss = ((mat - yy) ** 2).mean()
            loss = loss + PAIR_LOSS_W * F.nll_loss(
                torch.log(torch.clamp(probs, 1e-8, 1.0)),
                target_perm,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    test_ds = EpisodeDataset(test, None, vocab)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    pred, _ = predict_model(model, test_loader, device, temperature=1.0)

    sample = pd.read_csv(public_dir / "sample_submission.csv")
    out = pd.DataFrame({"id": test["id"].values})
    flat = pred.reshape(len(test), 16)
    for k, col in enumerate(sample.columns[1:]):
        out[col] = flat[:, k]

    out = out[sample.columns]

    if not np.isfinite(out.iloc[:, 1:].to_numpy()).all():
        raise RuntimeError("non-finite prediction")
    vals = out.iloc[:, 1:].to_numpy()
    if ((vals < 0) | (vals > 1)).any():
        raise RuntimeError("prediction outside [0,1]")
    if list(out.columns) != list(sample.columns):
        raise RuntimeError("submission columns differ from sample_submission.csv")
    if list(out["id"]) != list(test["id"]):
        raise RuntimeError("submission id order differs from test.csv")
    if out["id"].duplicated().any():
        raise RuntimeError("duplicate IDs")

    out.to_csv(out_csv, index=False)
    print(json.dumps({
        "rows": int(len(out)),
        "validation_errors": 0,
        "sha256": hashlib.sha256(Path(out_csv).read_bytes()).hexdigest(),
    }, indent=2))


def main():
    args = sys.argv[1:]
    if "--cv" in args:
        args.remove("--cv")
        if len(args) != 2:
            raise SystemExit(
                "Usage: python solution.py <public_dir> <cv.json> --cv"
            )
        run_cv(args[0], args[1])
        return

    if len(args) != 2:
        raise SystemExit(
            "Usage: python solution.py <public_dir> <submission.csv>"
        )
    run_submission(args[0], args[1])


if __name__ == "__main__":
    main()
