"""
Grid Balancing Action Portfolio - from-scratch solution.

Usage:
    python3 solution.py <public_dir> <submission_out>          # train on all training days, predict test
    python3 solution.py <public_dir> <cv_out.json> --cv         # identical pipeline under day-grouped CV

Pipeline (all models randomly initialised and trained only on the supplied training split):
  1. Candidate features: 29 raw features, context, settlement period, direction-aware merit-order ranks and
     cumulative MW, episode aggregates.
  2. Fold-safe unit-token statistics: smoothed acceptance priors per (unit token, direction), instruction-time
     and trajectory priors, per-token input normalisation. Training rows use leave-one-day-out statistics;
     prediction rows use all training days.
  3. Candidate acceptance: LightGBM binary model + from-scratch set-attention network with learned unit-key
     embeddings (3 seeds, CPU, deterministic), blended in logit space.
  4. Portfolio size: expected-F1-optimal k from blended probabilities, scaled.
  5. Action heads (LightGBM): instruction minute (L1), per-minute trajectory coverage (binary),
     per-minute absolute level as a residual to PN (L1).
  6. Decoding: rounded instruction minute, expected-IoU best interval for coverage, anchor snapping of levels
     to MEL/MIL/0/PN start/PN end, compression to piecewise-constant segments, canonical ordering.
No pretrained weights, external data, test labels, or cross-episode test linkage are used.
"""
import sys
import os
import json
import time
import random
import csv
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

SEED = 20260917
NUM_THREADS = 8
NFOLD = 4

GBDT_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                   verbose=-1, num_threads=NUM_THREADS, seed=7, deterministic=True, force_row_wise=True)
GBDT_ROUNDS = 1000
NN_CFG = dict(epochs=12, bs=8, lr=1e-3, drop=0.25, wd=1e-3, d=128, demb=16)
NN_SEEDS = [0, 1, 2]
W_NN = 0.6
K_SCALE = 1.1
HEAD_PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=NUM_THREADS,
                   seed=11, deterministic=True, force_row_wise=True)
HEAD_ROUNDS = (800, 700, 800)  # instruction, coverage, level
SNAP_DELTA = 1.0
COVER_MODE = "interval"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(NUM_THREADS)
    torch.use_deterministic_algorithms(True)


# =============================================================================================
# Data
# =============================================================================================

def period_of(ctx):
    return int(round((np.arctan2(ctx[0], ctx[1]) / (2 * np.pi) * 48) % 48))


def load_split(public_dir, split):
    df = pd.read_csv(os.path.join(public_dir, f"{split}.csv"))
    eps = []
    for r in df.itertuples(index=False):
        with np.load(os.path.join(public_dir, r.episode_path), allow_pickle=False) as z:
            e = {
                "features": z["features"].astype(np.float32),
                "cids": z["candidate_ids"].astype(str),
                "toks": z["unit_tokens"].astype(str),
                "dirs": z["directions"].astype(np.int8),
                "ctx": z["context"].astype(np.float32),
            }
        e["id"] = str(r.id)
        e["day"] = str(r.day_group)
        e["period"] = period_of(e["ctx"])
        if "portfolio_json" in df.columns:
            e["target"] = json.loads(r.portfolio_json)
            idx = {c: i for i, c in enumerate(e["cids"])}
            y = np.zeros(len(e["cids"]), np.int8)
            for a in e["target"]:
                y[idx[a["action_id"]]] = 1
            e["y"] = y
        eps.append(e)
    return df, eps


def true_grid(a):
    g = np.full(30, np.nan)
    for s in a["trajectory"]:
        g[int(round(s["t0"])):int(round(s["t1"]))] = s["level0"]
    return g


# =============================================================================================
# Candidate features
# =============================================================================================

def rank01(x):
    n = len(x)
    if n <= 1:
        return np.zeros(n, np.float32)
    o = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return (o / (n - 1)).astype(np.float32)


def episode_base_features(e):
    F = np.nan_to_num(e["features"].astype(np.float64))
    n = F.shape[0]
    d = F[:, 0]
    pn, pns, pne = F[:, 1], F[:, 2], F[:, 3]
    mel, mil, hr, fr, flex = F[:, 5], F[:, 6], F[:, 7], F[:, 8], F[:, 9]
    best, worst = F[:, 10], F[:, 11]
    offer = d > 0
    bid = ~offer
    room = np.where(offer, hr, fr)
    merit = np.where(offer, best, -best)
    placeholder = (np.abs(best) >= 999).astype(np.float64)

    out = {}
    for j in range(29):
        out[f"f{j}"] = F[:, j]
    out["room"] = room
    out["room_frac_flex"] = room / np.maximum(flex, 1.0)
    out["placeholder"] = placeholder
    out["running"] = (np.abs(pn) > 0).astype(np.float64)
    out["pn_delta"] = pne - pns
    out["pn_delta_dir"] = (pne - pns) * d
    out["mel_minus_mil"] = mel - mil
    out["price_spread"] = worst - best

    merit_rank = np.zeros(n)
    merit_rank_room = np.zeros(n)
    cum_room = np.zeros(n)
    cum_flex = np.zeros(n)
    zprice = np.zeros(n)
    for mask in (offer, bid):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        m = merit[idx].copy()
        m[placeholder[idx] > 0] = 1e6
        merit_rank[idx] = rank01(m)
        o = np.argsort(m, kind="mergesort")
        cr = np.cumsum(room[idx][o])
        cf = np.cumsum(flex[idx][o])
        tmp = np.empty(len(idx))
        tmp[o] = cr
        cum_room[idx] = tmp / max(cr[-1], 1.0)
        tmp[o] = cf
        cum_flex[idx] = tmp / max(cf[-1], 1.0)
        valid = placeholder[idx] == 0
        if valid.sum() > 2:
            mu = np.median(best[idx][valid])
            sd = np.std(best[idx][valid]) + 1.0
            zprice[idx] = np.clip((best[idx] - mu) / sd, -10, 10)
        mr = m.copy()
        mr[room[idx] <= 0] = 1e7
        merit_rank_room[idx] = rank01(mr)
    out["merit_rank"] = merit_rank
    out["merit_rank_room"] = merit_rank_room
    out["cum_room_merit"] = cum_room
    out["cum_flex_merit"] = cum_flex
    out["zprice_dir"] = zprice
    out["rank_flex"] = rank01(flex)
    out["rank_pn"] = rank01(pn)
    out["rank_room"] = rank01(room)

    ctx = e["ctx"].astype(np.float64)
    for j in range(6):
        out[f"ctx{j}"] = np.full(n, ctx[j])
    out["period"] = np.full(n, e["period"], np.float64)
    out["ep_n"] = np.full(n, n, np.float64)
    out["ep_offer_room"] = np.full(n, room[offer].sum())
    out["ep_bid_room"] = np.full(n, room[bid].sum())
    out["ep_pn_pos"] = np.full(n, pn[pn > 0].sum())
    out["ep_pn_neg"] = np.full(n, pn[pn < 0].sum())
    out["ep_negbid_pn"] = np.full(n, pn[bid & (best < 0) & (placeholder == 0)].sum())
    names = list(out.keys())
    X = np.stack([out[k] for k in names], axis=1).astype(np.float32)
    return X, names


def assign_keys(eps, vocab):
    for e in eps:
        keys = np.char.add(e["toks"], np.where(e["dirs"] > 0, "+", "-"))
        e["kidx"] = np.array([vocab.setdefault(k, len(vocab)) for k in keys], np.int32)


class KeyStats:
    """Per-day accumulated per-(token, direction) label statistics; any day subset is a sum."""
    NPB = 6

    def __init__(self, train_eps, K):
        days = sorted(set(e["day"] for e in train_eps))
        self.days = days
        dix = {d: i for i, d in enumerate(days)}
        D = len(days)
        self.cnt = np.zeros((D, K))
        self.pos = np.zeros((D, K))
        self.cnt_p = np.zeros((D, K, self.NPB))
        self.pos_p = np.zeros((D, K, self.NPB))
        self.ins = np.zeros((D, K))
        self.ins2 = np.zeros((D, K))
        self.cover = np.zeros((D, K))
        self.cov_m = np.zeros((D, K, 30), np.float32)
        self.lev_m = np.zeros((D, K, 30), np.float32)
        self.ins_h = np.zeros((D, K, 9), np.float32)
        for e in train_eps:
            di = dix[e["day"]]
            k = e["kidx"]
            pb = e["period"] // 8
            np.add.at(self.cnt[di], k, 1)
            np.add.at(self.pos[di], k, e["y"])
            np.add.at(self.cnt_p[di, :, pb], k, 1)
            np.add.at(self.pos_p[di, :, pb], k, e["y"])
            idx = {c: i for i, c in enumerate(e["cids"])}
            for a in e["target"]:
                i = idx[a["action_id"]]
                ki = k[i]
                self.ins[di, ki] += a["instruction_minute"]
                self.ins2[di, ki] += a["instruction_minute"] ** 2
                self.cover[di, ki] += sum(s["t1"] - s["t0"] for s in a["trajectory"])
                g = true_grid(a)
                c = ~np.isnan(g)
                self.cov_m[di, ki] += c
                self.lev_m[di, ki] += np.where(c, g - e["features"][i, 1], 0.0)
                self.ins_h[di, ki, int(np.clip((a["instruction_minute"] + 60) // 10, 0, 8))] += 1

    def priors(self, day_list, alpha=20.0):
        m = np.isin(self.days, list(day_list))
        cnt = self.cnt[m].sum(0)
        pos = self.pos[m].sum(0)
        cnt_p = self.cnt_p[m].sum(0)
        pos_p = self.pos_p[m].sum(0)
        base = pos.sum() / max(cnt.sum(), 1)
        rate = (pos + alpha * base) / (cnt + alpha)
        rate_p = (pos_p + alpha * rate[:, None]) / (cnt_p + alpha)
        dayfrac = (self.pos[m] > 0).sum(0) / max(m.sum(), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ins = np.where(pos > 0, self.ins[m].sum(0) / pos, np.nan)
            ins_sd = np.where(pos > 1, np.sqrt(np.maximum(self.ins2[m].sum(0) / pos - ins ** 2, 0)), np.nan)
            cover = np.where(pos > 0, self.cover[m].sum(0) / pos, np.nan)
        covm = self.cov_m[m].sum(0)
        levm = self.lev_m[m].sum(0)
        insh = self.ins_h[m].sum(0)
        gcov = covm.sum(0) / max(pos.sum(), 1)
        cov_rate = (covm + 5.0 * gcov[None, :]) / (pos[:, None] + 5.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lev_rel = np.where(covm > 0, levm / np.maximum(covm, 1), np.nan)
        ins_hist = (insh + 1.0) / (pos[:, None] + 9.0)
        return dict(rate=rate, rate_p=rate_p, pos=pos, dayfrac=dayfrac, ins=ins, ins_sd=ins_sd, cover=cover,
                    cov_rate=cov_rate, lev_rel=lev_rel, ins_hist=ins_hist)

    @staticmethod
    def transform(e, pr):
        k = e["kidx"]
        pb = e["period"] // 8
        cols = [pr["rate"][k], pr["rate_p"][k, pb], pr["pos"][k], pr["dayfrac"][k], pr["ins"][k], pr["ins_sd"][k],
                pr["cover"][k]]
        return np.stack(cols, 1).astype(np.float32), ["tok_rate", "tok_rate_pb", "tok_pos", "tok_dayfrac", "tok_ins",
                                                       "tok_ins_sd", "tok_cover"]


class KeyInputStats:
    """Per-key input statistics (all rows) and accepted-row conditional means, per day."""
    VARS = ["f1", "room", "f10", "f5", "merit_rank_room", "cum_flex_merit", "running"]

    def __init__(self, train_eps, K):
        days = sorted(set(e["day"] for e in train_eps))
        self.days = days
        dix = {d: i for i, d in enumerate(days)}
        D, V = len(days), len(self.VARS)
        self.n = np.zeros((D, K))
        self.s = np.zeros((D, K, V))
        self.s2 = np.zeros((D, K, V))
        self.na = np.zeros((D, K))
        self.sa = np.zeros((D, K, V))
        for e in train_eps:
            di = dix[e["day"]]
            Xz = np.nan_to_num(self._vals(e))
            k = e["kidx"]
            np.add.at(self.n[di], k, 1)
            np.add.at(self.s[di], k, Xz)
            np.add.at(self.s2[di], k, Xz ** 2)
            y = e["y"].astype(bool)
            np.add.at(self.na[di], k[y], 1)
            np.add.at(self.sa[di], k[y], Xz[y])

    def _vals(self, e):
        j = {nm: i for i, nm in enumerate(e["Xb_names"])}
        X = e["Xb"][:, [j[v] for v in self.VARS]].astype(np.float64)
        X[:, 2] = np.where(np.abs(X[:, 2]) >= 999, np.nan, X[:, 2])
        return X

    def stats(self, day_list):
        m = np.isin(self.days, list(day_list))
        n = self.n[m].sum(0)[:, None]
        s = self.s[m].sum(0)
        s2 = self.s2[m].sum(0)
        na = self.na[m].sum(0)[:, None]
        sa = self.sa[m].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = s / n
            sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1.0
            mua = np.where(na > 0, sa / na, np.nan)
        return mu, sd, mua

    def transform(self, e, st):
        mu, sd, mua = st
        X = self._vals(e)
        k = e["kidx"]
        with np.errstate(invalid="ignore"):
            z = (X - mu[k]) / sd[k]
            za = (X - mua[k]) / sd[k]
        names = [f"kz_{v}" for v in self.VARS] + [f"kza_{v}" for v in self.VARS]
        return np.concatenate([z, za], 1).astype(np.float32), names


class FeatureContext:
    """Caches prior/statistic dictionaries per day subset."""

    def __init__(self, ks, kis):
        self.ks, self.kis = ks, kis
        self.pr_cache, self.st_cache = {}, {}

    def priors(self, days):
        key = tuple(sorted(days))
        if key not in self.pr_cache:
            self.pr_cache[key] = self.ks.priors(key)
        return self.pr_cache[key]

    def stats(self, days):
        key = tuple(sorted(days))
        if key not in self.st_cache:
            self.st_cache[key] = self.kis.stats(key)
        return self.st_cache[key]


def candidate_rows(eps, fc, prior_days_fn):
    """Full candidate matrix over ALL legal candidates. Keys never accepted in the prior days keep neutral
    smoothed priors (global base rate, zero counts, missing timing statistics) instead of being excluded."""
    Xs, ys, rows_idx = [], [], []
    names = None
    for e in eps:
        days = prior_days_fn(e)
        pr = fc.priors(days)
        T, tn = KeyStats.transform(e, pr)
        Z, zn = fc.kis.transform(e, fc.stats(days))
        X = np.concatenate([e["Xb"], T, Z], 1)
        names = e["Xb_names"] + tn + zn
        idx = np.arange(len(X))
        Xs.append(X[idx])
        if "y" in e:
            ys.append(e["y"][idx])
        rows_idx.append(idx)
    X = np.concatenate(Xs) if Xs else np.zeros((0, len(names or [])), np.float32)
    y = np.concatenate(ys) if ys else None
    return X, y, rows_idx, names


# =============================================================================================
# Set network (from scratch)
# =============================================================================================

class Prep:
    def fit(self, X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        self.mu = np.nanmean(Z, 0)
        self.sd = np.nanstd(Z, 0) + 1e-3
        self.mu = np.nan_to_num(self.mu)
        self.sd = np.nan_to_num(self.sd, nan=1.0)
        self.nancols = np.where(np.isnan(X).any(0))[0]
        return self

    def transform(self, X):
        Z = (np.sign(X) * np.log1p(np.abs(X)) - self.mu) / self.sd
        miss = np.isnan(X[:, self.nancols]).astype(np.float32)
        Z = np.nan_to_num(np.clip(Z, -6, 6))
        return np.concatenate([Z, miss], 1).astype(np.float32)


class SetNet(nn.Module):
    def __init__(self, nin, nkeys, d=128, demb=16, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(nkeys, demb)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.inp = nn.Sequential(nn.Linear(nin + demb, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.GELU())
        self.mha = nn.MultiheadAttention(d, 4, dropout=0.0, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln2 = nn.LayerNorm(d)
        self.pool_q = nn.Linear(d, 1)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x, k, mask):
        h = self.inp(torch.cat([x, self.emb(k)], -1))
        a, _ = self.mha(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = self.ln1(h + a)
        h = self.ln2(h + self.ff(h))
        w = self.pool_q(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(w, 1).unsqueeze(-1)
        c = (w * h).sum(1, keepdim=True).expand_as(h)
        return self.out(torch.cat([h, c], -1)).squeeze(-1)


def pad_batch(items):
    n = max(len(x) for x, _, _ in items)
    B = len(items)
    F = items[0][0].shape[1]
    X = np.zeros((B, n, F), np.float32)
    K = np.zeros((B, n), np.int64)
    Y = np.zeros((B, n), np.float32)
    M = np.zeros((B, n), bool)
    for i, (x, k, y) in enumerate(items):
        m = len(x)
        X[i, :m] = x
        K[i, :m] = k
        M[i, :m] = True
        if y is not None:
            Y[i, :m] = y
    return torch.from_numpy(X), torch.from_numpy(K), torch.from_numpy(Y), torch.from_numpy(M)


def train_setnet(items, nin, nkeys, seed, epochs, bs, lr, drop, wd, d, demb):
    seed_everything(SEED + 1000 * seed)
    net = SetNet(nin, nkeys, d=d, demb=demb, drop=drop)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(items) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(SEED + seed)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(items))
        for b in range(0, len(perm), bs):
            X, K, Y, M = pad_batch([items[i] for i in perm[b:b + bs]])
            logit = net(X, K, M)
            loss = (lossf(logit, Y) * M).sum() / M.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    return net


@torch.no_grad()
def predict_setnet(net, items, bs=16):
    net.eval()
    out = []
    for b in range(0, len(items), bs):
        chunk = items[b:b + bs]
        X, K, _, M = pad_batch(chunk)
        p = torch.sigmoid(net(X, K, M)).numpy().astype(np.float64)
        for i, (x, _, _) in enumerate(chunk):
            out.append(p[i, :len(x)])
    return out


# =============================================================================================
# Action heads
# =============================================================================================

def action_matrix(e, cand_idx, pr):
    T, _ = KeyStats.transform(e, pr)
    k = e["kidx"][cand_idx]
    ih = pr["ins_hist"][k]
    X = np.concatenate([e["Xb"][cand_idx], T[cand_idx], ih], 1).astype(np.float32)
    return X, k


def minute_expand(Xa, k, pr):
    n = len(Xa)
    mins = np.tile(np.arange(30, dtype=np.float32), n)
    cr = pr["cov_rate"][k].reshape(-1, 1)
    lr_ = pr["lev_rel"][k].reshape(-1, 1)
    covsum = np.repeat(pr["cov_rate"][k].sum(1), 30)[:, None]
    cum_before = np.cumsum(pr["cov_rate"][k], 1).reshape(-1, 1)
    tins = np.nan_to_num(pr["ins"][k], nan=-13.0)
    m_minus_ins = (mins - np.repeat(tins, 30))[:, None]
    return np.concatenate([np.repeat(Xa, 30, 0), mins[:, None], cr, lr_, covsum, cum_before, m_minus_ins],
                          1).astype(np.float32)


def fit_heads(train_eps, fc, prior_days_fn):
    Xs, XMs, ins, grids, pns = [], [], [], [], []
    for e in train_eps:
        if not e["target"]:
            continue
        idx = {c: i for i, c in enumerate(e["cids"])}
        ci = np.array([idx[a["action_id"]] for a in e["target"]])
        days = prior_days_fn(e)
        pr = fc.priors(days)
        X, k = action_matrix(e, ci, pr)
        Xs.append(X)
        XMs.append(minute_expand(X, k, pr))
        ins.extend(a["instruction_minute"] for a in e["target"])
        grids.extend(true_grid(a) for a in e["target"])
        pns.append(e["features"][ci, 1].astype(np.float64))
    X = np.concatenate(Xs)
    XM = np.concatenate(XMs)
    ins = np.asarray(ins, np.float64)
    G = np.stack(grids)
    pn = np.concatenate(pns)
    del Xs, XMs
    m_ins = lgb.train(dict(HEAD_PARAMS, objective="l1"), lgb.Dataset(X, ins), HEAD_ROUNDS[0])
    cov = (~np.isnan(G)).reshape(-1)
    m_cov = lgb.train(dict(HEAD_PARAMS, objective="binary"), lgb.Dataset(XM, cov.astype(np.float32)), HEAD_ROUNDS[1])
    rel = (G - pn[:, None]).reshape(-1)
    m_lev = lgb.train(dict(HEAD_PARAMS, objective="l1"), lgb.Dataset(XM[cov], rel[cov]), HEAD_ROUNDS[2])
    return m_ins, m_cov, m_lev


def predict_heads(heads, e, ci, pr):
    m_ins, m_cov, m_lev = heads
    X, k = action_matrix(e, ci, pr)
    XM = minute_expand(X, k, pr)
    pins = m_ins.predict(X)
    pcov = m_cov.predict(XM).reshape(-1, 30)
    plev = m_lev.predict(XM).reshape(-1, 30) + e["features"][ci, 1][:, None].astype(np.float64)
    return pins, pcov, plev


# =============================================================================================
# Decoding
# =============================================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def choose_k(p, kscale):
    s = np.sort(p)[::-1]
    cs = np.cumsum(s)
    k = np.arange(1, len(s) + 1)
    f = 2 * cs / (k + p.sum())
    kb = int(np.argmax(f)) + 1
    return int(min(len(p), max(0, int(np.floor(kb * kscale + 0.5)))))


def coverage_mask(pc, mode):
    n = len(pc)
    if mode == "interval":
        cs = np.concatenate([np.zeros((n, 1)), np.cumsum(pc, 1)], 1)
        tot = cs[:, -1]
        best = np.full(n, -1.0)
        ba = np.zeros(n, int)
        bb = np.ones(n, int)
        for a in range(30):
            for b in range(a + 1, 31):
                inside = cs[:, b] - cs[:, a]
                v = inside / ((b - a) + tot - inside)
                upd = v > best
                best[upd] = v[upd]
                ba[upd] = a
                bb[upd] = b
        idx = np.arange(30)[None, :]
        M = (idx >= ba[:, None]) & (idx < bb[:, None])
    else:
        o = np.argsort(-pc, axis=1, kind="mergesort")
        s = np.take_along_axis(pc, o, 1)
        cs = np.cumsum(s, 1)
        tot = pc.sum(1, keepdims=True)
        j = np.arange(1, 31)[None, :]
        jb = np.argmax(cs / (j + tot - cs), 1) + 1
        rank = np.argsort(o, axis=1, kind="mergesort")
        M = rank < jb[:, None]
    empty = ~M.any(1)
    if empty.any():
        M[empty, np.argmax(pc[empty], 1)] = True
    return M


def snap_levels(plev, anchors, delta):
    d = np.abs(plev[:, :, None] - anchors[:, None, :])
    j = np.argmin(d, 2)
    dm = np.take_along_axis(d, j[:, :, None], 2)[:, :, 0]
    av = np.take_along_axis(np.broadcast_to(anchors[:, None, :], d.shape), j[:, :, None], 2)[:, :, 0]
    tol = delta * np.maximum(10.0, 0.1 * np.abs(av))
    return np.where(dm < tol, av, plev)


def grid_to_segments(g):
    segs = []
    j = 0
    while j < 30:
        if np.isnan(g[j]):
            j += 1
            continue
        v = round(float(g[j]), 3)
        k = j + 1
        while k < 30 and not np.isnan(g[k]) and round(float(g[k]), 3) == v:
            k += 1
        segs.append({"level0": v, "level1": v, "t0": float(j), "t1": float(k)})
        j = k
    return segs


def decode_actions(e, ci, pins, pcov, plev):
    ins = np.round(np.clip(pins, -59, 29))
    M = coverage_mask(pcov, COVER_MODE)
    F = e["features"][ci].astype(np.float64)
    anchors = np.stack([F[:, 5], F[:, 6], np.zeros(len(F)), F[:, 2], F[:, 3]], 1)
    lev = snap_levels(plev, anchors, SNAP_DELTA) if SNAP_DELTA > 0 else plev
    lev = np.clip(lev, -10000, 10000)
    G = np.where(M, lev, np.nan)
    acts = []
    for j, c in enumerate(ci):
        acts.append({"action_id": str(e["cids"][c]), "instruction_minute": float(ins[j]),
                     "trajectory": grid_to_segments(G[j])})
    acts.sort(key=lambda a: (a["instruction_minute"], a["action_id"]))
    return acts, G


# =============================================================================================
# Full pipeline: train on train_eps, predict pred_eps
# =============================================================================================

def run_pipeline(train_eps, pred_eps, log=print):
    t0 = time.time()
    seed_everything()
    for e in train_eps + pred_eps:
        if "Xb" not in e:
            e["Xb"], e["Xb_names"] = episode_base_features(e)
    vocab = {}
    assign_keys(sorted(train_eps, key=lambda e: e["id"]) + sorted(pred_eps, key=lambda e: e["id"]), vocab)
    K = len(vocab)
    ks = KeyStats(train_eps, K)
    kis = KeyInputStats(train_eps, K)
    fc = FeatureContext(ks, kis)
    tdays = ks.days
    lodo = lambda e: [d for d in tdays if d != e["day"]]
    alld = lambda e: tdays

    # ---- candidate matrices
    Xt, yt, rit, names = candidate_rows(train_eps, fc, lodo)
    Xp, _, rip, _ = candidate_rows(pred_eps, fc, alld)
    log(f"candidate rows train={len(Xt)} pred={len(Xp)} feats={Xt.shape[1]} t={time.time()-t0:.0f}s")

    # ---- GBDT
    gbm = lgb.train(GBDT_PARAMS, lgb.Dataset(Xt, yt.astype(np.float32)), GBDT_ROUNDS)
    g_raw = gbm.predict(Xp, raw_score=True)
    log(f"gbdt done t={time.time()-t0:.0f}s")

    # ---- set network
    prep = Prep().fit(Xt)
    Zt = prep.transform(Xt)
    Zp = prep.transform(Xp)
    tr_items, st = [], 0
    for e, r in zip(train_eps, rit):
        n = len(r)
        tr_items.append((Zt[st:st + n], e["kidx"][r], yt[st:st + n].astype(np.float32)))
        st += n
    pr_items, st = [], 0
    for e, r in zip(pred_eps, rip):
        n = len(r)
        pr_items.append((Zp[st:st + n], e["kidx"][r], None))
        st += n
    nn_sum = None
    for sd in NN_SEEDS:
        net = train_setnet(tr_items, Zt.shape[1], K, seed=sd, **NN_CFG)
        pp = predict_setnet(net, pr_items)
        nn_sum = pp if nn_sum is None else [a + b for a, b in zip(nn_sum, pp)]
        log(f"setnet seed {sd} done t={time.time()-t0:.0f}s")
    nn_p = [a / len(NN_SEEDS) for a in nn_sum]
    del Zt, Zp, tr_items

    # ---- heads
    heads = fit_heads(train_eps, fc, lodo)
    log(f"heads done t={time.time()-t0:.0f}s")
    seed_everything()

    # ---- decode
    pr_all = fc.priors(tdays)
    preds = {}
    grids = {}
    st = 0
    for e, r, pn_ in zip(pred_eps, rip, nn_p):
        n = len(r)
        s = (1 - W_NN) * g_raw[st:st + n] + W_NN * logit(pn_)
        st += n
        if n == 0:
            preds[e["id"]] = []
            continue
        p = sigmoid(s)
        k = choose_k(p, K_SCALE)
        order = np.argsort(-s, kind="mergesort")[:k]
        ci = r[order]
        if len(ci) == 0:
            preds[e["id"]] = []
            continue
        pins, pcov, plev = predict_heads(heads, e, ci, pr_all)
        acts, G = decode_actions(e, ci, pins, pcov, plev)
        preds[e["id"]] = acts
    log(f"decode done t={time.time()-t0:.0f}s")
    return preds


# =============================================================================================
# Local metric (reconstruction of the stated structure; exact official formula not available locally)
# =============================================================================================

def _order_sim(pt, tt):
    n = len(pt)
    if n < 2:
        return 1.0
    pt = np.asarray(pt, float)
    tt = np.asarray(tt, float)
    iu = np.triu_indices(n, 1)
    dp = np.sign(pt[:, None] - pt[None, :])[iu]
    dt = np.sign(tt[:, None] - tt[None, :])[iu]
    return float(np.where(dp == dt, 1.0, np.where((dp == 0) | (dt == 0), 0.5, 0.0)).mean())


METRIC_VARIANTS = {
    "official": {"time_kind": "official", "lev_kind": "official"},
    "main": {},
    "rec_cov": {"cov": "rec"},
    "lin_time": {"time_kind": "lin"},
    "strict_lev": {"lev_floor": 5.0, "lev_rel": 0.05, "traj_comb": "prod"},
    "loose_lev": {"lev_floor": 25.0, "lev_rel": 0.25, "lev_kind": "lin"},
    "lenient": {"time_kind": "lin", "lev_kind": "mae_rel"},
}


def _traj_sim(PG, TG, v):
    pc, tc = ~np.isnan(PG), ~np.isnan(TG)
    inter, union = pc & tc, pc | tc
    ni, nu = inter.sum(1), union.sum(1)
    iou = np.where(nu > 0, ni / np.maximum(nu, 1), 1.0)
    err = np.abs(np.nan_to_num(PG) - np.nan_to_num(TG))
    if v.get("lev_kind") == "official":
        mae = (err * inter).sum(1) / np.maximum(ni, 1)
        scale = np.maximum(25.0, 0.15 * (np.abs(np.nan_to_num(TG)) * tc).sum(1) / np.maximum(tc.sum(1), 1) + 25.0)
        s = 0.45 * iou + 0.55 * iou * np.clip(1.0 - mae / (2 * scale), 0, 1)
    elif v.get("lev_kind") == "mae_rel":
        mae = (err * inter).sum(1) / np.maximum(ni, 1)
        mag = np.maximum(10.0, (np.abs(np.nan_to_num(TG)) * tc).sum(1) / np.maximum(tc.sum(1), 1))
        s = 0.5 * iou + 0.5 * np.clip(1.0 - mae / mag, 0, 1)
    else:
        scale = np.maximum(v.get("lev_floor", 10.0), v.get("lev_rel", 0.1) * np.abs(np.nan_to_num(TG)))
        acc = np.exp(-err / scale) if v.get("lev_kind", "exp") == "exp" else np.clip(1 - err / (2 * scale), 0, 1)
        lev = np.where(ni > 0, (acc * inter).sum(1) / np.maximum(ni, 1), 0.0)
        s = 0.5 * iou + 0.5 * iou * lev if v.get("traj_comb", "mean") == "mean" else iou * lev
    return np.where(nu == 0, 1.0, np.where(ni == 0, 0.0, s))


def score_episode(pred, true, v):
    npred, ntrue = len(pred), len(true)
    if npred == 0 and ntrue == 0:
        return dict(score=1.0, f1=1.0, prec=1.0, rec=1.0, inst=1.0, traj=1.0, order=1.0, count=1.0, abs_ins_err=0.0)
    tm = {a["action_id"]: a for a in true}
    pairs = [(p, tm[p["action_id"]]) for p in pred if p["action_id"] in tm]
    m = len(pairs)
    f1 = 2 * m / (npred + ntrue)
    prec = m / npred if npred else 0.0
    rec = m / ntrue if ntrue else 0.0
    count = min(npred, ntrue) / max(npred, ntrue)
    if m:
        pi = np.array([p["instruction_minute"] for p, _ in pairs])
        ti = np.array([t["instruction_minute"] for _, t in pairs])
        dt = np.abs(pi - ti)
        if v.get("time_kind") == "official":
            inst = float(np.mean(np.clip(1 - dt / 20.0, 0, 1)))
        else:
            inst = float(np.mean(np.exp(-dt / 15.0) if v.get("time_kind", "exp") == "exp" else np.clip(1 - dt / 30.0, 0, 1)))
        PG = np.stack([true_grid(p) for p, _ in pairs])
        TG = np.stack([true_grid(t) for _, t in pairs])
        traj = float(_traj_sim(PG, TG, v).mean())
        order = _order_sim(pi, ti)
        abs_ins_err = float(dt.mean())
    else:
        inst = traj = order = 0.0
        abs_ins_err = float("nan")
    cov = {"f1": f1, "rec": rec}[v.get("cov", "f1")]
    score = 0.40 * f1 + cov * (0.15 * inst + 0.25 * traj + 0.15 * order) + 0.05 * count
    return dict(score=score, f1=f1, prec=prec, rec=rec, inst=inst, traj=traj, order=order, count=count,
                abs_ins_err=abs_ins_err)


# =============================================================================================
# Validation of submission structure
# =============================================================================================

def validate_submission(rows, test_eps):
    errors = []
    ids = [r["id"] for r in rows]
    exp_ids = [e["id"] for e in test_eps]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    if set(ids) != set(exp_ids):
        errors.append("id set mismatch")
    cand = {e["id"]: set(e["cids"].tolist()) for e in test_eps}
    n_actions = []
    for r in rows:
        try:
            acts = json.loads(r["portfolio_json"])
        except Exception:
            errors.append(f"{r['id']}: invalid json")
            continue
        if not isinstance(acts, list) or len(acts) > 256:
            errors.append(f"{r['id']}: not a list or >256 actions")
            continue
        n_actions.append(len(acts))
        seen = set()
        last = None
        for a in acts:
            if not isinstance(a, dict) or set(a.keys()) != {"action_id", "instruction_minute", "trajectory"}:
                errors.append(f"{r['id']}: bad action keys")
                continue
            aid = a["action_id"]
            if aid not in cand.get(r["id"], set()):
                errors.append(f"{r['id']}: action {aid} not a candidate")
            if aid in seen:
                errors.append(f"{r['id']}: duplicate action {aid}")
            seen.add(aid)
            im = a["instruction_minute"]
            if not (isinstance(im, (int, float)) and np.isfinite(im) and -90 <= im <= 30):
                errors.append(f"{r['id']}: bad instruction minute")
            segs = a["trajectory"]
            if not isinstance(segs, list) or not (1 <= len(segs) <= 32):
                errors.append(f"{r['id']}: bad segment count")
                continue
            prev_t1 = -1.0
            for s in segs:
                if set(s.keys()) != {"t0", "t1", "level0", "level1"}:
                    errors.append(f"{r['id']}: bad segment keys")
                    break
                vals = [s["t0"], s["t1"], s["level0"], s["level1"]]
                if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in vals):
                    errors.append(f"{r['id']}: non-finite segment")
                    break
                if not (0 <= s["t0"] < s["t1"] <= 30):
                    errors.append(f"{r['id']}: segment time out of range")
                if s["t0"] < prev_t1:
                    errors.append(f"{r['id']}: overlapping/unsorted segments")
                if not (-10000 <= s["level0"] <= 10000 and -10000 <= s["level1"] <= 10000):
                    errors.append(f"{r['id']}: level out of range")
                prev_t1 = s["t1"]
            key = (float(im), str(aid))
            if last is not None and key < last:
                errors.append(f"{r['id']}: actions not canonically sorted")
            last = key
    return errors, n_actions


# =============================================================================================
# Entry points
# =============================================================================================

def run_cv(public_dir, out_path):
    t_start = time.time()
    _, train_eps = load_split(public_dir, "train")
    days = sorted(set(e["day"] for e in train_eps))
    folds = [[d for i, d in enumerate(days) if i % NFOLD == f] for f in range(NFOLD)]
    rows = []
    action_rows = []
    for f, vdays in enumerate(folds):
        tr = [e for e in train_eps if e["day"] not in vdays]
        va = [e for e in train_eps if e["day"] in vdays]
        for e in train_eps:
            e.pop("kidx", None)
        preds = run_pipeline(tr, va, log=lambda s, f=f: print(f"[fold {f}] {s}", flush=True))
        key_pos = defaultdict(int)
        for e in tr:
            for t, d_, y_ in zip(e["toks"], e["dirs"], e["y"]):
                if y_:
                    key_pos[(t, int(d_))] += 1
        for e in va:
            pm = {a["action_id"]: a for a in preds[e["id"]]}
            idx = {c: i for i, c in enumerate(e["cids"])}
            for a in e["target"]:
                i = idx[a["action_id"]]
                p = pm.get(a["action_id"])
                rec_a = {"fold": f, "id": e["id"], "direction": int(e["dirs"][i]),
                         "token_train_acc": key_pos.get((e["toks"][i], int(e["dirs"][i])), 0),
                         "true_ins": a["instruction_minute"], "selected": int(p is not None),
                         "abs_ins_err": abs(p["instruction_minute"] - a["instruction_minute"]) if p else np.nan,
                         "traj_official": float(_traj_sim(true_grid(p)[None], true_grid(a)[None], METRIC_VARIANTS["official"])[0]) if p else np.nan}
                action_rows.append(rec_a)
        for e in va:
            r = {"fold": f, "id": e["id"], "day": e["day"], "period": e["period"], "npred": len(preds[e["id"]]),
                 "ntrue": len(e["target"]), "ncand": len(e["cids"])}
            for vn, v in METRIC_VARIANTS.items():
                s = score_episode(preds[e["id"]], e["target"], v)
                r[vn] = s["score"]
                if vn == "official":
                    r.update({k: s[k] for k in ("f1", "prec", "rec", "inst", "traj", "order", "count", "abs_ins_err")})
            rows.append(r)
        R = pd.DataFrame(rows)
        print(f"[fold {f}] official={R[R.fold == f].official.mean():.4f} f1={R[R.fold == f].f1.mean():.4f} "
              f"t={time.time()-t_start:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out_path.replace(".json", "_rows.csv"), index=False)
    pd.DataFrame(action_rows).to_csv(out_path.replace(".json", "_actions.csv"), index=False)
    summary = {}
    for vn in METRIC_VARIANTS:
        fm = R.groupby("fold")[vn].mean()
        summary[vn] = dict(folds=[round(x, 5) for x in fm.tolist()], mean=round(fm.mean(), 5),
                           std=round(fm.std(ddof=0), 5), worst=round(fm.min(), 5), median=round(fm.median(), 5))
    comp = R[["f1", "prec", "rec", "inst", "traj", "order", "count", "abs_ins_err"]].mean().round(5).to_dict()
    out = dict(scheme=f"{NFOLD}-fold GroupKFold by day_group (sorted day tokens, i % {NFOLD})", summary=summary,
               components=comp, runtime_s=round(time.time() - t_start, 1))
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


def run_submission(public_dir, submission_out):
    t_start = time.time()
    _, train_eps = load_split(public_dir, "train")
    test_df, test_eps = load_split(public_dir, "test")
    preds = run_pipeline(train_eps, test_eps, log=lambda s: print(s, flush=True))
    rows = [{"id": e["id"], "portfolio_json": json.dumps(preds[e["id"]], separators=(",", ":"), sort_keys=True)}
            for e in test_eps]
    errors, n_actions = validate_submission(rows, test_eps)
    if errors:
        raise RuntimeError(f"submission validation failed: {errors[:10]}")
    out_dir = os.path.dirname(os.path.abspath(submission_out))
    os.makedirs(out_dir, exist_ok=True)
    with open(submission_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "portfolio_json"])
        w.writeheader()
        w.writerows(rows)
    info = dict(train_rows=len(train_eps), test_rows=len(test_eps), runtime_s=round(time.time() - t_start, 1),
                predicted_actions=dict(mean=float(np.mean(n_actions)), min=int(np.min(n_actions)),
                                       max=int(np.max(n_actions)), median=float(np.median(n_actions))),
                validation_errors=len(errors))
    print(json.dumps(info, indent=2))
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit("Usage: python3 solution.py <public_dir> <submission_out> [--cv]")
    if "--cv" in sys.argv:
        run_cv(args[0], args[1])
    else:
        run_submission(args[0], args[1])


if __name__ == "__main__":
    main()
