#!/usr/bin/env python3
"""
EraMesh: Historical Order and Boundary Recovery.

Usage:
  python3 solution.py <public_dir> <submission.csv>
  python3 solution.py <public_dir> <cv.json> --cv

Approach
--------
The supplied `distance_mesh` is an unsigned, bucket-coarsened set of pairwise
time distances.  Every bucket is an interval [lo, hi] on |t_a - t_b|, so a
candidate chronological permutation is *consistent* with the mesh iff the
induced system of difference constraints

    x_{pi(j)} - x_{pi(i)} in [lo_ij, hi_ij]      (observed pairs)
    x_{pi(i)} <= x_{pi(i+1)}                     (monotone positions)

is satisfiable.  Feasibility of a difference-constraint system is exactly the
absence of negative cycles, which is decided by a min-plus (Floyd-Warshall)
closure in O(n^3); the total negative-diagonal mass is a smooth infeasibility
measure suitable for local search.  `precedence_cues` are hard directed
constraints (verified genuine on train) and additionally resolve the global
reflection symmetry of an unsigned metric.

  1) Per row, local search over permutations minimises mesh infeasibility plus
     cue violations, collecting a diverse set of *feasible* candidate orders.
  2) The mesh usually leaves several feasible orders.  A learned event-level
     "chronological score" (LightGBM on structural features + Ridge on masked
     text) ranks them; a temperature-weighted consensus over the candidate set
     yields the order maximising expected pairwise concordance.
  3) With the order fixed, the same min-plus closure yields tight lower/upper
     bounds on the time gap of *every* pair (including unobserved ones).  Those
     bounds drive a LightGBM same-era model and a LightGBM break-boundary model.
  4) Dynamic programming decodes exactly `era_count` contiguous nonempty eras.
  5) The exact official metric and the submission grammar are validated in-code.

Only the supplied public train inputs/labels and public test inputs are used.
All randomness is seeded per row, so sharding across processes does not change
the output.
"""

import os
import sys
import re
import json
import math
import random
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# ----------------------------- deterministic config --------------------------

SEED = 20260917
LGB_THREADS = 6

# Bucket edges are published in metadata.json ("distance_cut_years").
LO = np.array([0., 3., 10., 25., 60., 150., 300.])
HI = np.array([3., 10., 25., 60., 150., 300., 1e7])      # feasibility (open top)
HIC = np.array([3., 10., 25., 60., 150., 300., 900.])    # bounded, for features
INF = 1e9

CUE_WEIGHT = 500.0          # cue violations dominate mesh infeasibility
N_RESTART = 26              # local-search restarts per row
WANT_CANDS = 20             # distinct feasible orders to collect
PATIENCE = 8                # restarts without a new distinct solution
CONSENSUS_TEMP = 1.0        # temperature for candidate-weighted consensus
POINT_W = 0.60              # structural vs text chronological score
BOUNDARY_W = 1.0            # weight of the break model inside the DP

POINT_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31,
                    min_data_in_leaf=25, feature_fraction=0.8, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=3.0, verbose=-1,
                    num_threads=LGB_THREADS, seed=11, deterministic=True,
                    force_row_wise=True)
SAME_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                   min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=5.0, verbose=-1,
                   num_threads=LGB_THREADS, seed=13, deterministic=True,
                   force_row_wise=True)
BND_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbose=-1,
                  num_threads=LGB_THREADS, seed=17, deterministic=True,
                  force_row_wise=True)
POINT_ROUNDS = 300
SAME_ROUNDS = 400
BND_ROUNDS = 300

EVENT_RE = re.compile(r"(q\d+)\{cat=([^;]+);dur=([^;]+);src=([^;]*);title=(.*?);text=(.*)\}")
SRC_RE = re.compile(r"(s\d+)\{kind=([^;]+);label=(.*)")
CATS = ["colonization", "culture", "disaster", "economy", "founding",
        "independence", "migration", "politics", "religion", "war"]
DURS = ["d0", "d1", "d2", "d3"]
KINDS = ["academic", "archive", "encyclopedia", "gov", "museum", "primary", "reference"]
BANDS = ["easy", "medium", "hard"]
NEW_ERA = ["colonization", "independence", "founding"]


def seed_everything(seed=SEED):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------- parsing -----------------------------------

def parse_row(row):
    ev = {}
    for part in str(row["event_cards"]).split(" || "):
        m = EVENT_RE.fullmatch(part.strip())
        if m:
            q, cat, dur, src, title, text = m.groups()
            ev[q] = {"cat": cat, "dur": dur,
                     "src": src.split(",") if src else [],
                     "title": title, "text": text}
    src = {}
    for part in str(row["source_ledger"]).split(" || "):
        m = SRC_RE.fullmatch(part.strip())
        if m:
            src[m.group(1)] = {"kind": m.group(2), "label": m.group(3)}
    mesh = {}
    for tok in str(row["distance_mesh"]).split():
        m = re.fullmatch(r"(q\d+)~(q\d+):g(\d+)", tok)
        if m:
            a, b, g = m.groups()
            mesh[tuple(sorted((a, b)))] = int(g)
    cues = [tuple(x.split("<")) for x in str(row["precedence_cues"]).split() if "<" in x]
    return ev, src, mesh, cues


def gold_parts(prog):
    toks = str(prog).split()
    order = [x for x in toks if x != "BREAK"]
    segs, cur = [], []
    for x in toks:
        if x == "BREAK":
            segs.append(cur)
            cur = []
        else:
            cur.append(x)
    segs.append(cur)
    return order, segs


# ------------------- mesh feasibility / candidate generation ------------------

class MeshRow(object):
    """Difference-constraint view of one row's distance mesh."""

    __slots__ = ("qs", "n", "qi", "mi", "mj", "mg", "lo", "hi", "cu", "base")

    def __init__(self, qs, mesh, cues):
        self.qs = qs
        self.n = n = len(qs)
        self.qi = qi = {q: i for i, q in enumerate(qs)}
        items = sorted(mesh.items())
        self.mi = np.array([qi[a] for (a, b), g in items], dtype=np.intp)
        self.mj = np.array([qi[b] for (a, b), g in items], dtype=np.intp)
        self.mg = np.array([g for (a, b), g in items], dtype=np.intp)
        self.lo = LO[self.mg]
        self.hi = HI[self.mg]
        self.cu = np.array([[qi[a], qi[b]] for a, b in cues if a in qi and b in qi],
                           dtype=np.intp).reshape(-1, 2)
        B = np.full((n, n), INF)
        np.fill_diagonal(B, 0.0)
        if n > 1:
            B[np.arange(1, n), np.arange(0, n - 1)] = 0.0
        self.base = B


def mesh_cost(R, perm):
    """Negative-cycle mass of the induced system + weighted cue violations."""
    n = R.n
    pos = np.empty(n, dtype=np.intp)
    pos[perm] = np.arange(n)
    W = R.base.copy()
    if len(R.mi):
        pi, pj = pos[R.mi], pos[R.mj]
        a = np.minimum(pi, pj)
        b = np.maximum(pi, pj)
        W[a, b] = R.hi
        W[b, a] = -R.lo
    for k in range(n):
        W = np.minimum(W, W[:, k:k + 1] + W[k:k + 1, :])
    d = np.diag(W)
    c = float(-d[d < 0].sum())
    if len(R.cu):
        c += CUE_WEIGHT * float((pos[R.cu[:, 0]] > pos[R.cu[:, 1]]).sum())
    return c


def local_opt(R, perm, maxpass=30):
    perm = list(perm)
    cur = mesh_cost(R, perm)
    n = R.n
    for _ in range(maxpass):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = perm.copy()
                cand.insert(j, cand.pop(i))
                v = mesh_cost(R, cand)
                if v < cur - 1e-9:
                    perm, cur = cand, v
                    improved = True
                    if cur <= 1e-9:
                        return perm, cur
        if not improved:
            break
    return perm, cur


def gen_candidates(R, rng):
    """Collect distinct permutations attaining the lowest infeasibility found."""
    feas, seen, bestv, since = [], set(), 1e18, 0
    for rs in range(N_RESTART):
        if rs == 0:
            p = list(range(R.n))
        elif feas and rs % 2 == 1:
            p = list(feas[int(rng.integers(len(feas)))])
            for _ in range(int(rng.integers(2, 5))):
                i, j = int(rng.integers(R.n)), int(rng.integers(R.n))
                p.insert(j, p.pop(i))
        else:
            p = list(rng.permutation(R.n))
        p, v = local_opt(R, p)
        t = tuple(p)
        if v < bestv - 1e-9:
            bestv, feas, seen, since = v, [], set(), 0
        if v <= bestv + 1e-9 and t not in seen:
            seen.add(t)
            feas.append(t)
            since = 0
        else:
            since += 1
        if len(feas) >= WANT_CANDS or (since >= PATIENCE and len(feas) >= 4):
            break
    return bestv, [list(p) for p in feas]


def _cand_worker(args):
    split_tag, rid, qs, mesh, cues = args
    R = MeshRow(qs, mesh, cues)
    rng = np.random.default_rng(1000003 * (3 if split_tag == "train" else 5)
                                + 7919 * int(rid) + 13)
    bestv, cands = gen_candidates(R, rng)
    return rid, cands, bestv


def build_candidates(df, split_tag, workers=None):
    """Per-row RNG seeding keeps the result identical under any sharding."""
    jobs = []
    for _, row in df.iterrows():
        ev, src, mesh, cues = parse_row(row)
        jobs.append((split_tag, int(row["id"]), sorted(ev), mesh, cues))
    out = {}
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 1) - 1))
    if workers > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker) as pool:
                for rid, cands, bestv in pool.imap_unordered(_cand_worker, jobs, chunksize=4):
                    out[rid] = {"cands": cands, "bestv": bestv}
        except Exception:
            out = {}
    if not out:
        for job in jobs:
            rid, cands, bestv = _cand_worker(job)
            out[rid] = {"cands": cands, "bestv": bestv}
    return {k: out[k] for k in sorted(out)}


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def fw_bounds(order_idx, mi, mj, mg, n):
    """Tightest bounds implied by the mesh once the permutation is fixed.

    Returns W with W[i, j] = max attainable (x_j - x_i) in position space;
    -W[j, i] is therefore the minimum attainable gap.
    """
    pos = np.empty(n, dtype=np.intp)
    pos[np.asarray(order_idx, dtype=np.intp)] = np.arange(n)
    W = np.full((n, n), INF)
    np.fill_diagonal(W, 0.0)
    if n > 1:
        W[np.arange(1, n), np.arange(0, n - 1)] = 0.0
    if len(mi):
        pi, pj = pos[mi], pos[mj]
        a = np.minimum(pi, pj)
        b = np.maximum(pi, pj)
        W[a, b] = HIC[mg]
        W[b, a] = -LO[mg]
    for k in range(n):
        W = np.minimum(W, W[:, k:k + 1] + W[k:k + 1, :])
    return W


def gap_bounds(W):
    """Min/max attainable gaps, clipped to a sane range.

    On the rare row where no fully feasible permutation is found the closure can
    emit negative or non-finite bounds; clipping keeps the downstream log1p
    features well defined.  Feasible rows are unaffected.
    """
    # `lo` is a genuine implied lower bound and is deliberately NOT capped: long
    # chronologies legitimately accumulate gaps far beyond any single bucket.
    # Only `hi` is clipped, which matters solely on the rare row where no fully
    # feasible permutation is found (there the closure can emit negative or
    # non-finite upper bounds).  Feasible rows are bit-identical either way.
    lo = np.nan_to_num(np.maximum(-W.T, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    hi = np.nan_to_num(np.minimum(W, 900.0), nan=900.0, posinf=900.0, neginf=0.0)
    return lo, np.clip(hi, 0.0, 900.0)


# ------------------------------ row features ---------------------------------

def ev_text(e, src):
    kinds = " ".join("KIND_" + src[s]["kind"] for s in e["src"] if s in src)
    labels = " ".join(src[s]["label"] for s in e["src"] if s in src)
    t = (e["title"] + " " + e["text"] + " " + labels).lower()
    t = re.sub(r"v\d+", " VAL ", t).replace("<mask>", " MSK ").replace("<time>", " TMK ")
    return "CAT_" + e["cat"] + " DUR_" + e["dur"] + " " + kinds + " " + t


def event_features(row, ev, src, mesh):
    qs = sorted(ev)
    n = len(qs)
    qi = {q: i for i, q in enumerate(qs)}
    G = np.full((n, n), -1.0)
    for (a, b), g in mesh.items():
        G[qi[a], qi[b]] = g
        G[qi[b], qi[a]] = g
    band = BANDS.index(row["visibility_band"])
    ec = int(row["era_count"])
    F = []
    for q in qs:
        e = ev[q]
        i = qi[q]
        obs = G[i][G[i] >= 0]
        kinds = [src[s]["kind"] for s in e["src"] if s in src]
        txt = e["title"] + " " + e["text"]
        f = [CATS.index(e["cat"]) if e["cat"] in CATS else -1]
        f += [int(e["cat"] == c) for c in CATS]
        f += [DURS.index(e["dur"]) if e["dur"] in DURS else -1]
        f += [int(e["dur"] == d) for d in DURS]
        f += [len(e["src"])] + [int(k in kinds) for k in KINDS]
        f += [len(txt.split()), txt.count("<mask>"), len(re.findall(r"v\d+", txt)),
              txt.count("<time>"), len(e["title"].split()), e["title"].count("<mask>")]
        f += [float(obs.mean()) if len(obs) else -1.0,
              float(np.median(obs)) if len(obs) else -1.0,
              float((obs >= 5).mean()) if len(obs) else -1.0,
              float((obs <= 1).mean()) if len(obs) else -1.0,
              float((obs >= 3).mean()) if len(obs) else -1.0,
              len(obs) / max(1, n - 1),
              float(obs.max()) if len(obs) else -1.0,
              float(obs.min()) if len(obs) else -1.0]
        f += [n, ec, band]
        F.append(f)
    return qs, np.asarray(F, dtype=np.float32)


def prepare_rows(df, cand_map, with_gold):
    rows = []
    for _, r in df.iterrows():
        ev, src, mesh, cues = parse_row(r)
        qs, F = event_features(r, ev, src, mesh)
        qi = {q: i for i, q in enumerate(qs)}
        items = sorted(mesh.items())
        R = dict(id=int(r["id"]), band=r["visibility_band"], ec=int(r["era_count"]),
                 n=len(qs), qs=qs, ev=ev, src=src, F=F,
                 texts=[ev_text(ev[q], src) for q in qs],
                 mi=np.array([qi[a] for (a, b), g in items], dtype=np.intp),
                 mj=np.array([qi[b] for (a, b), g in items], dtype=np.intp),
                 mg=np.array([g for (a, b), g in items], dtype=np.intp),
                 cands=cand_map[int(r["id"])]["cands"])
        if not R["cands"]:
            R["cands"] = [list(range(len(qs)))]
        if with_gold:
            go, gsegs = gold_parts(r["chronicle_program"])
            R["go"], R["gsegs"] = go, gsegs
        rows.append(R)
    return rows


# ------------------------- candidate ranking / consensus ----------------------

def pointwise_targets(R):
    pos = {q: i for i, q in enumerate(R["go"])}
    n = R["n"]
    return np.array([pos[q] / max(1, n - 1) for q in R["qs"]], dtype=np.float32)


def pick_order(R, escore, temp=CONSENSUS_TEMP):
    n = R["n"]
    w = 2.0 * np.arange(n) - (n - 1)
    s = np.array([float(np.dot(escore[np.asarray(c)], w)) for c in R["cands"]])
    best = list(R["cands"][int(np.argmax(s))])
    if temp <= 0 or len(R["cands"]) == 1:
        return best
    wt = np.exp((s - s.max()) / temp)
    wt /= wt.sum()
    M = np.zeros((n, n))
    for wk, c in zip(wt, R["cands"]):
        pos = np.empty(n, dtype=np.intp)
        pos[np.asarray(c)] = np.arange(n)
        M += wk * (pos[:, None] < pos[None, :])
    S = M - M.T
    iu = np.triu_indices(n, 1)

    def obj(p):
        idx = np.asarray(p)
        return float(S[np.ix_(idx, idx)][iu].sum())

    cur = obj(best)
    for _ in range(12):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c2 = best.copy()
                c2.insert(j, c2.pop(i))
                v = obj(c2)
                if v > cur + 1e-12:
                    best, cur = c2, v
                    improved = True
        if not improved:
            break
    return best


# -------------------------- era pair / boundary features ----------------------

def _row_context(R, order):
    n = R["n"]
    lo, hi = gap_bounds(fw_bounds(order, R["mi"], R["mj"], R["mg"], n))
    qs = R["qs"]
    cats = [R["ev"][qs[k]]["cat"] for k in order]
    srcs = [set(R["ev"][qs[k]]["src"]) for k in order]
    kinds = [set(R["src"][s]["kind"] for s in R["ev"][qs[k]]["src"] if s in R["src"])
             for k in order]
    pos = np.empty(n, dtype=np.intp)
    pos[np.asarray(order)] = np.arange(n)
    mg = np.full((n, n), -1.0)
    for k in range(len(R["mi"])):
        a, b = pos[R["mi"][k]], pos[R["mj"][k]]
        mg[a, b] = mg[b, a] = R["mg"][k]
    return lo, hi, cats, srcs, kinds, mg


def seg_features(R, order):
    n = R["n"]
    lo, hi, cats, srcs, kinds, mesh_g = _row_context(R, order)
    step_lo = np.array([lo[i, i + 1] for i in range(n - 1)] + [0.0])
    cum_lo = np.concatenate([[0.0], np.cumsum(step_lo[:n - 1])])
    newera = np.array([1.0 if c in NEW_ERA else 0.0 for c in cats])
    cn_new = np.concatenate([[0.0], np.cumsum(newera)])
    band = BANDS.index(R["band"])
    feats, idx = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            seg_lo, seg_hi = lo[i, j], hi[i, j]
            inner = step_lo[i:j]
            feats.append([
                d, d / (n - 1), i / (n - 1), j / (n - 1), n, R["ec"], band,
                np.log1p(seg_lo), np.log1p(seg_hi), np.log1p(0.5 * (seg_lo + seg_hi)),
                mesh_g[i, j], int(mesh_g[i, j] >= 0),
                float(inner.max()) if len(inner) else 0.0,
                np.log1p(float(inner.max()) if len(inner) else 0.0),
                float(inner.sum()),
                float(np.sort(inner)[-2]) if len(inner) > 1 else 0.0,
                cum_lo[j] - cum_lo[i],
                cn_new[j] - cn_new[i + 1] if j > i + 1 else 0.0,
                cn_new[j + 1] - cn_new[i + 1],
                int(cats[i] == cats[j]),
                CATS.index(cats[i]) if cats[i] in CATS else -1,
                CATS.index(cats[j]) if cats[j] in CATS else -1,
                int(cats[j] in NEW_ERA), int(cats[i] in NEW_ERA),
                len(srcs[i] & srcs[j]), len(srcs[i] | srcs[j]),
                len(srcs[i] & srcs[j]) / max(1, min(len(srcs[i]), len(srcs[j]))),
                len(kinds[i] & kinds[j]),
                (n - 1) / max(1, R["ec"]),
                seg_lo / max(1.0, cum_lo[n - 1]),
            ])
            idx.append((i, j))
    return np.asarray(feats, dtype=np.float32), idx


def bnd_features(R, order):
    n = R["n"]
    lo, hi, cats, srcs, kinds, mg = _row_context(R, order)
    step = np.array([lo[i, i + 1] for i in range(n - 1)])
    steph = np.array([hi[i, i + 1] for i in range(n - 1)])
    mx = max(1.0, float(step.max()) if len(step) else 1.0)
    rk = np.argsort(np.argsort(-step, kind="mergesort")) if len(step) else np.zeros(0)
    band = BANDS.index(R["band"])
    F = []
    for i in range(n - 1):
        F.append([i, i / (n - 2) if n > 2 else 0.5, n, R["ec"], band,
                  (R["ec"] - 1) / (n - 1),
                  step[i], np.log1p(step[i]), steph[i], np.log1p(steph[i]),
                  step[i] / mx, rk[i], rk[i] / max(1, n - 2), int(rk[i] < R["ec"] - 1),
                  mg[i, i + 1], int(mg[i, i + 1] >= 0),
                  CATS.index(cats[i]) if cats[i] in CATS else -1,
                  CATS.index(cats[i + 1]) if cats[i + 1] in CATS else -1,
                  int(cats[i] == cats[i + 1]), int(cats[i + 1] in NEW_ERA),
                  int(cats[i] in NEW_ERA),
                  len(srcs[i] & srcs[i + 1]), len(kinds[i] & kinds[i + 1]),
                  float(np.log1p(lo[0, i + 1])), float(np.log1p(lo[i + 1, n - 1]))])
    return np.asarray(F, dtype=np.float32)


def seg_targets(R, order):
    qs = R["qs"]
    gs = {q: k for k, s in enumerate(R["gsegs"]) for q in s}
    n = R["n"]
    return np.asarray([int(gs[qs[order[i]]] == gs[qs[order[j]]])
                       for i in range(n) for j in range(i + 1, n)], dtype=np.float32)


def bnd_targets(R, order):
    qs = R["qs"]
    gs = {q: k for k, s in enumerate(R["gsegs"]) for q in s}
    n = R["n"]
    return np.asarray([int(gs[qs[order[i]]] != gs[qs[order[i + 1]]])
                       for i in range(n - 1)], dtype=np.float32)


# ------------------------------ segmentation DP -------------------------------

def dp_segments(n, delta, bound_lp, K):
    """Split positions 0..n-1 into exactly K contiguous nonempty eras."""
    cum = np.zeros((n, n + 1))
    for l in range(n):
        c = 0.0
        for r in range(l + 1, n + 1):
            j = r - 1
            if j > l:
                c += delta[l:j, j].sum()
            cum[l, r] = c
    K = max(1, min(int(K), n))
    NEG = -1e18
    dp = np.full((K + 1, n + 1), NEG)
    prev = np.full((K + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, K + 1):
        for r in range(k, n + 1):
            best, bl = NEG, -1
            for l in range(k - 1, r):
                v = dp[k - 1, l] + cum[l, r]
                if l > 0:
                    v += bound_lp[l - 1]
                if v > best:
                    best, bl = v, l
            dp[k, r] = best
            prev[k, r] = bl
    bounds, r = [], n
    for k in range(K, 0, -1):
        l = int(prev[k, r])
        bounds.append((l, r))
        r = l
    bounds.reverse()
    return bounds


# ------------------------------ official metric -------------------------------

def score_row(pred_order, pred_segs, gold_order, gold_segs):
    """Exact metric: 0.50 * nonneg Kendall + 0.40 * nonneg pair MCC + 0.10 * adj F1."""
    n = len(gold_order)
    posp = {q: i for i, q in enumerate(pred_order)}
    if n <= 1:
        O = 1.0
    else:
        agree = 0
        tot = n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                agree += int(posp[gold_order[i]] < posp[gold_order[j]])
        O = max(0.0, 2.0 * agree / tot - 1.0)

    gs = {q: k for k, s in enumerate(gold_segs) for q in s}
    ps = {q: k for k, s in enumerate(pred_segs) for q in s}
    tp = tn = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = gold_order[i], gold_order[j]
            sg, sp = gs[a] == gs[b], ps[a] == ps[b]
            if sg and sp:
                tp += 1
            elif (not sg) and (not sp):
                tn += 1
            elif sp:
                fp += 1
            else:
                fn += 1
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    M = max(0.0, (tp * tn - fp * fn) / den) if den > 0 else 0.0

    def edges(segs):
        ed = set()
        if not segs or not segs[0]:
            return ed
        ed.add(("START", segs[0][0], "START"))
        for s in segs:
            for a, b in zip(s, s[1:]):
                ed.add((a, b, "WITHIN"))
        for s1, s2 in zip(segs[:-1], segs[1:]):
            ed.add((s1[-1], s2[0], "BREAK"))
        ed.add((segs[-1][-1], "END", "END"))
        return ed

    eg, ep = edges(gold_segs), edges(pred_segs)
    A = 2.0 * len(eg & ep) / max(1, len(eg) + len(ep))
    return 0.50 * O + 0.40 * M + 0.10 * A, O, M, A


# --------------------- solver-visible component proxy groups ------------------

def row_shingles(row):
    t = (str(row["event_cards"]) + " " + str(row["source_ledger"])).lower()
    t = re.sub(r"q\d+", "Q", t)
    t = re.sub(r"v\d+", "V", t)
    t = t.replace("<mask>", "MASK").replace("<time>", "TIME")
    toks = re.findall(r"[a-z]+|mask|time|d[0-3]|q|v", t)
    if len(toks) < 5:
        return set()
    return set(zip(*(toks[i:] for i in range(5))))


def make_groups(df, threshold=0.45):
    sh = [row_shingles(r) for _, r in df.iterrows()]
    n = len(sh)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if len(sh[i] & sh[j]) / max(1, len(sh[i] | sh[j])) >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    roots, g = {}, np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        g[i] = roots[r]
    return g


# --------------------------------- models -------------------------------------

class EraMeshModel(object):
    def fit(self, rows):
        Xp = np.vstack([R["F"] for R in rows])
        yp = np.concatenate([pointwise_targets(R) for R in rows])
        self.m_point = lgb.train(POINT_PARAMS, lgb.Dataset(Xp, label=yp),
                                 num_boost_round=POINT_ROUNDS)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                   max_features=40000, sublinear_tf=True)
        Xt = self.vec.fit_transform([t for R in rows for t in R["texts"]])
        self.m_text = Ridge(alpha=2.0, random_state=SEED)
        self.m_text.fit(Xt, yp)

        orders = self.predict_orders(rows)
        Xs = np.vstack([seg_features(R, o)[0] for R, o in zip(rows, orders)])
        ys = np.concatenate([seg_targets(R, o) for R, o in zip(rows, orders)])
        self.m_same = lgb.train(SAME_PARAMS, lgb.Dataset(Xs, label=ys),
                                num_boost_round=SAME_ROUNDS)
        Xb = np.vstack([bnd_features(R, o) for R, o in zip(rows, orders)])
        yb = np.concatenate([bnd_targets(R, o) for R, o in zip(rows, orders)])
        self.m_bnd = lgb.train(BND_PARAMS, lgb.Dataset(Xb, label=yb),
                               num_boost_round=BND_ROUNDS)
        return self

    def event_scores(self, rows):
        Xa = np.vstack([R["F"] for R in rows])
        pa = self.m_point.predict(Xa)
        ta = self.m_text.predict(self.vec.transform([t for R in rows for t in R["texts"]]))
        out, c = [], 0
        for R in rows:
            n = R["n"]
            out.append(POINT_W * pa[c:c + n] + (1.0 - POINT_W) * ta[c:c + n])
            c += n
        return out

    def predict_orders(self, rows):
        es = self.event_scores(rows)
        return [pick_order(R, e) for R, e in zip(rows, es)]

    def predict(self, rows):
        """Returns (order_idx, segment bounds) per row."""
        orders = self.predict_orders(rows)
        out = []
        for R, o in zip(rows, orders):
            n = R["n"]
            f, idx = seg_features(R, o)
            p = np.clip(self.m_same.predict(f), 1e-4, 1 - 1e-4)
            delta = np.zeros((n, n))
            for (i, j), pv in zip(idx, p):
                delta[i, j] = math.log(pv / (1.0 - pv))
            blp = np.zeros(n)
            if BOUNDARY_W > 0 and n > 1:
                pb = np.clip(self.m_bnd.predict(bnd_features(R, o)), 1e-4, 1 - 1e-4)
                blp[:n - 1] = BOUNDARY_W * np.log(pb / (1.0 - pb))
            out.append((o, dp_segments(n, delta, blp, R["ec"])))
        return out


def program_of(R, order, bounds):
    qs = R["qs"]
    segs = [[qs[k] for k in order[l:r]] for l, r in bounds]
    toks = []
    for si, seg in enumerate(segs):
        toks.extend(seg)
        if si < len(segs) - 1:
            toks.append("BREAK")
    return " ".join(toks), segs


# --------------------------- submission validation ----------------------------

def validate_submission(df, submission):
    errors = []
    if list(submission.columns) != ["id", "chronicle_program"]:
        errors.append("wrong columns/order")
    if len(submission) != len(df):
        errors.append("wrong row count: %d" % len(submission))
    ids = [int(x) for x in submission["id"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate IDs")
    if set(ids) != set(int(x) for x in df["id"]):
        errors.append("ID set mismatch")

    era = dict(zip(df["id"].astype(int), df["era_count"].astype(int)))
    nev = dict(zip(df["id"].astype(int), df["event_count"].astype(int)))
    handles = {}
    for _, row in df.iterrows():
        ev, _, _, _ = parse_row(row)
        handles[int(row["id"])] = set(ev)

    for rid, prog in zip(submission["id"].astype(int), submission["chronicle_program"]):
        prog = str(prog)
        toks = prog.split()
        if not toks:
            errors.append("%d: empty program" % rid)
            continue
        if prog != prog.strip() or "  " in prog:
            errors.append("%d: bad spacing" % rid)
        if "BREAK BREAK" in prog:
            errors.append("%d: consecutive BREAK" % rid)
        if toks[0] == "BREAK" or toks[-1] == "BREAK":
            errors.append("%d: empty segment at edge" % rid)
        if toks.count("BREAK") != era[rid] - 1:
            errors.append("%d: wrong BREAK count" % rid)
        hs = [t for t in toks if t != "BREAK"]
        if len(hs) != nev[rid]:
            errors.append("%d: wrong handle count" % rid)
        if len(hs) != len(set(hs)):
            errors.append("%d: repeated handle" % rid)
        if set(hs) != handles[rid]:
            errors.append("%d: handle set mismatch" % rid)
    return errors


# ----------------------------------- runners ----------------------------------

def run_cv(public_dir, out_json, n_splits=4):
    seed_everything()
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    cands = build_candidates(train, "train")
    rows = prepare_rows(train, cands, with_gold=True)
    groups = make_groups(train, 0.45)

    fold_scores, band_scores, comp = [], defaultdict(list), defaultdict(list)
    slice_scores = defaultdict(list)
    train_fit_scores = []
    for fold, (tri, vai) in enumerate(GroupKFold(n_splits).split(np.arange(len(rows)),
                                                                 groups=groups)):
        TR = [rows[i] for i in tri]
        VA = [rows[i] for i in vai]
        model = EraMeshModel().fit(TR)

        per_band = defaultdict(list)
        for R, (o, b) in zip(VA, model.predict(VA)):
            _, segs = program_of(R, o, b)
            s, O, M, A = score_row([R["qs"][k] for k in o], segs, R["go"], R["gsegs"])
            per_band[R["band"]].append(s)
            comp["O"].append(O)
            comp["M"].append(M)
            comp["A"].append(A)
            nb = "06-08" if R["n"] <= 8 else ("09-12" if R["n"] <= 12 else "13-18")
            slice_scores[nb].append(s)
        fs = float(np.mean([np.mean(per_band[b]) for b in BANDS]))
        fold_scores.append(fs)
        for b in BANDS:
            band_scores[b].append(float(np.mean(per_band[b])))

        tb = defaultdict(list)
        for R, (o, b) in zip(TR, model.predict(TR)):
            _, segs = program_of(R, o, b)
            s, _, _, _ = score_row([R["qs"][k] for k in o], segs, R["go"], R["gsegs"])
            tb[R["band"]].append(s)
        train_fit_scores.append(float(np.mean([np.mean(tb[b]) for b in BANDS])))
        print("[fold %d] val=%.5f train=%.5f  %s" %
              (fold, fs, train_fit_scores[-1],
               "  ".join("%s=%.4f" % (b, np.mean(per_band[b])) for b in BANDS)), flush=True)

    report = {
        "validation": {
            "scheme": "4-fold GroupKFold on solver-visible row-component proxy "
                      "(five-token shingle Jaccard >= 0.45); fold score = equal "
                      "mean of easy/medium/hard row means, exact official metric",
            "fold_scores": [round(x, 5) for x in fold_scores],
            "mean": round(float(np.mean(fold_scores)), 5),
            "std": round(float(np.std(fold_scores)), 5),
            "worst": round(float(np.min(fold_scores)), 5),
            "train_fit_mean": round(float(np.mean(train_fit_scores)), 5),
            "by_band": {b: round(float(np.mean(band_scores[b])), 5) for b in BANDS},
            "components": {k: round(float(np.mean(v)), 5) for k, v in sorted(comp.items())},
            "by_event_count": {k: round(float(np.mean(v)), 5)
                               for k, v in sorted(slice_scores.items())},
        }
    }
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_submission(public_dir, submission_out):
    seed_everything()
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")

    tr_rows = prepare_rows(train, build_candidates(train, "train"), with_gold=True)
    te_rows = prepare_rows(test, build_candidates(test, "test"), with_gold=False)

    model = EraMeshModel().fit(tr_rows)
    preds = {}
    for R, (o, b) in zip(te_rows, model.predict(te_rows)):
        preds[R["id"]], _ = program_of(R, o, b)

    submission = pd.DataFrame({
        "id": test["id"].astype(int),
        "chronicle_program": [preds[int(x)] for x in test["id"]],
    })
    errors = validate_submission(test, submission)
    if errors:
        raise RuntimeError("Submission validation failed: " + " | ".join(errors[:12]))

    out = Path(submission_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False, lineterminator="\n")
    print(json.dumps({
        "rows": int(len(submission)),
        "validation_errors": 0,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }, indent=2))


def main():
    args = sys.argv[1:]
    if "--cv" in args:
        args.remove("--cv")
        if len(args) != 2:
            raise SystemExit("Usage: python3 solution.py <public_dir> <cv.json> --cv")
        run_cv(args[0], args[1])
        return
    if len(args) != 2:
        raise SystemExit("Usage: python3 solution.py <public_dir> <submission.csv>")
    run_submission(args[0], args[1])


if __name__ == "__main__":
    main()
