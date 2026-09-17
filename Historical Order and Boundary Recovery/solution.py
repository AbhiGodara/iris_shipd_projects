#!/usr/bin/env python3
"""
EraMesh: Historical Order and Boundary Recovery
Strong deterministic CPU/GPU-independent starter.

Usage:
  python3 solution.py <public_dir> <submission.csv>
  python3 solution.py <public_dir> <cv.json> --cv

Core approach:
  1) Parse row-local event/source cards.
  2) Learn pairwise precedence from numeric structural features with LightGBM.
  3) Learn complementary pairwise precedence from masked event/source text with TF-IDF + logistic regression.
  4) Recover an unsupervised 1-D chronology from the unsigned distance mesh using classical MDS.
  5) Fuse learned precedence + mesh chronology + genuine precedence cues.
  6) Refine the global order with pairwise objective + mesh stress.
  7) Learn same-era pair probabilities and decode exactly era_count contiguous segments by dynamic programming.
  8) Validate the exact submission grammar before writing.
All predictions use only public training data and test inputs.
"""

import os
import sys
import re
import json
import math
import random
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse.csgraph import shortest_path
from numpy.linalg import eigh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold


# ------------------------------ deterministic config -------------------------

SEED = 20260917
LGB_THREADS = 6

ORDER_PARAMS = dict(
    objective="binary", learning_rate=0.04, num_leaves=31,
    min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9,
    bagging_freq=1, lambda_l2=3.0, verbose=-1, num_threads=LGB_THREADS,
    seed=42, deterministic=True, force_row_wise=True,
)
SAME_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31,
    min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9,
    bagging_freq=1, lambda_l2=3.0, verbose=-1, num_threads=LGB_THREADS,
    seed=43, deterministic=True, force_row_wise=True,
)
ORDER_ROUNDS = 260
SAME_ROUNDS = 230
TEXT_C = 1.0
TEXT_MAX_FEATURES = 20000
W_NUMERIC = 0.60
W_TEXT = 0.20
W_MDS = 0.20
MESH_STRESS_WEIGHT = 0.80
MDS_SIGMOID_SCALE = 1.50

GAP_REP = np.array([1.5, 6.5, 17.5, 42.5, 105.0, 225.0, 400.0], dtype=float)

CATEGORIES = [
    "colonization", "culture", "disaster", "economy", "founding",
    "independence", "migration", "politics", "religion", "war",
]
DURS = ["d0", "d1", "d2", "d3"]
TEMP_WORDS = {
    "after", "before", "later", "earlier", "then", "following",
    "subsequent", "previously", "first", "finally", "eventually",
    "began", "begin", "ended", "end", "appointed", "elected",
    "became", "invaded", "invasion", "war", "peace", "revolt",
    "revolution", "died", "death", "born", "arrived", "departed",
    "moved", "colony", "colonial", "reign", "ruled", "capital",
    "occupied", "annexed", "merged", "seceded", "withdrew", "signed",
}

EVENT_RE = re.compile(
    r"(q\d+)\{cat=([^;]+);dur=([^;]+);src=([^;]*);title=(.*?);text=(.*)\}"
)
SRC_RE = re.compile(r"(s\d+)\{kind=([^;]+);label=(.*)")


# ------------------------------ basic utilities -----------------------------

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(LGB_THREADS)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def normalize_text(s):
    s = str(s).lower()
    s = re.sub(r"v\d+", " V_ALIAS ", s)
    s = s.replace("<mask>", " MASK ").replace("<time>", " TIME ")
    s = re.sub(r"q\d+", " Q_HANDLE ", s)
    return re.findall(r"[a-z_]+|time|mask", s)


def word_set(s):
    return set(w.lower() for w in re.findall(r"[A-Za-z]+", str(s))
               if w.lower() not in {"mask", "time"})


# ------------------------------- parsing ------------------------------------

def parse_row(row):
    events = {}
    for part in str(row["event_cards"]).split(" || "):
        m = EVENT_RE.fullmatch(part.strip())
        if not m:
            continue
        q, cat, dur, src, title, text = m.groups()
        events[q] = {
            "cat": cat,
            "dur": dur,
            "src": src.split(",") if src else [],
            "title": title,
            "text": text,
        }

    sources = {}
    for part in str(row["source_ledger"]).split(" || "):
        m = SRC_RE.fullmatch(part.strip())
        if not m:
            continue
        s, kind, label = m.groups()
        sources[s] = {"kind": kind, "label": label}

    cues = [x for x in str(row["precedence_cues"]).split() if "<" in x]
    mesh = {}
    for tok in str(row["distance_mesh"]).split():
        m = re.fullmatch(r"(q\d+)~(q\d+):g(\d+)", tok)
        if m:
            a, b, g = m.groups()
            mesh[tuple(sorted((a, b)))] = int(g)

    program = str(row["chronicle_program"]).split() if "chronicle_program" in row else []
    gold_order = [x for x in program if x != "BREAK"]
    gold_segments = []
    cur = []
    for x in program:
        if x == "BREAK":
            if cur:
                gold_segments.append(cur)
            cur = []
        else:
            cur.append(x)
    if cur:
        gold_segments.append(cur)

    return events, sources, cues, mesh, gold_order, gold_segments


def pair_text(events, sources, a, b):
    def build(q):
        e = events[q]
        src_kinds = [sources[s]["kind"] for s in e["src"] if s in sources]
        toks = [e["cat"], e["dur"], *src_kinds, e["title"], e["text"]]
        return normalize_text(" ".join(map(str, toks)))
    ta, tb = build(a), build(b)
    return (
        " ".join("A_" + w for w in ta)
        + " [PAIRSEP] "
        + " ".join("B_" + w for w in tb)
    )


# ---------------------------- numeric pair features -------------------------

def lexical_features(ea, eb):
    wa = word_set(ea["title"] + " " + ea["text"])
    wb = word_set(eb["title"] + " " + eb["text"])
    inter = len(wa & wb)
    union = len(wa | wb)
    ta = wa & TEMP_WORDS
    tb = wb & TEMP_WORDS
    return [
        len(wa), len(wb), inter,
        inter / max(1, union),
        inter / max(1, min(len(wa), len(wb))),
        len(ta), len(tb), len(ta & tb), len(wa ^ wb),
    ]


def pair_features(events, sources, cues, mesh, a, b):
    ea, eb = events[a], events[b]

    fa = [int(ea["cat"] == c) for c in CATEGORIES]
    fb = [int(eb["cat"] == c) for c in CATEGORIES]
    f = [*fa, *fb, int(ea["cat"] == eb["cat"])]

    f += [int(ea["dur"] == d) for d in DURS]
    f += [int(eb["dur"] == d) for d in DURS]
    f += [
        int(ea["dur"] == eb["dur"]),
        DURS.index(ea["dur"]),
        DURS.index(eb["dur"]),
    ]

    sa, sb = set(ea["src"]), set(eb["src"])
    f += [
        len(sa), len(sb), len(sa & sb), len(sa | sb),
        len(sa & sb) / max(1, min(len(sa), len(sb))),
    ]

    ka = [sources[s]["kind"] for s in ea["src"] if s in sources]
    kb = [sources[s]["kind"] for s in eb["src"] if s in sources]
    f += [
        len(set(ka) & set(kb)),
        int(bool(set(ka) & set(kb))),
        int("gov" in ka), int("gov" in kb),
        int("academic" in ka), int("academic" in kb),
        int("primary" in ka), int("primary" in kb),
        int("archive" in ka), int("archive" in kb),
        int("reference" in ka), int("reference" in kb),
    ]

    f += lexical_features(ea, eb)

    cue_ab = int(f"{a}<{b}" in cues)
    cue_ba = int(f"{b}<{a}" in cues)
    f += [cue_ab, cue_ba]

    g = mesh.get(tuple(sorted((a, b))), -1)
    f += [
        g, int(g >= 0), int(g == 0), int(g <= 1),
        int(g >= 4), int(g >= 5), (g if g >= 0 else -1) / 6.0,
    ]

    ta = (ea["title"] + " " + ea["text"]).lower()
    tb = (eb["title"] + " " + eb["text"]).lower()
    early_words = ["before", "earlier", "previously", "first"]
    late_words = ["after", "following", "later", "subsequent", "then"]
    f += [
        sum(ta.count(w) for w in early_words),
        sum(ta.count(w) for w in late_words),
        sum(tb.count(w) for w in early_words),
        sum(tb.count(w) for w in late_words),
    ]
    return np.asarray(f, dtype=np.float32)


def build_pair_dataset(df):
    parsed = [parse_row(r) for _, r in df.iterrows()]
    X, y_order, y_same = [], [], []
    texts, row_pairs, row_slices = [], [], []
    start = 0

    for rid, (_, row) in enumerate(df.iterrows()):
        events, sources, cues, mesh, gold_order, gold_segments = parsed[rid]
        qlist = sorted(events)
        pos = {q: i for i, q in enumerate(gold_order)}
        seg_id = {q: si for si, seg in enumerate(gold_segments) for q in seg}

        has_target = bool(gold_order)
        pairs = []
        for i in range(len(qlist)):
            for j in range(i + 1, len(qlist)):
                a, b = qlist[i], qlist[j]
                pairs.append((a, b))
                X.append(pair_features(events, sources, cues, mesh, a, b))
                texts.append(pair_text(events, sources, a, b))
                if has_target:
                    y_order.append(int(pos[a] < pos[b]))
                    y_same.append(int(seg_id[a] == seg_id[b]))

        row_pairs.append(pairs)
        row_slices.append((start, start + len(pairs)))
        start += len(pairs)

    return parsed, np.stack(X), np.asarray(y_order), np.asarray(y_same), texts, row_pairs, row_slices


# ----------------------- component proxy for validation ----------------------

def row_shingles(row):
    text = (str(row["event_cards"]) + " " + str(row["source_ledger"])).lower()
    text = re.sub(r"q\d+", "Q", text)
    text = re.sub(r"v\d+", "V", text)
    text = text.replace("<mask>", "MASK").replace("<time>", "TIME")
    toks = re.findall(r"[a-z]+|mask|time|d[0-3]|q|v", text)
    if len(toks) < 5:
        return set()
    return set(zip(*(toks[i:] for i in range(5))))


def make_component_groups(df, threshold=0.45):
    sh = [row_shingles(r) for _, r in df.iterrows()]
    parent = list(range(len(sh)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            jac = len(sh[i] & sh[j]) / max(1, len(sh[i] | sh[j]))
            if jac >= threshold:
                union(i, j)

    roots = {}
    groups = np.empty(len(sh), dtype=int)
    for i in range(len(sh)):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        groups[i] = roots[r]
    return groups


# ----------------------------- mesh chronology ------------------------------

def mesh_mds(events, cues, mesh):
    qlist = sorted(events)
    n = len(qlist)
    qidx = {q: i for i, q in enumerate(qlist)}

    D = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)
    for (a, b), g in mesh.items():
        d = GAP_REP[g]
        D[qidx[a], qidx[b]] = d
        D[qidx[b], qidx[a]] = d

    SP = shortest_path(D, directed=False, unweighted=False)
    finite = SP[np.isfinite(SP)]
    fill = float(np.max(finite)) if len(finite) else 1.0
    SP[~np.isfinite(SP)] = fill * 1.5

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (SP ** 2) @ J
    vals, vecs = eigh(B)
    coord = vecs[:, -1] * math.sqrt(max(float(vals[-1]), 1e-9))

    order_idx = np.argsort(coord)
    order = [qlist[i] for i in order_idx]

    pos = {q: i for i, q in enumerate(order)}
    cue_pairs = [c.split("<") for c in cues if "<" in c]
    violations = sum(pos[a] > pos[b] for a, b in cue_pairs if a in pos and b in pos)
    if violations > len(cue_pairs) - violations:
        coord = -coord
        order_idx = np.argsort(coord)
        order = [qlist[i] for i in order_idx]

    ordered_coord = coord[order_idx]
    return order, ordered_coord


def mesh_pair_probabilities(events, cues, mesh, pairs):
    order, coord = mesh_mds(events, cues, mesh)
    cmap = {q: coord[i] for i, q in enumerate(order)}
    out = []
    for a, b in pairs:
        diff = cmap[b] - cmap[a]
        out.append(float(sigmoid(diff / MDS_SIGMOID_SCALE)))
    return np.asarray(out, dtype=float)


# --------------------------- global order decoder ----------------------------

def refine_order(events, pairs, p_before, cues, mesh):
    qlist = sorted(events)
    qidx = {q: i for i, q in enumerate(qlist)}
    W = np.zeros((len(qlist), len(qlist)), dtype=float)

    for (a, b), p in zip(pairs, p_before):
        l = float(logit(p))
        W[qidx[a], qidx[b]] = l
        W[qidx[b], qidx[a]] = -l

    for c in cues:
        a, b = c.split("<")
        if a in qidx and b in qidx:
            W[qidx[a], qidx[b]] += 4.0
            W[qidx[b], qidx[a]] -= 4.0

    seed = [qlist[i] for i in np.argsort(-W.sum(axis=1), kind="mergesort")]

    mesh_expected = GAP_REP.copy()

    def objective(order):
        pos = {q: i for i, q in enumerate(order)}
        val = 0.0
        n = len(order)
        for i in range(n):
            ai = qidx[order[i]]
            for j in range(i + 1, n):
                bj = qidx[order[j]]
                val += W[ai, bj]

        # Add a soft global mesh-stress term. This uses only the supplied
        # unsigned/coarsened mesh and is evaluated on the final permutation.
        denom = max(1, n - 1)
        for (a, b), g in mesh.items():
            d_rank = abs(pos[a] - pos[b]) / denom
            # Rank-scale target is intentionally coarse; the learned pair model
            # remains the dominant ordering signal.
            target = min(0.85, 0.10 + 0.12 * g)
            val -= MESH_STRESS_WEIGHT * 4.0 * (d_rank - target) ** 2
        return val

    order = seed
    best = objective(order)

    for _ in range(12):
        improved = False

        for i in range(len(order) - 1):
            cand = order.copy()
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            sc = objective(cand)
            if sc > best + 1e-9:
                order, best, improved = cand, sc, True

        if len(order) <= 15:
            for i in range(len(order)):
                for j in range(i + 2, len(order)):
                    cand = order.copy()
                    x = cand.pop(i)
                    cand.insert(j, x)
                    sc = objective(cand)
                    if sc > best + 1e-9:
                        order, best, improved = cand, sc, True

        if not improved:
            break

    return order


# --------------------------- era segmentation -------------------------------

def decode_segments(order, pair_same_prob, era_count):
    n = len(order)
    qidx = {q: i for i, q in enumerate(order)}

    P = np.full((n, n), 0.5, dtype=float)
    for (a, b), p in pair_same_prob.items():
        i, j = qidx[a], qidx[b]
        P[i, j] = P[j, i] = float(np.clip(p, 1e-5, 1 - 1e-5))

    # If all pairs were same, the DP still yields the required nonempty segments.
    base = 0.0
    delta = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            p = P[i, j]
            base += math.log(1.0 - p)
            delta[i, j] = math.log(p) - math.log(1.0 - p)

    bonus = np.zeros((n, n + 1), dtype=float)
    for l in range(n):
        cur = 0.0
        for r in range(l + 1, n + 1):
            j = r - 1
            for i in range(l, j):
                cur += delta[i, j]
            bonus[l, r] = cur

    K = min(int(era_count), n)
    NEG = -1e100
    dp = np.full((K + 1, n + 1), NEG)
    prev = np.full((K + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, K + 1):
        for r in range(k, n + 1):
            best = NEG
            best_l = -1
            for l in range(k - 1, r):
                v = dp[k - 1, l] + bonus[l, r]
                if v > best:
                    best = v
                    best_l = l
            dp[k, r] = best
            prev[k, r] = best_l

    bounds = []
    r = n
    for k in range(K, 0, -1):
        l = int(prev[k, r])
        bounds.append((l, r))
        r = l
    bounds.reverse()

    return [order[l:r] for l, r in bounds]


# ------------------------------- scoring ------------------------------------

def gold_segments_from_program(program):
    segs, cur = [], []
    for tok in str(program).split():
        if tok == "BREAK":
            segs.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        segs.append(cur)
    return segs


def score_row(pred_order, pred_segments, gold_order, gold_segments):
    n = len(gold_order)
    if n <= 1:
        O = 1.0
    else:
        pos_p = {q: i for i, q in enumerate(pred_order)}
        agree = 0
        total = n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                a, b = gold_order[i], gold_order[j]
                agree += int((pos_p[a] - pos_p[b]) * (i - j) > 0)
        C = agree / total
        O = max(0.0, 2 * C - 1)

    gseg = {q: si for si, seg in enumerate(gold_segments) for q in seg}
    pseg = {q: si for si, seg in enumerate(pred_segments) for q in seg}
    tp = tn = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_g = gseg[gold_order[i]] == gseg[gold_order[j]]
            same_p = pseg[gold_order[i]] == pseg[gold_order[j]]
            if same_g and same_p:
                tp += 1
            elif (not same_g) and (not same_p):
                tn += 1
            elif same_p:
                fp += 1
            else:
                fn += 1

    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    M = max(0.0, (tp * tn - fp * fn) / den) if den > 0 else 0.0

    def edges(segs):
        ed = set()
        if not segs:
            return ed
        ed.add(("START", segs[0][0], "START"))
        for seg in segs:
            for a, b in zip(seg, seg[1:]):
                ed.add((a, b, "WITHIN"))
        for s1, s2 in zip(segs[:-1], segs[1:]):
            ed.add((s1[-1], s2[0], "BREAK"))
        ed.add(("END", segs[-1][-1], "END"))
        return ed

    eg, ep = edges(gold_segments), edges(pred_segments)
    A = 2.0 * len(eg & ep) / max(1, len(eg) + len(ep))
    return float(0.50 * O + 0.40 * M + 0.10 * A)


# ----------------------------- model training -------------------------------

def fit_models(train_df):
    parsed, X, yord, ysame, texts, row_pairs, row_slices = build_pair_dataset(train_df)

    order_model = lgb.train(
        ORDER_PARAMS, lgb.Dataset(X, label=yord.astype(np.float32)),
        num_boost_round=ORDER_ROUNDS
    )
    same_model = lgb.train(
        SAME_PARAMS, lgb.Dataset(X, label=ysame.astype(np.float32)),
        num_boost_round=SAME_ROUNDS
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2,
        max_features=TEXT_MAX_FEATURES, sublinear_tf=True
    )
    Xt = vectorizer.fit_transform(texts)
    text_model = LogisticRegression(
        C=TEXT_C, max_iter=250, solver="liblinear", random_state=SEED
    )
    text_model.fit(Xt, yord)

    return {
        "parsed": parsed,
        "X": X,
        "yord": yord,
        "ysame": ysame,
        "texts": texts,
        "row_pairs": row_pairs,
        "row_slices": row_slices,
        "order_model": order_model,
        "same_model": same_model,
        "vectorizer": vectorizer,
        "text_model": text_model,
    }


def predict_rows(models, df):
    parsed, X, _, _, texts, row_pairs, row_slices = build_pair_dataset(
        df.assign(chronicle_program=[""] * len(df))
    )
    # The synthetic empty target above only affects parsing of the unused gold
    # fields; all pair features remain test-only.
    p_num = models["order_model"].predict(X)
    p_txt = models["text_model"].predict_proba(models["vectorizer"].transform(texts))[:, 1]
    p_same = models["same_model"].predict(X)

    out = {}
    cursor = 0
    for rid, (_, row) in enumerate(df.iterrows()):
        lo, hi = row_slices[rid]
        pairs = row_pairs[rid]
        pn = p_num[cursor:cursor + (hi - lo)]
        pt = p_txt[cursor:cursor + (hi - lo)]
        ps = p_same[cursor:cursor + (hi - lo)]
        cursor = hi

        mds_p = mesh_pair_probabilities(
            parsed[rid][0], parsed[rid][2], parsed[rid][3], pairs
        )
        blend = W_NUMERIC * logit(pn) + W_TEXT * logit(pt) + W_MDS * logit(mds_p)
        p_before = sigmoid(blend)

        events, sources, cues, mesh, _, _ = parsed[rid]
        order = refine_order(events, pairs, p_before, cues, mesh)
        same_map = {(a, b): float(p) for (a, b), p in zip(pairs, ps)}
        segs = decode_segments(order, same_map, int(row["era_count"]))

        out[int(row["id"])] = " ".join(
            [tok for si, seg in enumerate(segs)
             for tok in (seg + (["BREAK"] if si < len(segs) - 1 else []))]
        )

    return out


# --------------------------- validation harness -----------------------------

def validate_submission(df, submission):
    expected = set(int(x) for x in df["id"])
    rows = list(submission.itertuples(index=False))
    errors = []

    if list(submission.columns) != ["id", "chronicle_program"]:
        errors.append("wrong columns/order")
    if len(rows) != len(df):
        errors.append(f"wrong row count: {len(rows)}")
    ids = [int(r.id) for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate IDs")
    if set(ids) != expected:
        errors.append("ID set mismatch")

    event_counts = dict(zip(df["id"].astype(int), df["event_count"].astype(int)))
    era_counts = dict(zip(df["id"].astype(int), df["era_count"].astype(int)))
    event_maps = {}

    for _, row in df.iterrows():
        events, _, _, _, _, _ = parse_row(row)
        event_maps[int(row["id"])] = set(events)

    for r in rows:
        rid = int(r.id)
        toks = str(r.chronicle_program).split()
        if len(toks) == 0:
            errors.append(f"{rid}: empty program")
            continue
        expected_events = event_counts[rid]
        expected_breaks = era_counts[rid] - 1
        if toks.count("BREAK") != expected_breaks:
            errors.append(f"{rid}: wrong separator count")
        handles = [t for t in toks if t != "BREAK"]
        if len(handles) != expected_events:
            errors.append(f"{rid}: wrong handle count")
        if len(handles) != len(set(handles)):
            errors.append(f"{rid}: repeated handle")
        if set(handles) != event_maps[rid]:
            errors.append(f"{rid}: handle set mismatch")
        if "BREAK BREAK" in str(r.chronicle_program):
            errors.append(f"{rid}: consecutive BREAK")
        if str(r.chronicle_program).strip() != str(r.chronicle_program):
            errors.append(f"{rid}: leading/trailing whitespace")
    return errors


def build_validation_groups(df):
    return make_component_groups(df, threshold=0.45)


def run_cv(public_dir, out_json):
    seed_everything()
    train = pd.read_csv(Path(public_dir) / "train.csv")
    groups = build_validation_groups(train)

    gkf = GroupKFold(4)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=groups)):
        tr = train.iloc[tr_idx].copy()
        va = train.iloc[va_idx].copy()

        models = fit_models(tr)

        # Test/validation pair predictions.
        parsed_pred, Xv, _, _, texts_v, pairs_v, slices_v = build_pair_dataset(
            va.assign(chronicle_program=[""] * len(va))
        )
        parsed_gold = [parse_row(r) for _, r in va.iterrows()]
        p_num = models["order_model"].predict(Xv)
        p_txt = models["text_model"].predict_proba(models["vectorizer"].transform(texts_v))[:, 1]
        p_same = models["same_model"].predict(Xv)

        cursor = 0
        scores = []

        for rid, (_, row) in enumerate(va.iterrows()):
            lo, hi = slices_v[rid]
            pairs = pairs_v[rid]
            pn = p_num[cursor:hi]
            pt = p_txt[cursor:hi]
            ps = p_same[cursor:hi]
            cursor = hi

            mds_p = mesh_pair_probabilities(
                parsed_pred[rid][0], parsed_pred[rid][2], parsed_pred[rid][3], pairs
            )
            blend = W_NUMERIC * logit(pn) + W_TEXT * logit(pt) + W_MDS * logit(mds_p)
            p_before = sigmoid(blend)

            events, sources, cues, mesh, _, _ = parsed_pred[rid]
            _, _, _, _, gold_order, gold_segments = parsed_gold[rid]
            pred_order = refine_order(events, pairs, p_before, cues, mesh)
            same_map = {(a, b): float(p) for (a, b), p in zip(pairs, ps)}
            pred_segments = decode_segments(pred_order, same_map, int(row["era_count"]))
            scores.append(score_row(
                pred_order, pred_segments, gold_order, gold_segments
            ))

        fold_score = float(np.mean(scores))
        fold_scores.append(fold_score)
        print(f"[fold {fold}] score={fold_score:.5f}", flush=True)

    report = {
        "validation": {
            "scheme": "4-fold GroupKFold on solver-visible row-component proxy "
                      "(five-token shingle Jaccard >= 0.45)",
            "fold_scores": [round(x, 5) for x in fold_scores],
            "mean": round(float(np.mean(fold_scores)), 5),
            "std": round(float(np.std(fold_scores)), 5),
            "worst": round(float(np.min(fold_scores)), 5),
        },
        "model": {
            "numeric_pair_model": "LightGBM",
            "text_pair_model": "TF-IDF + LogisticRegression",
            "mesh_order_model": "1-D classical MDS on unsigned distance mesh",
            "same_period_model": "LightGBM",
            "blend_logit_weights": [W_NUMERIC, W_TEXT, W_MDS],
            "mesh_stress_weight": MESH_STRESS_WEIGHT,
        },
        "constraints": {
            "pretrained_models": False,
            "external_data": False,
            "external_lookup": False,
            "test_labels": False,
            "hard_coded_test_outputs": False,
            "deterministic": True,
        },
    }
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_submission(public_dir, submission_out):
    seed_everything()
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")

    models = fit_models(train)
    pred_map = predict_rows(models, test)

    submission = pd.DataFrame({
        "id": test["id"].astype(int),
        "chronicle_program": [pred_map[int(x)] for x in test["id"]],
    })

    errors = validate_submission(test, submission)
    if errors:
        raise RuntimeError("Submission validation failed: " + " | ".join(errors[:12]))

    Path(submission_out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)

    # Save a small run report next to the CSV.
    counts = submission["chronicle_program"].str.split().map(
        lambda x: len([t for t in x if t != "BREAK"])
    )
    run_info = {
        "rows": len(submission),
        "validation_errors": 0,
        "mean_event_count": float(counts.mean()),
        "min_event_count": int(counts.min()),
        "max_event_count": int(counts.max()),
        "sha256": hashlib.sha256(Path(submission_out).read_bytes()).hexdigest(),
    }
    info_path = Path(submission_out).with_name("run_info.json")
    info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))


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
