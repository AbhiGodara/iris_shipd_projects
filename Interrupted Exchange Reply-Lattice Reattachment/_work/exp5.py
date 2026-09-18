"""Exp5: exp4 stacking + extra kNN views + LightGBM stage-2; evaluates blends with e3 OOF."""
import sys, time, json, types
sys.path.insert(0, "."); sys.path.insert(0, "_work")
import numpy as np, torch, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
import solution as S
from exp4 import fit_bil, bil_comps, stack_feats, fit_stage2, pred_stage2

CM = np.zeros(S.V, np.float32); CM[S.W_LO:S.N_LO] = 1


def view(D, kind):
    v = types.SimpleNamespace(bp=D.bp, bo=D.bo)
    if kind == "prevlast":
        v.bl = D.bl + D.bp
    else:  # content-only
        v.bl, v.bp, v.bo = D.bl * CM, D.bp * CM, D.bo * CM
    return v


def extra_knn(D, gi, q_rows, m_rows, groups=None, dev=None):
    outs = []
    for kind in ("prevlast", "content"):
        Vw = view(D, kind)
        if groups is None:
            outs.append(S.knn_feats(Vw, q_rows, Vw, m_rows, gi, S.idf_of(Vw, m_rows, dev), dev))
        else:
            f = np.zeros((len(q_rows), 4, 4, S.N_KNN), np.float32)
            for a, b in GroupKFold(S.N_INNER).split(q_rows, groups=groups[q_rows]):
                f[b] = S.knn_feats(Vw, q_rows[b], Vw, q_rows[a], gi, S.idf_of(Vw, q_rows[a], dev), dev)
            outs.append(f)
    return np.concatenate(outs, -1)


LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=60,
                  feature_fraction=0.8, lambda_l2=1.0, verbose=-1, seed=S.SEED, deterministic=True,
                  num_threads=8, force_row_wise=True)


def lgb_perm_scores(x_tr, y_tr, x_q, rounds=400):
    ds = lgb.Dataset(x_tr.reshape(-1, x_tr.shape[-1]), y_tr.reshape(-1))
    bst = lgb.train(LGB_PARAMS, ds, num_boost_round=rounds)
    raw = bst.predict(x_q.reshape(-1, x_q.shape[-1]), raw_score=True).reshape(-1, 4, 4)
    tr_raw = bst.predict(x_tr[:2500].reshape(-1, x_tr.shape[-1]), raw_score=True).reshape(-1, 4, 4)
    p = torch.as_tensor(S.PERMS_NP)
    ps = lambda r: S.perm_scores(torch.as_tensor(r), p).numpy()
    return ps(raw), ps(tr_raw)


def lsm(o):
    o = o - o.max(1, keepdims=True)
    return o - np.log(np.exp(o).sum(1, keepdims=True))


def main():
    t0 = time.time(); dev = torch.device("cuda"); S.seed_everything()
    _, train, y, gi, gold = S.load(".")
    groups = train["group"].astype(str).values
    D = S.Data(S.featurize(train), {})
    folds = list(GroupKFold(4).split(train, groups=groups))
    oof_mlp = np.zeros((len(train), 24)); oof_lgb = np.zeros((len(train), 24)); tf = {"mlp": [], "lgb": []}
    for f, (tri, vai) in enumerate(folds):
        base_tr = np.concatenate([S.labelled_features(D, gi, tri, groups, dev), extra_knn(D, gi, tri, None, groups, dev)], -1)
        base_va = np.concatenate([S.query_features(D, vai, D, gi, tri, dev), extra_knn(D, gi, vai, tri, None, dev)], -1)
        comps_tr = np.zeros((len(tri), 4, 4, 4), np.float32)
        for a, b in GroupKFold(S.N_INNER).split(tri, groups=groups[tri]):
            comps_tr[b] = bil_comps(fit_bil(D.bags(tri[a]), gold[tri[a]], dev), D.bags(tri[b]), dev)
        comps_va = bil_comps(fit_bil(D.bags(tri), gold[tri], dev), D.bags(vai), dev)
        x_tr, x_va = S.standardize(stack_feats(base_tr, comps_tr), stack_feats(base_va, comps_va))
        m2 = fit_stage2(x_tr, gold[tri], dev)
        oof_mlp[vai] = pred_stage2(m2, x_va, dev)
        tf["mlp"].append(S.metric(S.map_matrix(pred_stage2(m2, x_tr, dev)), y[tri]))
        oof_lgb[vai], tr_ps = lgb_perm_scores(x_tr, y[tri], x_va)
        tf["lgb"].append(S.metric(S.map_matrix(tr_ps), y[tri][:2500]))
        print(f"fold {f} mlp={S.metric(S.map_matrix(oof_mlp[vai]), y[vai]):.4f} lgb={S.metric(S.map_matrix(oof_lgb[vai]), y[vai]):.4f} "
              f"trfit mlp={tf['mlp'][-1]:.3f} lgb={tf['lgb'][-1]:.3f} t={time.time()-t0:.0f}s", flush=True)
    np.save("_work/oof_e5_mlp.npy", oof_mlp); np.save("_work/oof_e5_lgb.npy", oof_lgb)
    e3 = np.load("_work/oof_e3_lam1e-3.npy")
    res = {"train_fit": tf, "time_s": round(time.time() - t0, 1)}
    for name, o in (("mlp", oof_mlp), ("lgb", oof_lgb), ("mlp+lgb", lsm(oof_mlp) + lsm(oof_lgb)),
                    ("e3+mlp", lsm(e3) + lsm(oof_mlp)), ("e3+lgb", lsm(e3) + lsm(oof_lgb)),
                    ("e3+mlp+lgb", lsm(e3) + lsm(oof_mlp) + lsm(oof_lgb))):
        fs = [S.metric(S.map_matrix(o[v]), y[v]) for _, v in folds]
        res[name] = {"folds": [round(x, 4) for x in fs], "mean": round(float(np.mean(fs)), 4),
                     "std": round(float(np.std(fs)), 4), "worst": round(float(np.min(fs)), 4)}
    print(json.dumps(res), flush=True)
    Path("_work/res_e5.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
