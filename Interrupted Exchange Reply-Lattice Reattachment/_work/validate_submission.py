"""Independent submission validator (does not import solution.py)."""
import sys, hashlib, json
import numpy as np, pandas as pd
sub_path = sys.argv[1]
sub = pd.read_csv(sub_path); test = pd.read_csv("test.csv"); sample = pd.read_csv("sample_submission.csv")
errs = []
if len(sub) != len(test): errs.append(f"row count {len(sub)} != {len(test)}")
if list(sub.columns) != list(sample.columns): errs.append("column schema/order differs from sample")
if list(sub["id"]) != list(test["id"]): errs.append("id order differs from test.csv")
if sub["id"].duplicated().any(): errs.append("duplicate ids")
if sub.isna().any().any(): errs.append("missing values")
v = sub.iloc[:, 1:].to_numpy(dtype=float)
if not np.isfinite(v).all(): errs.append("non-finite values")
if ((v < 0) | (v > 1)).any(): errs.append("values outside [0,1]")
m = v.reshape(-1, 4, 4)
print(json.dumps({"file": sub_path, "rows": len(sub), "errors": errs,
                  "max_row_sum_dev": float(np.abs(m.sum(2) - 1).max()), "max_col_sum_dev": float(np.abs(m.sum(1) - 1).max()),
                  "sha256": hashlib.sha256(open(sub_path, "rb").read()).hexdigest()}))
