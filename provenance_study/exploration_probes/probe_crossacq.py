"""Development-only probe for cross-acquisition provenance robustness.

Touches ONLY development constructed batches 0-7 (both source replicates).
Locked batches 8-9 are never loaded. This is an exploratory diagnostic to
choose a novel method direction; it is not a locked or confirmatory result.

Three questions:
  A. How large is the cross-cube (source-replicate) transfer drop for
     mean-spectrum models? (headroom for an "obvious effect")
  B. Is the within-origin between-cube shift a low-rank, shared nuisance that
     projecting out helps? (motivates acquisition-invariant discriminant)
  C. Does within-seed pixel-population heterogeneity transfer across cubes
     better than / add to the mean spectrum? (motivates distributional model)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(r"d:\projects\Semen-Ziziphi-Spinosae")
sys.path.insert(0, str(ROOT))
from provenance_study.core import discover_manifest, load_csv_split  # noqa: E402

NUM_CLASSES = 8
t0 = time.perf_counter()

manifest = discover_manifest(ROOT / "data")  # hash_files=False -> no locked bytes
dev = load_csv_split(manifest, split="development", verify_hashes=False)
X = dev.X            # (1012, 392) mean reflectance
y = dev.y
recs = dev.records
rep = np.array([r.replicate for r in recs])
batch = np.array([r.constructed_batch for r in recs])
wl = dev.wavelengths
print(f"[load] dev seeds={X.shape[0]} bands={X.shape[1]} "
      f"rep1={int((rep==1).sum())} rep2={int((rep==2).sum())} "
      f"({time.perf_counter()-t0:.1f}s)")
assert set(np.unique(batch)) <= set(range(8)), "locked batch leaked!"


def sg1(A):
    return savgol_filter(A, 15, 2, deriv=1, axis=1, mode="interp")


def snv(A):
    m = A.mean(1, keepdims=True)
    s = A.std(1, ddof=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return (A - m) / s


def make_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def make_lr():
    return LogisticRegression(C=1.0, max_iter=5000, tol=1e-4)


def eval_lobo(feat, model_fn):
    pred = np.empty_like(y)
    for b in range(8):
        tr, te = batch != b, batch == b
        sc = StandardScaler().fit(feat[tr])
        m = model_fn().fit(sc.transform(feat[tr]), y[tr])
        pred[te] = m.predict(sc.transform(feat[te]))
    return balanced_accuracy_score(y, pred)


def eval_crosscube(feat, model_fn):
    out = []
    for tr, te in ((1, 2), (2, 1)):
        itr, ite = rep == tr, rep == te
        sc = StandardScaler().fit(feat[itr])
        m = model_fn().fit(sc.transform(feat[itr]), y[itr])
        out.append(balanced_accuracy_score(y[ite], m.predict(sc.transform(feat[ite]))))
    return out[0], out[1], float(np.mean(out))


# ---------------------------------------------------------------- Part A
print("\n=== A. Cross-acquisition headroom (mean-spectrum models) ===")
print(f"{'prep':6} {'model':4} {'LOBO_BA':>8} {'cc_1to2':>8} {'cc_2to1':>8} {'cc_theta':>8}")
preps = {"raw": X, "snv": snv(X), "sg1": sg1(X)}
for pname, feat in preps.items():
    for mname, mfn in (("lda", make_lda), ("lr", make_lr)):
        lobo = eval_lobo(feat, mfn)
        a, b_, th = eval_crosscube(feat, mfn)
        print(f"{pname:6} {mname:4} {lobo:8.4f} {a:8.4f} {b_:8.4f} {th:8.4f}")

# ---------------------------------------------------------------- Part B
print("\n=== B. Between-cube shift structure (sg1 space, descriptive) ===")
F = StandardScaler().fit_transform(sg1(X))
diffs = np.array([F[(y == c) & (rep == 1)].mean(0) - F[(y == c) & (rep == 2)].mean(0)
                  for c in range(8)])
U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
energy = S**2 / (S**2).sum()
print("between-cube diff singular-value energy fraction:",
      np.round(energy, 3).tolist())
print(f"top-1 explains {energy[0]:.1%}, top-2 {energy[:2].sum():.1%}, "
      f"top-3 {energy[:3].sum():.1%} of between-cube shift energy")
# pairwise cosine among per-origin shift vectors (are they aligned?)
Dn = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12)
cos = Dn @ Dn.T
iu = np.triu_indices(8, 1)
print(f"per-origin shift-vector pairwise cosine: mean={cos[iu].mean():+.3f} "
      f"min={cos[iu].min():+.3f} max={cos[iu].max():+.3f}")

# B1: optimistic (uses both cubes to estimate subspace) projection sweep
print("\n  B1 optimistic shared-subspace projection (upper bound):")
print(f"  {'k':>2} {'cc_1to2':>8} {'cc_2to1':>8} {'cc_theta':>8}")
raw_sg1 = sg1(X)
for k in (0, 1, 2, 3, 5):
    if k == 0:
        proj = raw_sg1
    else:
        Vk = Vt[:k]                                   # k x p directions
        proj = raw_sg1 - (raw_sg1 @ Vk.T) @ Vk        # remove shared subspace
    a, b_, th = eval_crosscube(proj, make_lda)
    print(f"  {k:>2} {a:8.4f} {b_:8.4f} {th:8.4f}")

# B2: honest leave-one-origin-out subspace transfer
print("\n  B2 honest leave-one-origin-out shared-subspace projection:")
print(f"  {'k':>2} {'cc_theta_LOO':>12}")
for k in (0, 1, 2, 3, 5):
    thetas = []
    for tr_rep, te_rep in ((1, 2), (2, 1)):
        # estimate subspace from origins != held-out, using standardized sg1
        for held in range(8):
            pass  # placeholder to keep structure; real loop below
    # proper implementation: for each direction, hold out one origin, estimate
    # shift subspace from the other 7 origins (both cubes), project, classify
    # the held-out origin's test-cube seeds using a model trained on train-cube.
    per_dir = []
    for tr_rep, te_rep in ((1, 2), (2, 1)):
        correct_by_origin = []
        # train one model per held-out origin is wasteful; instead project all
        # features with a subspace that excludes the evaluated origin, then use
        # a single model. To stay strictly honest we loop origins.
        for held in range(8):
            others = [c for c in range(8) if c != held]
            d_oth = np.array([F[(y == c) & (rep == 1)].mean(0)
                              - F[(y == c) & (rep == 2)].mean(0) for c in others])
            if k == 0:
                Vk = np.zeros((0, F.shape[1]))
            else:
                _, _, vt = np.linalg.svd(d_oth, full_matrices=False)
                Vk = vt[:k]
            def prj(A):
                return A - (A @ Vk.T) @ Vk if k else A
            itr = rep == tr_rep
            ite = (rep == te_rep) & (y == held)
            sc = StandardScaler().fit(prj(raw_sg1[itr]))
            m = make_lda().fit(sc.transform(prj(raw_sg1[itr])), y[itr])
            pr = m.predict(sc.transform(prj(raw_sg1[ite])))
            correct_by_origin.append((pr == y[ite]).mean())
        per_dir.append(np.mean(correct_by_origin))
    print(f"  {k:>2} {np.mean(per_dir):12.4f}")

# ---------------------------------------------------------------- Part C
print("\n=== C. Within-seed pixel-population transfer (streams MAT dev only) ===")
tc = time.perf_counter()
seed_mean = np.empty((len(recs), 392))
seed_snvstd = np.empty((len(recs), 392))
rng = np.random.default_rng(20260722)
for i, r in enumerate(recs):
    with h5py.File(r.mat_path, "r") as h:
        p = h["patch_chw"][()]
        msk = np.asarray(h["crop_mask"][()]).squeeze() > 0.5
    px = p[msk].astype(np.float64)             # (n_px, 392)
    seed_mean[i] = px.mean(0)
    pxs = snv(px)                              # per-pixel SNV -> intrinsic shape
    seed_snvstd[i] = pxs.std(0, ddof=1)
print(f"[C load] streamed {len(recs)} MAT seeds ({time.perf_counter()-tc:.1f}s)")

featsC = {
    "mean_raw": seed_mean,
    "mean_sg1": sg1(seed_mean),
    "snv_pixel_std": seed_snvstd,
    "mean_sg1 + snv_pixel_std": np.hstack([sg1(seed_mean), seed_snvstd]),
}
print(f"{'feature':28} {'LOBO_BA':>8} {'cc_1to2':>8} {'cc_2to1':>8} {'cc_theta':>8}")
for name, feat in featsC.items():
    lobo = eval_lobo(feat, make_lda)
    a, b_, th = eval_crosscube(feat, make_lda)
    print(f"{name:28} {lobo:8.4f} {a:8.4f} {b_:8.4f} {th:8.4f}")

print(f"\n[done] total {time.perf_counter()-t0:.1f}s")
