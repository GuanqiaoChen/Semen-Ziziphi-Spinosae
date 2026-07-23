"""Probe 2: where does a LARGE cross-acquisition effect live?
Development batches 0-7 only. No locked data. Exploratory.

D. Calibration/uncertainty gap under cube shift (NLL, ECE, accuracy).
E. Per-origin cross-cube recall: which origins collapse and are rescuable?
F. Proper within-seed covariance descriptor (band-band structure), cube transfer.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import h5py
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(r"d:\projects\Semen-Ziziphi-Spinosae")
sys.path.insert(0, str(ROOT))
from provenance_study.core import discover_manifest, load_csv_split, multiclass_metrics  # noqa

CLASS = ("HBS","HBX","HNA","HNX","NX","SXD","SXQ","XJH")
t0 = time.perf_counter()
man = discover_manifest(ROOT/"data")
dev = load_csv_split(man, split="development", verify_hashes=False)
X, y, recs = dev.X, dev.y, dev.records
rep = np.array([r.replicate for r in recs]); batch = np.array([r.constructed_batch for r in recs])

def sg1(A): return savgol_filter(A,15,2,deriv=1,axis=1,mode="interp")
def snv(A):
    m=A.mean(1,keepdims=True); s=A.std(1,ddof=1,keepdims=True); s=np.where(s<1e-12,1,s); return (A-m)/s
def lda(): return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
def lr(): return LogisticRegression(C=1.0,max_iter=5000,tol=1e-4)

def crosscube_proba(feat, mfn):
    """Return stacked (y_true, proba) over both directions."""
    ys, ps = [], []
    for tr,te in ((1,2),(2,1)):
        itr,ite = rep==tr, rep==te
        sc=StandardScaler().fit(feat[itr])
        m=mfn().fit(sc.transform(feat[itr]), y[itr])
        ps.append(m.predict_proba(sc.transform(feat[ite]))); ys.append(y[ite])
    return np.concatenate(ys), np.vstack(ps)

def lobo_proba(feat, mfn):
    P=np.zeros((len(y),8))
    for b in range(8):
        tr,te=batch!=b,batch==b
        sc=StandardScaler().fit(feat[tr]); m=mfn().fit(sc.transform(feat[tr]),y[tr])
        P[te]=m.predict_proba(sc.transform(feat[te]))
    return y, P

# ---- D. calibration gap ----
print("=== D. same-domain vs cross-cube discrimination + calibration (sg1) ===")
F=sg1(X)
for mname,mfn in (("lda",lda),("lr",lr)):
    yl,Pl=lobo_proba(F,mfn); ml=multiclass_metrics(yl,Pl)
    yc,Pc=crosscube_proba(F,mfn); mc=multiclass_metrics(yc,Pc)
    print(f"[{mname}] LOBO   BA={ml['balanced_accuracy']:.4f} NLL={ml['negative_log_likelihood']:.3f} "
          f"ECE={ml['expected_calibration_error']:.3f} Brier={ml['multiclass_brier_score']:.3f}")
    print(f"[{mname}] xcube  BA={mc['balanced_accuracy']:.4f} NLL={mc['negative_log_likelihood']:.3f} "
          f"ECE={mc['expected_calibration_error']:.3f} Brier={mc['multiclass_brier_score']:.3f}")

# ---- E. per-origin cross-cube recall ----
print("\n=== E. per-origin cross-cube recall (sg1-lda), avg of both directions ===")
recs_dir=[]
for tr,te in ((1,2),(2,1)):
    itr,ite=rep==tr,rep==te
    sc=StandardScaler().fit(F[itr]); m=lda().fit(sc.transform(F[itr]),y[itr])
    pr=m.predict(sc.transform(F[ite]))
    recs_dir.append(recall_score(y[ite],pr,labels=range(8),average=None,zero_division=0))
per_origin=np.mean(recs_dir,axis=0)
for c in range(8):
    print(f"  {CLASS[c]}: {per_origin[c]:.3f}   (dir1={recs_dir[0][c]:.2f} dir2={recs_dir[1][c]:.2f})")
print(f"  worst origins: {[CLASS[c] for c in np.argsort(per_origin)[:3]]}")

# ---- F. within-seed covariance descriptor ----
print("\n=== F. within-seed covariance descriptor (streams MAT dev only) ===")
tc=time.perf_counter()
# pooled PCA basis on SNV pixels
rng=np.random.default_rng(7)
pool=[]
allpx=[]
for i,r in enumerate(recs):
    with h5py.File(r.mat_path,"r") as h:
        p=h["patch_chw"][()]; msk=np.asarray(h["crop_mask"][()]).squeeze()>0.5
    px=snv(p[msk].astype(np.float64))
    allpx.append(px)
    idx=rng.choice(px.shape[0],min(40,px.shape[0]),replace=False)
    pool.append(px[idx])
pool=np.vstack(pool)
K=16
basis=PCA(n_components=K, random_state=0).fit(pool)
iu=np.triu_indices(K)
seed_cov=np.empty((len(recs), iu[0].size)); seed_mean=np.empty((len(recs),392))
for i,px in enumerate(allpx):
    z=basis.transform(px)               # (n_px, K)
    C=np.cov(z,rowvar=False)            # (K,K)
    seed_cov[i]=C[iu]
    seed_mean[i]=allpx[i].mean(0)  # note: mean of SNV pixels ~0; use raw mean instead
# raw mean spectra already in X; use sg1(X)
print(f"[F load] {len(recs)} seeds, PCA-{K} cov desc ({time.perf_counter()-tc:.1f}s)")
featsF={
    "mean_sg1(ref)": sg1(X),
    "cov_desc_only": seed_cov,
    "mean_sg1 + cov_desc": np.hstack([sg1(X), seed_cov]),
}
print(f"{'feature':22} {'LOBO_BA':>8} {'xcube_BA':>8} {'xcube_NLL':>9}")
for name,feat in featsF.items():
    yl,Pl=lobo_proba(feat,lda); yc,Pc=crosscube_proba(feat,lda)
    print(f"{name:22} {balanced_accuracy_score(yl,Pl.argmax(1)):8.4f} "
          f"{balanced_accuracy_score(yc,Pc.argmax(1)):8.4f} "
          f"{multiclass_metrics(yc,Pc)['negative_log_likelihood']:9.3f}")
print(f"\n[done] {time.perf_counter()-t0:.1f}s")
