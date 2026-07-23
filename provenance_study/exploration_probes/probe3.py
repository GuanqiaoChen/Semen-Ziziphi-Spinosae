"""Probe 3: is the cross-acquisition calibration collapse RECOVERABLE, and can
an acquisition-shift-aware calibration (fit WITHOUT cross-cube labels) approach
the oracle? Development batches 0-7 only, CSV mean spectra only. Exploratory.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"d:\projects\Semen-Ziziphi-Spinosae"); sys.path.insert(0, str(ROOT))
from provenance_study.core import discover_manifest, load_csv_split, multiclass_metrics  # noqa

man=discover_manifest(ROOT/"data"); dev=load_csv_split(man,split="development",verify_hashes=False)
X,y,recs=dev.X,dev.y,dev.records
rep=np.array([r.replicate for r in recs]); batch=np.array([r.constructed_batch for r in recs])
def sg1(A): return savgol_filter(A,15,2,deriv=1,axis=1,mode="interp")
def lda(): return LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
def lr(): return LogisticRegression(C=1.0,max_iter=5000,tol=1e-4)
F=sg1(X)
TGRID=np.exp(np.linspace(np.log(0.3),np.log(6.0),200))

def temp_apply(logits,T):
    z=logits/T; z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def best_T(logits,yy):
    nll=[-np.log(np.clip(temp_apply(logits,T)[np.arange(len(yy)),yy],1e-15,1)).mean() for T in TGRID]
    return float(TGRID[int(np.argmin(nll))])

def member_logits(feat, mfn, Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr); m=mfn().fit(sc.transform(Xtr),ytr)
    return np.log(np.clip(m.predict_proba(sc.transform(Xte)),1e-12,1))

def crosscube_logits(feat,mfn):
    L,Y=[],[]
    for tr,te in ((1,2),(2,1)):
        L.append(member_logits(feat,mfn,feat[rep==tr],y[rep==tr],feat[rep==te])); Y.append(y[rep==te])
    return np.vstack(L), np.concatenate(Y)

def shift_aware_T(feat,mfn):
    """Fit T on within-TRAIN-cube leave-one-batch-out preds (simulated shift),
    pooled over both training cubes. Uses NO cross-cube test labels."""
    L,Y=[],[]
    for cube in (1,2):
        idx=np.where(rep==cube)[0]
        for b in np.unique(batch[idx]):
            trm=idx[batch[idx]!=b]; tem=idx[batch[idx]==b]
            L.append(member_logits(feat,mfn,feat[trm],y[trm],feat[tem])); Y.append(y[tem])
    return best_T(np.vstack(L),np.concatenate(Y))

def same_domain_T(feat,mfn):
    """Naive iid temperature: pooled leave-one-batch-out ignoring cube."""
    L,Y=[],[]
    for b in range(8):
        trm=batch!=b; tem=batch==b
        L.append(member_logits(feat,mfn,feat[trm],y[trm],feat[tem])); Y.append(y[tem])
    return best_T(np.vstack(L),np.concatenate(Y))

def report(tag,logits,yy,T):
    P=temp_apply(logits,T); m=multiclass_metrics(yy,P)
    print(f"  {tag:22} T={T:5.2f}  BA={m['balanced_accuracy']:.4f} "
          f"NLL={m['negative_log_likelihood']:.3f} ECE={m['expected_calibration_error']:.3f} "
          f"Brier={m['multiclass_brier_score']:.3f}")

print("=== Cross-cube calibration recovery ===")
for mname,mfn in (("sg1-LDA",lda),("sg1-LR",lr)):
    Lc,Yc=crosscube_logits(F,mfn)
    print(f"\n[{mname}] cross-cube (pooled both directions, n={len(Yc)})")
    report("T=1 (uncalibrated)",Lc,Yc,1.0)
    report("naive iid T",Lc,Yc,same_domain_T(F,mfn))
    report("shift-aware T (ours)",Lc,Yc,shift_aware_T(F,mfn))
    report("oracle T (uses test)",Lc,Yc,best_T(Lc,Yc))

# ensemble lda+lr avg proba
print("\n[sg1-LDA+LR ensemble] cross-cube")
def ens_cross():
    Ps,Y=[],[]
    for tr,te in ((1,2),(2,1)):
        pa=[]
        for mfn in (lda,lr):
            sc=StandardScaler().fit(F[rep==tr]); m=mfn().fit(sc.transform(F[rep==tr]),y[rep==tr])
            pa.append(m.predict_proba(sc.transform(F[rep==te])))
        Ps.append(np.mean(pa,0)); Y.append(y[rep==te])
    return np.log(np.clip(np.vstack(Ps),1e-12,1)), np.concatenate(Y)
Le,Ye=ens_cross()
report("T=1",Le,Ye,1.0); report("oracle T",Le,Ye,best_T(Le,Ye))
print("\n[done]")
