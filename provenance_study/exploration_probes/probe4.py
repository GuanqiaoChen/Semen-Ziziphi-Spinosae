"""Probe 4: can UNSUPERVISED adaptation to the incoming acquisition cube rescue
cross-cube origin accuracy? (Target cube used only via its unlabeled features.)
Development batches 0-7 only. CSV mean spectra. Exploratory.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(r"d:\projects\Semen-Ziziphi-Spinosae"); sys.path.insert(0, str(ROOT))
from provenance_study.core import discover_manifest, load_csv_split, multiclass_metrics  # noqa
CLASS=("HBS","HBX","HNA","HNX","NX","SXD","SXQ","XJH")
man=discover_manifest(ROOT/"data"); dev=load_csv_split(man,split="development",verify_hashes=False)
X,y,recs=dev.X,dev.y,dev.records
rep=np.array([r.replicate for r in recs]); batch=np.array([r.constructed_batch for r in recs])
def sg1(A): return savgol_filter(A,15,2,deriv=1,axis=1,mode="interp")
F=sg1(X)

def lda(): return LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")

def whiten_matrix(Z, shrink=0.3):
    C=np.cov(Z,rowvar=False); C=(1-shrink)*C+shrink*np.eye(C.shape[0])*np.trace(C)/C.shape[0]
    vals,vecs=np.linalg.eigh(C); vals=np.clip(vals,1e-9,None)
    return vecs@np.diag(vals**-0.5)@vecs.T

def run(method):
    thetas=[]; perc=[]
    for tr,te in ((1,2),(2,1)):
        itr,ite=rep==tr,rep==te
        Ftr,Fte=F[itr],F[ite]; ytr,yte=y[itr],y[ite]
        if method=="baseline":
            sc=StandardScaler().fit(Ftr); Xtr,Xte=sc.transform(Ftr),sc.transform(Fte)
        elif method=="target_std":                       # per-cube moment matching
            Xtr=StandardScaler().fit(Ftr).transform(Ftr)
            Xte=StandardScaler().fit(Fte).transform(Fte)
        elif method=="mean_center_target":               # remove per-cube mean only
            s=StandardScaler().fit(Ftr)
            Xtr=(Ftr-Ftr.mean(0))/s.scale_; Xte=(Fte-Fte.mean(0))/s.scale_
        elif method=="pca50_coral":                      # full-cov align in PCA-50
            pca=PCA(50,random_state=0).fit(Ftr)
            Ztr=pca.transform(Ftr); Zte=pca.transform(Fte)
            Wtr=whiten_matrix(Ztr); Wte=whiten_matrix(Zte)
            Xtr=(Ztr-Ztr.mean(0))@Wtr; Xte=(Zte-Zte.mean(0))@Wte
        elif method=="pca50_target_std":
            pca=PCA(50,random_state=0).fit(Ftr)
            Ztr=pca.transform(Ftr); Zte=pca.transform(Fte)
            Xtr=StandardScaler().fit(Ztr).transform(Ztr)
            Xte=StandardScaler().fit(Zte).transform(Zte)
        else: raise ValueError(method)
        m=lda().fit(Xtr,ytr); pr=m.predict(Xte)
        thetas.append(balanced_accuracy_score(yte,pr))
        perc.append(recall_score(yte,pr,labels=range(8),average=None,zero_division=0))
    per=np.mean(perc,0)
    return np.mean(thetas), thetas, per

print(f"{'method':20}{'cc_theta':>9}{'dir1':>8}{'dir2':>8}   worst-origin recalls")
for method in ("baseline","mean_center_target","target_std","pca50_target_std","pca50_coral"):
    th,ths,per=run(method)
    worst=", ".join(f"{CLASS[c]}={per[c]:.2f}" for c in np.argsort(per)[:3])
    print(f"{method:20}{th:9.4f}{ths[0]:8.4f}{ths[1]:8.4f}   {worst}")
