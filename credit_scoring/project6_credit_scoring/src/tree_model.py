"""XGBoost doi chieu voi scorecard, tren cung bo fold va cung bo chia bin.

Chay:  python src/tree_model.py

Cau hoi cua buoc nay khong phai "cay co manh hon khong" ma la "phan manh hon den
tu dau". Khoi 3 da do duoc rang khong bien nao con hinh dang phi don dieu co gia
tri, nen neu XGBoost vuot scorecard thi phan vuot chi con ba cho co the den:

  1. tuong tac giua cac bien   -> so M2 (cay tren cot WOE) voi scorecard
  2. cho cat bin               -> so M3 (cay tren bien goc) voi M2
  3. bien da bi loai           -> so M4 (cay tren tat ca) voi M3

Ba model cay dung DUNG bo fold cua cv_check.make_folds, nho vay so sanh theo cap
voi scorecard con y nghia: phuong sai do fold (bien do 0,021 Gini o bo nay) trie
tieu khi tru theo tung fold.

Mot thien lech co y: sieu tham so cua cay duoc chon bang chinh 5-fold CV do, tuc
con so CV cua cay LAC QUAN hon thuc te, con scorecard thi khong duoc uu ai gi.
Neu voi thien lech nghieng ve phia cay ma cay van khong vuot duoc bao nhieu thi
ket luan cang chac. Chon huong thien lech nguoc lai voi ket luan minh mong doi la
cach re nhat de mot phep so sanh tu bao ve duoc.

Ca ba model deu fit tren tron 80% cua moi fold, dung bang scorecard. Xem docstring
cua cv_gini ve ly do bo early stopping: no khuech dai nhieu chu khong chong overfit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import config, cv_check
except ImportError:
    from src import config, cv_check


# --- bo feature ---------------------------------------------------------------

KEPT = ["revolving_util", "age", "debt_ratio_valid", "monthly_income",
        "late_30_59", "late_60_89", "late_90", "real_estate_loans", "dependents"]

FLAGS = ["flag_sentinel", "flag_income_missing", "flag_income_zero",
         "flag_dependents_missing", "flag_util_implausible", "flag_debt_ratio_invalid"]


def feature_sets(ap: pd.DataFrame, fw: pd.DataFrame) -> dict:
    """Ba ma tran feature, cung thu tu dong (theo id tang dan).

    XGBoost nhan NaN truc tiep nen bien goc khong can impute; day cung la mot
    khac biet dang noi: scorecard phai bien missing thanh mot bin rieng, con cay
    tu hoc huong di cho NaN o moi node.
    """
    ap = ap.sort_values("id").reset_index(drop=True)
    fw = fw.sort_values("id").reset_index(drop=True)
    assert (ap.id.values == fw.id.values).all(), "hai bang khong cung tap id"
    return {
        "M2_woe":  fw[[c for c in fw.columns if c.startswith("woe_")]],
        "M3_goc":  ap[KEPT + FLAGS],
        "M4_all":  ap[KEPT + FLAGS + ["open_credit_lines", "debt_ratio"]],
    }, ap.split.values, (1 - ap.target).values      # y = 1 la GOOD, giong scorecard


# --- do luong ------------------------------------------------------------------

def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def cv_gini(X, y, params, k=5, seed=None, seed_model=None):
    """Gini tren tung fold, dung dung bo fold cua cv_check.

    params PHAI chua n_estimators. Khong dung early stopping, va day la mot
    quyet dinh do bang so chu khong phai so thich.

    Ban dau ham nay co early_stopping_rounds=30 tren 20% cuoi cua fold-fit. Do
    lai thi eval set do chi khoang 16.800 dong voi ~1.120 ca duong, tuc AUC tren
    no rat nhieu, va voi learning_rate nho thi mot dao dong ngau nhien du de tao
    ra mot dinh gia som roi 30 vong sau khong vuot duoc. Hau qua do duoc, tren
    cau hinh depth5 lr0.05:

        bat early stopping : best_iteration moi fold = [31, 75, 8, 98, 156]
                             doi random_state -> Gini dao dong bien do 0,00458
        so vong co dinh    : doi random_state -> Gini dao dong bien do 0,00036

    Mot fold dung o vong 8 voi learning_rate 0,05 la mot model chua hoc gi. Early
    stopping o day khong chong overfit ma bien nhieu cua eval set thanh nhieu cua
    ket qua, gap 13 lan. No cung lam cau hinh thang doi giua hai may va lam chinh
    phep do "muc nhieu" khong tai lap duoc.

    Bo no di duoc them hai thu. Cay gio fit tren TRON 80% tap train, dung bang
    scorecard, nen mat mot thien lech khong tach duoc. Va bang test o cuoi
    notebook dung dung so vong cua bang CV, nen hai bang doc cheo nhau duoc.

    Cai gia phai tra: so vong thanh mot sieu tham so nua. No duoc khai truoc theo
    quy tac learning_rate * n_estimators ~ 25 chu khong do tim, de khong bien
    thanh mot vong toi uu nua tren chinh tap bao cao.

    seed_model doi rieng random_state cua XGBoost ma GIU NGUYEN bo fold, de do
    xem ket qua dao dong bao nhieu vi subsample/colsample chu khong vi chia fold.
    """
    import xgboost as xgb
    if "n_estimators" not in params:
        raise ValueError("params phai chua n_estimators; ham nay khong con early stopping")
    X = np.asarray(X, dtype=float)
    folds = cv_check.make_folds(len(X), k, seed)
    out = []
    for f in range(k):
        va = folds[f]
        fit = np.concatenate([folds[i] for i in range(k) if i != f])
        m = xgb.XGBClassifier(tree_method="hist", eval_metric="auc",
                              random_state=config.SEED if seed_model is None else seed_model,
                              **params)
        m.fit(X[fit], y[fit], verbose=False)
        out.append(gini(y[va], m.predict_proba(X[va])[:, 1]))
    return np.array(out)


# So vong khai truoc theo quy tac learning_rate * n_estimators ~ 25, khong do tim.
GRID = [
    dict(max_depth=3, learning_rate=0.10, n_estimators=250, min_child_weight=50, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.10, n_estimators=250, min_child_weight=50, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, learning_rate=0.05, n_estimators=500, min_child_weight=50, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, learning_rate=0.05, n_estimators=500, min_child_weight=20, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.05, n_estimators=500, min_child_weight=200, subsample=0.8, colsample_bytree=0.8),
]


def search(X, y, grid=None, k=5):
    """Chon sieu tham so bang chinh 5-fold CV trong train. Tra ve (params, gini)."""
    best = None
    for prm in (grid or GRID):
        g = cv_gini(X, y, prm, k=k)
        if best is None or g.mean() > best[1].mean():
            best = (prm, g)
        print(f"   {prm}  ->  {g.mean():.5f}")
    return best
