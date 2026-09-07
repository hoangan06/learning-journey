"""XGBoost doi chieu voi scorecard, tren cung bo fold va cung bo chia bin.

Chay:  python src/tree_model.py

Cau hoi khong phai "cay co manh hon khong" ma la "phan manh hon den tu dau". Khoi 3
da do duoc rang khong bien nao con hinh dang phi don dieu co gia tri, nen neu
XGBoost vuot scorecard thi phan vuot chi con ba cho co the den: tuong tac giua cac
bien (M2, cay tren cot WOE), cho cat bin (M3, cay tren bien goc), va bien da bi loai
(M4, cay tren tat ca).

Hai dieu kien de phep so sanh con y nghia. Ba model dung DUNG bo fold cua
cv_check.make_folds, nho vay phuong sai do fold (bien do 0,021 Gini o bo nay) triet
tieu khi tru theo cap. Va moi model chay tren cot WOE deu tinh lai WOE trong tung
fold qua woe_design, dung nhu scorecard lam tu khoi 3.

Mot thien lech co y: sieu tham so cua cay chon bang chinh bo CV dung de bao cao, con
scorecard khong duoc uu ai gi. Chon huong thien lech nguoc voi ket luan minh mong doi
la cach re nhat de mot phep so sanh tu bao ve duoc.
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


def woe_design(bins, vars_, them=None):
    """Ham dung ma tran WOE RIENG cho tung fold, de cay khong nhin nhan fold danh gia.

    Ban dau ba model cay deu nhan thang bang features_woe_mono, tuc cot WOE tinh tren
    tron tap train. M3 va M4 chay tren bien goc nen khong sao, nhung M2 va cay depth=1
    chay tren chinh cot WOE do. Do duoc bang logistic, cung bo fold: WOE tinh trong
    fold cho 0,71507 con WOE toan train cho 0,71570, chenh 0,00064, trong do Laplace
    cua woe_of chi dang 0,00001.

    Muc anh huong khac han giua hai lop model: logistic dung tri so WOE lam log-odds
    nen moi xe dich do nhan gay ra deu di thang vao du bao, con cay chi dung WOE lam
    thu tu roi tu cat lai. Vi vay khong duoc lay so cua ben nay suy ra cho ben kia.

    them: cot ghep them nguyen ven, khong qua WOE, cho cac phep "chi them mot cot".
    """
    tr = bins[bins.split == "train"].reset_index(drop=True)
    extra = None if them is None else np.asarray(them, dtype=float)
    if extra is not None and extra.ndim == 1:
        extra = extra.reshape(-1, 1)

    def build(fit_idx, va_idx):
        # _design tinh WOE tren rieng phan fit roi ap sang phan danh gia.
        Xa, Xb = cv_check._design(tr.iloc[fit_idx], tr.iloc[va_idx], vars_)
        if extra is not None:
            Xa = np.column_stack([Xa, extra[fit_idx]])
            Xb = np.column_stack([Xb, extra[va_idx]])
        return Xa, Xb

    return build


# --- do luong ------------------------------------------------------------------

def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def cv_gini(X, y, params, k=5, seed=None, seed_model=None, design=None):
    """Gini tren tung fold, dung dung bo fold cua cv_check.

    params PHAI chua n_estimators: ham nay khong dung early stopping, va do la mot
    quyet dinh do bang so. Ban dau no co early_stopping_rounds=30 tren 20% cuoi cua
    fold-fit, nhung eval set do chi ~16.800 dong voi ~1.120 ca duong nen AUC tren no
    rat nhieu: co fold dung o vong 8 voi learning_rate 0,05, va doi random_state thi
    Gini dao dong bien do 0,00458 so voi 0,00036 khi so vong co dinh. No con lam cau
    hinh thang doi giua Windows va Linux. Cai gia phai tra la so vong thanh mot sieu
    tham so nua, duoc khai truoc theo quy tac learning_rate * n_estimators ~ 25 chu
    khong do tim.

    seed_model doi rieng random_state cua XGBoost ma giu nguyen bo fold.
    design: ham (fit_idx, va_idx) -> (Xa, Xb) khi feature phai tinh lai trong tung
    fold; bat buoc voi model chay tren cot WOE, xem woe_design. Khi co design thi X
    bi bo qua.
    """
    import xgboost as xgb
    if "n_estimators" not in params:
        raise ValueError("params phai chua n_estimators; ham nay khong con early stopping")
    y = np.asarray(y)
    if design is None:
        X = np.asarray(X, dtype=float)
    folds = cv_check.make_folds(len(y), k, seed)
    out = []
    for f in range(k):
        va = folds[f]
        fit = np.concatenate([folds[i] for i in range(k) if i != f])
        Xa, Xb = (X[fit], X[va]) if design is None else design(fit, va)
        m = xgb.XGBClassifier(tree_method="hist", eval_metric="auc",
                              random_state=config.SEED if seed_model is None else seed_model,
                              **params)
        m.fit(Xa, y[fit], verbose=False)
        out.append(gini(y[va], m.predict_proba(Xb)[:, 1]))
    return np.array(out)


# So vong khai truoc theo quy tac learning_rate * n_estimators ~ 25, khong do tim.
GRID = [
    dict(max_depth=3, learning_rate=0.10, n_estimators=250, min_child_weight=50, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.10, n_estimators=250, min_child_weight=50, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, learning_rate=0.05, n_estimators=500, min_child_weight=50, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, learning_rate=0.05, n_estimators=500, min_child_weight=20, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.05, n_estimators=500, min_child_weight=200, subsample=0.8, colsample_bytree=0.8),
]


def search(X, y, grid=None, k=5, design=None):
    """Chon sieu tham so bang chinh 5-fold CV trong train. Tra ve (params, gini)."""
    best = None
    for prm in (grid or GRID):
        g = cv_gini(X, y, prm, k=k, design=design)
        if best is None or g.mean() > best[1].mean():
            best = (prm, g)
        print(f"   {prm}  ->  {g.mean():.5f}")
    return best
