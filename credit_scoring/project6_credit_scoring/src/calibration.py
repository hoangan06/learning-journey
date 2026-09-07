"""Do va sua calibration: PD co phai con so that khong.

Chay:  python src/calibration.py

Quy uoc cua file nay KHAC cac file truoc va phai noi ro. Khoi 3 va 4 model
log-odds cua GOOD vi WOE dinh huong theo good, nen `p` o do la P(good). O day
moi ham nhan (y, p) voi y = 1 nghia la BAD va p = PD, vi calibration la phat
bieu ve PD: "cham 5% cho mot nghin nguoi thi khoang nam muoi nguoi vo no".
Doi qua lai bang y = target va p = 1 - predict_proba[:, 1].

Gini bat bien khi dao ca hai nen khong anh huong gi den con so xep hang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# --- do -----------------------------------------------------------------------

def brier(y, p) -> float:
    """Sai so binh phuong trung binh tren xac suat."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability(y, p, k: int = 10) -> pd.DataFrame:
    """Bang decile: PD du bao trung binh so voi bad rate thuc, tung nhom.

    Chia theo phan vi cua p chu khong theo khoang deu, vi phan bo PD lech manh
    ve phia 0 nen khoang deu se cho vai nhom rong khong.

    Xep hang bang method="first" nen cac dong CO CUNG p bi tach sang hai decile
    khac nhau theo thu tu dong. Doi voi scorecard thi khong dang ke (10.589 muc
    PD phan biet tren 22.500 dong), nhung doi voi dau ra cua isotonic thi dang ke:
    chi con 74 muc, va 59,98% so dong nam trong mot khoi hoa bi bien decile cat
    ngang. Brier khong bi anh huong vi no tinh tren tung dong, nhung phan ra
    Reliability/Resolution cua isotonic phai doc kem canh bao nay.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    r = pd.qcut(pd.Series(p).rank(method="first"), k, labels=False)
    d = pd.DataFrame({"y": y, "p": p, "dec": r})
    out = d.groupby("dec").agg(n=("y", "size"), pd_du_bao=("p", "mean"), bad_rate_thuc=("y", "mean"))
    out["lech"] = out.pd_du_bao - out.bad_rate_thuc
    return out


def murphy(y, p, k: int = 10) -> dict:
    """Phan ra Brier = Reliability - Resolution + Uncertainty, kem PHAN DU.

    Reliability nho thi tot, Resolution lon thi tot, Uncertainty = p_ngang*(1-p_ngang)
    va khong giam duoc. Nhung dang thuc ba so hang chi DUNG CHINH XAC khi p la hang so
    trong tung nhom; chia theo decile thi p bien thien ben trong nhom va con lai mot so
    hang thu tu:

        du = Brier - (Reliability - Resolution + Uncertainty)
           = SUM_k w_k * [ var_k(p) - 2*cov_k(p, y) ]

    Tren `oot` voi k = 10: du = -1,472e-03, tach ra thanh var trong nhom +4,252e-03 va
    so hang cheo -5,723e-03. Do lon cua no gap gan ba muoi lan chinh Reliability, nen
    ham nay bao du ra thay vi de nguoi doc tin vao mot dang thuc khong thoa. Reliability
    va Resolution chi so sanh duoc GIUA cac model tren cung mot k.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    tab = reliability(y, p, k)
    n = len(y)
    w = tab.n / n
    rel = float((w * (tab.pd_du_bao - tab.bad_rate_thuc) ** 2).sum())
    ybar = float(y.mean())
    res = float((w * (tab.bad_rate_thuc - ybar) ** 2).sum())
    unc = float(ybar * (1 - ybar))
    b = brier(y, p)
    return {"brier": b, "reliability": rel, "resolution": res, "uncertainty": unc,
            "du": b - (rel - res + unc), "k": k}


# --- sua ----------------------------------------------------------------------

def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class Platt:
    """Platt scaling: mot logistic mot chieu tu logit(PD) sang nhan.

    Fit tren logit(p) chu khong tren p: tren thang log-odds thi phep hieu chinh dung la
    mot phep dan va dich, con fit tren p la bat mot ham hai tham so lam them viec dao
    nguoc sigmoid cua chinh model.

    Dung dung cau hinh solver cua `scorecard.fit_logit`. Ban dau lop nay de C=1e6 voi
    lbfgs va tol mac dinh, dung o |gradient| = 1,65; phep kiem MLE bat duoc no, trung
    binh du bao tren chinh tap fit cho 6,68201% so voi bad rate 6,68444%. Voi C=inf,
    newton-cholesky va tol=1e-12 thi |gradient| xuong 4,4e-12 va hai con so bang nhau.
    """

    def __init__(self):
        self.lr = LogisticRegression(C=np.inf, solver="newton-cholesky",
                                     max_iter=1000, tol=1e-12)

    def fit(self, y, p):
        self.lr.fit(_logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
        self.a = float(self.lr.coef_[0][0])
        self.b = float(self.lr.intercept_[0])
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    """Isotonic regression: ham don dieu bat ky tu PD sang nhan.

    out_of_bounds="clip" de PD ngoai khoang da thay luc fit khong tra ve NaN.
    Linh hoat hon Platt nhung nhieu bac tu do hon, va o day chi co 1.504 ca duong
    de fit nen rui ro overfit la that chu khong phai ly thuyet.
    """

    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y, p):
        self.ir.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, p):
        return self.ir.predict(np.asarray(p, dtype=float))


# --- phep kiem ----------------------------------------------------------------

def rank_invariance(y, p_truoc, p_sau) -> dict:
    """Kiem mot phep hieu chinh co doi thu hang khong.

    Cau ban dau trong ghi chu nen tang la "ca ba phep hieu chinh deu don dieu nen
    khong bao gio doi Gini/AUC/KS". Phep kiem nay dung de kiem cau do va no da bat
    duoc cau do noi qua: dung cho Platt, sai cho isotonic.

    Platt la don dieu NGHIEM NGAT nen bat buoc giu nguyen Gini tuyet doi; lech du
    mot chu so la loi cai dat. Isotonic chi don dieu KHONG NGHIEM NGAT: no nen
    10.589 muc PD xuong 74, hai nguoi khac diem truoc do co the thanh bang diem,
    ma AUC tinh moi cap hoa la 0,5 nen Gini giam that. Do la ly do bao ca spearman
    va so muc: spearman ~ 1 ma Gini van lech thi nguyen nhan la hoa chu khong phai
    sai thu tu, va cot so muc chi thang ra cho tao hoa.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    g1 = 2 * roc_auc_score(y, p_truoc) - 1
    g2 = 2 * roc_auc_score(y, p_sau) - 1
    rho = float(spearmanr(p_truoc, p_sau).statistic)
    return {"gini_truoc": g1, "gini_sau": g2, "lech_gini": g2 - g1, "spearman": rho,
            "so_muc_truoc": int(len(np.unique(p_truoc))), "so_muc_sau": int(len(np.unique(p_sau)))}


def main():
    try:
        import config, scorecard
    except ImportError:
        from src import config, scorecard
    bins, woe, _ = scorecard.load_data(mono=True)
    W = scorecard.woe_matrix(bins, woe)
    tr, te, oo = (bins.split == s for s in ("train", "test", "oot"))
    lr, coef, se = scorecard.fit_logit(W[tr], (1 - bins.target)[tr])
    pd_ = lambda m: 1 - lr.predict_proba(W[m])[:, 1]
    y = bins.target.values
    for ten, m in [("train", tr), ("test", te), ("oot", oo)]:
        p = pd_(m)
        mu = murphy(y[m.values], p)
        print(f"{ten:6s} n={int(m.sum()):6d}  PD TB={p.mean()*100:6.3f}%  bad rate={y[m.values].mean()*100:6.3f}%"
              f"  Brier={mu['brier']:.5f}  rel={mu['reliability']:.6f}  res={mu['resolution']:.5f}")


if __name__ == "__main__":
    main()
