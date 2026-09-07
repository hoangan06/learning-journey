"""Quy doi model logistic tren WOE thanh mot bang diem scorecard.

Chay:  python src/scorecard.py

Cac buoc:
  1. Doc features_woe (ma tran model) va woe_lookup (bang tra WOE) tu credit.db.
  2. Fit logistic KHONG regularization tren train, model log-odds cua GOOD.
  3. Quy doi he so thanh diem theo PDO / base odds.
  4. Cham diem toan bo bang va xuat ma ly do.

Quy uoc dau: model log-odds cua GOOD (y = 1 - target) chu khong phai cua BAD.
WOE trong du an nay tinh la ln(pct_good / pct_bad), tuc WOE cao = an toan. Neu
model log-odds cua BAD thi moi he so se am va bang diem phai doi dau lai. Model
GOOD giu duoc quy tac kiem tra don gian: MOI he so phai duong, mot he so am la
dau hieu bien do bi bien khac nuot mat va can xem lai.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import config
except ImportError:  # khi import tu notebook
    from src import config


# --- doc du lieu ------------------------------------------------------------

def _ro_uri(path):
    """Duong dan -> URI SQLite read-only.

    Phai di qua Path.as_uri() chu khong noi chuoi bang tay: tren Windows duong
    dan la D:\\hoc-ai\\... , ma dau gach nguoc khong phai ky tu hop le trong URI
    nen f"file:{path}?mode=ro" se hong. as_uri() cho file:///D:/hoc-ai/... va
    tu ma hoa cac ky tu dac biet trong ten thu muc.

    Mo read-only vi buoc nay chi doc. Day cung la lop bao ve: mot lenh ghi lo
    tay se bao loi thay vi dong vao credit.db.
    """
    from pathlib import Path
    return Path(path).resolve().as_uri() + "?mode=ro"


def load_data(db_path=None, drop=None, mono=False):
    """Doc ba bang can cho scorecard tu credit.db.

    mono=True doc bo bang *_mono cua khoi 4 (bin da gop cho don dieu) thay cho bo
    bang cua khoi 2. Ca hai bo cung cau truc nen moi buoc phia sau khong doi.

    Ma tran model o day duoc dung LAI tu row_bins + woe_lookup chu khong lay san
    bang features_woe cua khoi 2. Hai ly do: ten bien trong row_bins/woe_lookup
    va ten cot trong features_woe khong khop nhau tuyet doi (cot pivot dat theo
    ten cot goc, bien dat theo ten ban da dan xuat), va dung lai tu bang tra cho
    them mot phep doi chieu doc lap nua - notebook 03 kiem hai ma tran co trung
    khop khong. Bang diem cung sinh ra tu chinh woe_lookup nay nen ca ba thu
    (WOE, ma tran, diem) chac chan cung mot nguon.
    """
    db_path = config.DB_PATH if db_path is None else db_path
    drop = config.SCORECARD_DROP if drop is None else drop
    sfx = "_mono" if mono else ""
    con = sqlite3.connect(_ro_uri(db_path), uri=True)
    try:
        long = pd.read_sql(f"SELECT id, variable, bin FROM row_bins{sfx}", con)
        base = pd.read_sql(f"SELECT id, split, target FROM {config.TABLE}", con)
        woe = pd.read_sql(f"SELECT variable, bin, n, n_bad, woe FROM woe_lookup{sfx}", con)
        wide_sql = pd.read_sql(f"SELECT * FROM features_woe{sfx}", con)
    finally:
        con.close()
    bins = long.pivot(index="id", columns="variable", values="bin")
    bins = bins.drop(columns=[c for c in drop if c in bins.columns])
    bins = base.set_index("id").join(bins)
    woe = woe[~woe.variable.isin(drop)].reset_index(drop=True)
    return bins, woe, wide_sql


def woe_matrix(bins, woe):
    """Doi bang bin sang ma tran WOE bang cach tra woe_lookup."""
    cols = sorted(woe.variable.unique())
    out = pd.DataFrame(index=bins.index)
    for v in cols:
        m = dict(zip(woe.loc[woe.variable == v, "bin"], woe.loc[woe.variable == v, "woe"]))
        col = bins[v].map(m)
        if col.isna().any():
            raise ValueError(f"{v}: {int(col.isna().sum())} dong co bin khong co trong woe_lookup")
        out[v] = col.values
    return out


# --- fit --------------------------------------------------------------------

def fit_logit(X, y):
    """Logistic KHONG phat, uoc luong hop ly cuc dai that su, kem SE tu Hessian.

    Ban dau ham nay dung LogisticRegression(C=1e12, max_iter=2000), voi y dinh
    "phat lon den muc coi nhu khong phat". Y dinh dung nhung cai dat sai: C lon
    lam ham muc tieu rat phang, con tol mac dinh 1e-4 thi lbfgs dung som. Do
    duoc tren bo nay: |gradient| tai nghiem = 5,67 thay vi ~0, va sum(p) - sum(y)
    = +5,67 trong khi MLE bat buoc phai bang 0.

    Cho nen "PD du bao trung binh 6,678% so voi bad rate 6,684%" o bao cao khoi 3
    KHONG phai mot thanh tich lam tron dep - no chinh la phep kiem hoi tu da fail
    ma bi doc nham thanh thanh tich. Voi nghiem thuc su thi hai con so bang nhau
    den chu so thu chin, vi do la phuong trinh chuan tac cua logistic.

    penalty=None bo han so hang phat thay vi lam no nho; newton-cholesky voi
    tol=1e-12 dua |gradient| ve 6e-10. Sai lech he so so voi ban cu toi 0,012
    (dependents, ~5%), du de doi mot dong bang diem.
    """
    # C=np.inf thay cho penalty=None: sklearn 1.8 deprecate penalty= va chi ra
    # dung C=np.inf. Hai cach cho he so trung den chu so thu sau tren sklearn 1.7.
    lr = LogisticRegression(C=np.inf, solver="newton-cholesky",
                            max_iter=1000, tol=1e-12).fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    w = p * (1 - p)
    Xd = np.column_stack([np.ones(len(X)), np.asarray(X)])
    # cov = (X' W X)^-1, dang chuan cua ma tran hiep phuong sai uoc luong MLE
    cov = np.linalg.inv((Xd * w[:, None]).T @ Xd)
    se = np.sqrt(np.diag(cov))
    coef = np.r_[lr.intercept_, lr.coef_[0]]
    return lr, coef, se


# --- quy doi diem -----------------------------------------------------------

def scaling_constants(pdo=None, score_base=None, odds_base=None):
    """Factor va Offset cua thang diem.

    Dat score = Offset + Factor * ln(odds). Hai rang buoc:
      score_base       = Offset + Factor * ln(odds_base)
      score_base + pdo = Offset + Factor * ln(2 * odds_base)
    Tru ve nhau, ln(2*odds) - ln(odds) = ln 2, con lai pdo = Factor * ln 2.
    """
    pdo = config.PDO if pdo is None else pdo
    score_base = config.SCORE_BASE if score_base is None else score_base
    odds_base = config.ODDS_BASE if odds_base is None else odds_base
    factor = pdo / math.log(2)
    offset = score_base - factor * math.log(odds_base)
    return factor, offset


def build_points(coef, cols, woe, factor, offset):
    """Chia diem ve tung bin.

    score = Offset + Factor * (b0 + sum_j b_j * WOE_j). Tong nay phai tach thanh
    mot so hang cho moi bien de con cham diem tren giay va xuat ma ly do, nen b0
    va Offset duoc chia deu cho n bien:

        diem(bien j, bin i) = (b_j * WOE_ij + b0/n) * Factor + Offset/n

    Cong n so hang lai dung bang score. Chia deu la quy uoc thuan tien, khong
    phai ket qua toan hoc: no chi doi diem giua cac bien chu khong doi tong.
    """
    b0, betas = coef[0], dict(zip(cols, coef[1:]))
    n = len(cols)
    out = woe.copy()
    out["beta"] = out.variable.map(betas)
    out["diem_raw"] = (out.beta * out.woe + b0 / n) * factor + offset / n
    out["diem"] = out.diem_raw.round().astype(int)
    return out.sort_values(["variable", "bin"]).reset_index(drop=True)


def apply_points(bins, points, col="diem"):
    """Cham diem: tra bang theo (bien, bin) roi cong lai."""
    total = np.zeros(len(bins))
    for var, tab in points.groupby("variable"):
        m = dict(zip(tab.bin, tab[col]))
        v = bins[var].map(m)
        if v.isna().any():
            raise ValueError(f"{var}: {int(v.isna().sum())} dong co bin khong co trong bang diem")
        total += v.values
    return pd.Series(total, index=bins.index, name="score")


def score_to_pd(score, factor, offset):
    """Nguoc lai: diem -> log-odds good -> PD. Dung de kiem tra round-trip."""
    return 1.0 / (1.0 + np.exp((np.asarray(score) - offset) / factor))


def is_special(b):
    """Bin dac biet: missing, sentinel, gia tri bat kha tin. Nhan dat theo tien to."""
    return str(b).startswith(config.SPECIAL_BIN_PREFIXES)


# --- ma ly do ---------------------------------------------------------------

def reason_codes(bins, points, top_k=3, col="diem"):
    """Top-k bien lam mat nhieu diem nhat so voi bin THUONG tot nhat cua bien do.

    Moc so la diem cao nhat ma bien do co the cho, NHUNG chi tinh cac bin thuong:
    bin dac biet (X_, 9_) bi loai khoi tap ung vien lam moc. Day la mot lan sua.
    Ban dau moc lay max tren moi bin, va do la mot loi nghiep vu: moc cua
    dependents roi vao 9_MISSING con moc cua monthly_income roi vao X_ZERO, nen
    ma ly do noi voi khach hang la "anh mat diem vi ho so cua anh KHONG bi thieu
    du lieu". Do duoc: 365 tren 8.291 ho so bi tu choi (4,4%) nhan mot ma nhu vay.

    Bin dac biet la trang thai cua BAN GHI chu khong phai trang thai tin dung, nen
    no khong the lam muc chuan de so. Ho so dang nam o mot bin dac biet co diem
    cao hon moc thi ra thieu diem am va bi loai boi dieu kien L > 0 o duoi.

    Cach khac la so voi bin trung binh (neutral) hoac voi mot ho so tham chieu;
    ECOA Regulation B khong quy dinh cong thuc, chi doi hoi "principal reasons"
    phai la ly do that su dan den quyet dinh. Lay max cua bin thuong la cach de
    bao ve nhat vi no tra loi dung cau khach hang hoi (toi mat diem o dau) va cau
    tra loi tro toi mot thu khach hang doi duoc.
    """
    thuong = points[~points.bin.map(is_special)]
    best = thuong.groupby("variable")[col].max().to_dict()
    lack = {}
    for var, tab in points.groupby("variable"):
        m = dict(zip(tab.bin, tab[col]))
        lack[var] = best[var] - bins[var].map(m).values
    L = pd.DataFrame(lack, index=bins.index)
    # kind='stable' vi co nhieu muc thieu diem trung nhau GIUA CAC BIEN (vi du
    # late_30_59 bin 9_SENTINEL va late_90 bin 3-4 cung thieu 52 diem); khong co
    # no thi thu tu ma ly do phu thuoc vao thu tu cot.
    order = np.argsort(-L.values, axis=1, kind="stable")[:, :top_k]
    cols = np.array(L.columns)
    rows = []
    for i, idx in enumerate(order):
        rows.append([(cols[j], int(L.values[i, j]), bins.iloc[i][cols[j]])
                     for j in idx if L.values[i, j] > 0])
    return rows


# --- do luong ---------------------------------------------------------------

def gini(y, s):
    from sklearn.metrics import roc_auc_score
    return 2 * roc_auc_score(y, s) - 1


def ks(y, s):
    """KS = khoang cach doc lon nhat giua hai ham phan phoi tich luy good va bad.

    Cong don theo TUNG MUC DIEM chu khong theo tung dong. Diem cua scorecard chi
    nhan huu han gia tri (69.016 muc tren 149.999 dong o bo nay), nen di theo
    tung dong thi phep max co the dung giua mot nhom diem bang nhau va lay mot
    con so khong ung voi nguong cat nao co that. O bo nay hai cach cho ket qua
    giong nhau den 6 chu so, nhung trung nhau khong phai la mot dam bao.
    """
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)}).sort_values("s", ascending=False)
    g = d.groupby("s", sort=False).y.agg(["sum", "size"])
    good = g["sum"].cumsum() / g["sum"].sum()
    bad = (g["size"] - g["sum"]).cumsum() / (g["size"] - g["sum"]).sum()
    return float((good - bad).abs().max())


def main(mono: bool = False):
    bins, woe, wide_sql = load_data(mono=mono)
    W = woe_matrix(bins, woe)
    cols = list(W.columns)

    # doi chieu: ma tran dung lai tu woe_lookup phai trung voi features_woe cua
    # khoi 2 (chi doi chieu cac bien con giu lai)
    # reindex(W.index) chu khong phai .values: hai truy van co the tra ve khac thu
    # tu dong, va so theo vi tri thi phep doi chieu nay khong the fail dung o
    # truong hop no sinh ra de bat.
    ref = wide_sql.set_index("id")
    if not ref.index.sort_values().equals(W.index.sort_values()):
        raise ValueError("features_woe va row_bins khong cung tap id")
    # Kiem NaN truoc khi lay max: reindex bien id thieu thanh NaN, ma np.max cua
    # mot mang co NaN tra ve NaN, va max() cua Python tren day co NaN thi ket qua
    # phu thuoc thu tu - co the van in ra 0.00e+00. Phep kiem phai fail duoc.
    lech = 0.0
    for v in cols:
        # bang khoi 2 dat cot la woe_debt_ratio con bien la debt_ratio_valid; bang
        # _mono cua khoi 4 da dat dung ten bien nen khong can nhanh nay nua
        name = "woe_" + v if ("woe_" + v) in ref.columns else "woe_debt_ratio"
        col = ref[name].reindex(W.index)
        if col.isna().any():
            raise ValueError(f"{v}: {int(col.isna().sum())} id co trong row_bins ma khong co trong features_woe")
        lech = max(lech, float(np.abs(W[v].values - col.values).max()))
    print(f"doi chieu ma tran WOE voi features_woe: lech lon nhat = {lech:.2e}")

    tr = bins.split == "train"
    te = bins.split == "test"
    y = 1 - bins.target
    lr, coef, se = fit_logit(W[tr], y[tr])

    print(f"\n{'':22s}{'he so':>9s}{'SE':>8s}{'z':>9s}")
    for name, c, s_ in zip(["(intercept)"] + cols, coef, se):
        print(f"{name:22s}{c:9.4f}{s_:8.4f}{c / s_:9.2f}")
    if (coef[1:] < 0).any():
        print(f"CANH BAO: he so am o {[cols[i] for i in np.where(coef[1:] < 0)[0]]}")

    for name, m in [("train", tr), ("test", te)]:
        p_ = lr.predict_proba(W[m])[:, 1]
        print(f"{name:6s} n={int(m.sum()):6d}  Gini={gini(y[m], p_):.4f}  KS={ks(y[m], p_):.4f}")

    factor, offset = scaling_constants()
    print(f"\nFactor = {config.PDO}/ln2 = {factor:.4f}   "
          f"Offset = {config.SCORE_BASE} - Factor*ln({config.ODDS_BASE}) = {offset:.4f}")
    points = build_points(coef, cols, woe, factor, offset)

    # round-trip: cong diem chua lam tron roi giai nguoc phai ra dung log-odds
    lo_model = coef[0] + W.values @ coef[1:]
    lo_score = (apply_points(bins, points, col="diem_raw").values - offset) / factor
    print(f"round-trip log-odds: lech lon nhat = {np.abs(lo_model - lo_score).max():.2e}")

    score = apply_points(bins, points)
    pd_model = 1 / (1 + np.exp(lo_model))
    pd_round = score_to_pd(score.values, factor, offset)
    print(f"lam tron diem ve so nguyen: |PD lech| lon nhat = {np.abs(pd_model - pd_round).max():.5f}, "
          f"trung binh = {np.abs(pd_model - pd_round).mean():.6f}")

    rng = points.groupby("variable").diem.agg(["min", "max"])
    rng["bien_do"] = rng["max"] - rng["min"]
    print(f"\nbang diem: {len(points)} dong, tong diem {rng['min'].sum()}-{rng['max'].sum()}")
    print(rng.sort_values("bien_do", ascending=False).to_string())
    return points, bins.assign(score=score, pd=pd_model)


if __name__ == "__main__":
    import sys
    main(mono="--mono" in sys.argv)
