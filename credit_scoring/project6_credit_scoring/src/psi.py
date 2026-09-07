"""PSI va CSI: quan the co con giong luc build khong, khi chua co nhan.

Chay:  python src/psi.py

PSI tra loi mot cau khac han moi thu do o khoi 3 va 4. Gini, KS, Brier deu can
nhan; PSI thi khong. Trong doi that nhan den sau 12 den 24 thang, nen giai doan
do PSI la thu duy nhat noi duoc model con dung duoc hay khong.

Cai gia phai tra: PSI nhin P(X) va KHONG BAO GIO nhin P(y|X). No khong biet
model con dung khong, no chi biet dau vao co con giong luc build khong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- ranh gioi bin dong bang ---------------------------------------------------

def freeze_cuts(x_ref, k: int = 10) -> np.ndarray:
    """Diem cat phan vi tren tap THAM CHIEU, tinh mot lan roi dung mai mai.

    Day la ham quan trong nhat file va cung la cho de sai nhat. Neu moi ky giam
    sat lai chia decile moi tren du lieu moi thi ti trong quan sat luon bang 0,10
    o moi bin va PSI luon bang 0 du quan the co dich chuyen toi dau: ta dang do
    phan phoi cua du lieu moi so voi chinh no. Cai bay do im lang tuyet doi.

    Ham `psi()` vi vay bat buoc nhan `cuts` tu ngoai vao chu khong tu tinh, de
    khong ai vo tinh goi no theo kieu tinh lai moi ky. Muc 4.3 cua notebook 05 do
    hau qua cua viec goi sai bang mot con so.
    """
    x = np.asarray(x_ref, dtype=float)
    q = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.unique(q)


def bin_by_cuts(x, cuts) -> np.ndarray:
    """Gan bin theo diem cat da dong bang. Bin i = so diem cat bi vuot."""
    return np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(x, dtype=float), side="left")


def shares(idx, n_bins: int) -> np.ndarray:
    """Ti trong tung bin. Giu ca bin rong de PSI thay chung."""
    c = np.bincount(np.asarray(idx, dtype=int), minlength=n_bins).astype(float)
    return c / c.sum()


# --- PSI ------------------------------------------------------------------------

def psi_from_shares(e, a, eps: float = 1e-4) -> float:
    """PSI = SUM (a_i - e_i) * ln(a_i / e_i).

    Tung so hang khong am nen PSI >= 0, bang 0 khi hai phan phoi trung khit.
    Cung cong thuc voi IV, tuc Jeffreys divergence.

    eps thay cho ti trong bang 0 de ln khong ra vo cuc. Day la mot lua chon co
    hau qua: bin rong o mau moi cho so hang (0 - e)*ln(eps/e), va gia tri do phu
    thuoc eps chu khong phai du lieu. Voi eps = 1e-4 va e = 0,1 thi mot bin rong
    dong gop khoang 0,69 mot minh, tuc du de mot minh no day PSI qua nguong. Bao
    kem so bin rong moi khi PSI lon la bat buoc, khong phai tuy chon.
    """
    e = np.asarray(e, dtype=float).copy()
    a = np.asarray(a, dtype=float).copy()
    e[e <= 0] = eps
    a[a <= 0] = eps
    return float(np.sum((a - e) * np.log(a / e)))


def psi(x_ref, x_new, cuts, eps: float = 1e-4) -> dict:
    """PSI cua mot bien lien tuc, dung diem cat da dong bang."""
    nb = len(cuts) + 1
    e = shares(bin_by_cuts(x_ref, cuts), nb)
    a = shares(bin_by_cuts(x_new, cuts), nb)
    return {"psi": psi_from_shares(e, a, eps), "e": e, "a": a,
            "bin_rong_moi": int((a == 0).sum()), "n_bin": nb}


def psi_table(x_ref, x_new, cuts, eps: float = 1e-4) -> pd.DataFrame:
    """Bang tung bin, de nhin duoc PSI den tu dau chu khong chi doc mot so."""
    r = psi(x_ref, x_new, cuts, eps)
    e, a = r["e"].copy(), r["a"].copy()
    e[e <= 0], a[a <= 0] = eps, eps
    return pd.DataFrame({"bin": range(r["n_bin"]), "ti_trong_ref": r["e"], "ti_trong_moi": r["a"],
                         "dong_gop": (a - e) * np.log(a / e)})


# --- CSI --------------------------------------------------------------------------

def csi(bins_ref: pd.Series, bins_new: pd.Series, eps: float = 1e-4) -> float:
    """CSI cua mot bien da co bin san (row_bins), tuc PSI tren ti trong bin cua no.

    Trong scorecard co mot thuan tien dang ke: feature da la WOE, tuc ham bac thang
    tren cac bin, nen CSI cua mot bien CHINH LA PSI tren cac bin cua bien do. Khong
    can cong cu rieng, cung mot phep dem ti trong.
    """
    lv = sorted(set(bins_ref.unique()) | set(bins_new.unique()))
    e = bins_ref.value_counts(normalize=True).reindex(lv, fill_value=0.0).values
    a = bins_new.value_counts(normalize=True).reindex(lv, fill_value=0.0).values
    return psi_from_shares(e, a, eps)


def csi_table(long_ref: pd.DataFrame, long_new: pd.DataFrame, eps: float = 1e-4) -> pd.DataFrame:
    """CSI cho moi bien trong bang row_bins dang long (cot: variable, bin)."""
    out = []
    for v in sorted(long_ref.variable.unique()):
        out.append({"variable": v,
                    "csi": csi(long_ref.loc[long_ref.variable == v, "bin"],
                               long_new.loc[long_new.variable == v, "bin"], eps)})
    return pd.DataFrame(out).sort_values("csi", ascending=False).reset_index(drop=True)


# --- OOT dich nhan tao -------------------------------------------------------------

def shift_sample(df: pd.DataFrame, col: str, alpha: float, n: int, seed: int = 42) -> pd.DataFrame:
    """Lay mau lai thien ve gia tri CAO cua `col`, mo phong mot chien dich mang ve
    khach rui ro hon.

    Trong so w_i = exp(alpha * u_i) voi u_i la phan vi cua col trong chinh df, nen
    alpha = 0 cho lay mau deu va alpha cang lon cang thien lech. Lay KHONG hoan lai
    de khong tao dong trung, doi lai n phai nho hon len(df).

    DIEU QUAN TRONG NHAT cua ham nay: trong so chi phu thuoc X, tuyet doi khong
    nhin y. Nho vay phep lay mau doi P(X) ma giu nguyen P(y|X), va do la dieu kien
    de kiem duoc mot menh de cu the: PSI se keu to trong khi calibration van dung.
    Neu trong so co dinh dang toi y thi ca phep kiem do mat nghia.

    Mot cho phai noi thang: lay mau KHONG hoan lai voi `p=w` thi xac suat mot dong
    duoc chon KHONG ti le dung voi w, vi numpy rut lan luot va rut roi thi bo ra.
    Do lech cang ro khi alpha lon. Dieu do khong lam hong phep kiem, vi phep kiem
    chi can dung mot tinh chat la "trong so khong nhin y"; nhung no la ly do alpha
    phai len toi 6 moi day duoc PSI qua 0,4, chu khong phai vi quan the kho dich.
    """
    if n >= len(df):
        raise ValueError(f"n={n} phai nho hon len(df)={len(df)} vi lay mau khong hoan lai")
    u = df[col].rank(pct=True).values
    w = np.exp(alpha * u)
    w = w / w.sum()
    idx = np.random.default_rng(seed).choice(len(df), size=n, replace=False, p=w)
    return df.iloc[np.sort(idx)].copy()


def main():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import config, scorecard
    bins, woe, _ = scorecard.load_data(mono=True)
    W = scorecard.woe_matrix(bins, woe)
    tr, oo = (bins.split == "train").values, (bins.split == "oot").values
    lr, coef, se = scorecard.fit_logit(W[tr], (1 - bins.target)[tr])
    s_all = 1 - lr.predict_proba(W)[:, 1]
    cuts = freeze_cuts(s_all[tr], 10)
    r = psi(s_all[tr], s_all[oo], cuts)
    print(f"PSI diem so, train -> oot ngau nhien : {r['psi']:.6f}   ({r['n_bin']} bin, "
          f"{r['bin_rong_moi']} bin rong)")
    print(f"xap xi ly thuyet (k-1)(1/n1+1/n2)     : {9*(1/tr.sum()+1/oo.sum()):.6f}")


if __name__ == "__main__":
    main()
