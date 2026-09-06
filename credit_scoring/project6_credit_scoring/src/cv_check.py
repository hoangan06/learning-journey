"""Kiem chon bien va chon cach chia bin bang 5-fold CV BEN TRONG train.

Ly do phai co file rieng thay vi goi thang sklearn: WOE la tham so hoc duoc, nen
moi fold phai TINH LAI WOE tren 4 fold con lai roi ap sang fold thu 5. Neu dung
san cot WOE trong features_woe (von tinh tren ca train) thi fold danh gia da gop
phan tao ra chinh feature cua no, va chenh lech Gini do duoc se lac quan.

Tap test khong duoc dung o day. Test chi de kiem cac du doan da ghi truoc, con
moi quyet dinh chon bien deu ra doi trong train.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    import config
except ImportError:
    from src import config


def woe_of(df, var, target="target"):
    """WOE tung bin, mau so la TONG good/bad cua df.

    Cong 0,5 vao tu so kieu Laplace: trong mot fold co the co bin khong con dong
    bad nao va ln(0) se lam hong ca fold. Voi bin lon thi 0,5 khong doi duoc gi,
    nhung voi bin nho thi CO: X_IMPLAUSIBLE (177 dong) cho WOE -0,2130 o day so
    voi -0,1816 trong woe_lookup khong smoothing, lech 0,031. Cac bin do gan nhu
    khong co trong luong khi fit nen he so cuoi chi lech khoang 6e-04, nhung
    khong duoc noi la smoothing vo hai.
    """
    g = df.groupby(var)[target].agg(bad="sum", n="size")
    g["good"] = g["n"] - g["bad"]
    k = len(g)
    pg = (g["good"] + 0.5) / (g["good"].sum() + 0.5 * k)
    pb = (g["bad"] + 0.5) / (g["bad"].sum() + 0.5 * k)
    return np.log(pg / pb)


def is_special(b):
    """Bin dac biet (khong nam tren truc gia tri) hay bin thuong.

    Mot bin dac biet la mot ma trang thai chu khong phai mot khoang gia tri:
    X_MISSING, X_ZERO, X_INVALID, X_IMPLAUSIBLE, 9_SENTINEL, 9_MISSING. Chung
    khong co thu tu so voi cac bin con lai nen phai dung ngoai moi phep ep don
    dieu.
    """
    return str(b).startswith(config.SPECIAL_BIN_PREFIXES)


def bin_order(b):
    """Khoa sap xep cua mot bin thuong: lay cum so dau tien.

    KHONG dung sorted() trên chuoi. Nhan bin trong du an co ba dang: so co
    zero-pad ('01'..'10'), so tran ('0','1','2') va khoang mo ('3+', '3-4',
    '5+'). sorted() chuoi xep '10' truoc '2', va str.isdigit() thi loai han
    '3+' ra khoi day - dung nham cai nao cung lam PAVA chay tren mot tap thieu
    ma khong bao gi.
    """
    m = re.match(r"(\d+)", str(b))
    if m is None:
        raise ValueError(f"bin khong doc duoc thu tu: {b!r}")
    return int(m.group(1))


def _pava(order, values, weights):
    """Ep don dieu tang theo thu tu bin bang pool-adjacent-violators.

    Gap hai bin lien ke nao vi pham thu tu thanh mot khoi, gia tri khoi la trung
    binh co trong so theo so dong, lap den khi khong con vi pham. Day la nghiem
    binh phuong toi thieu cua bai toan hoi quy don dieu (Barlow et al. 1972).
    """
    blocks = [[values[k], weights[k], [k]] for k in order]
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1):
            if blocks[i][0] > blocks[i + 1][0]:
                (v1, w1, k1), (v2, w2, k2) = blocks[i], blocks[i + 1]
                blocks[i:i + 2] = [[(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, k1 + k2]]
                changed = True
                break
    return {k: v for v, _, ks in blocks for k in ks}


def force_monotone(woe, counts, direction):
    """Ep don dieu mot bien, bo qua bin dac biet. direction: "tang" hoac "giam".

    Chieu la THAM SO BAT BUOC, khong co mac dinh va khong co che do tu doan. Voi
    bien don dieu san thi chieu hien nhien, nhung voi bien hinh chu U thi khong:
    ep tang gop phang nhanh phai, ep giam gop phang nhanh trai, va o bo nay hai
    chieu chenh nhau toi 0,05 Gini. Ban truoc cua ham nay tu doan chieu bang dau
    hiep phuong sai co trong so, doan sai o dung bien dang can phan xu, va cho ra
    mot ket luan nguoc hoan toan. Ai goi ham nay thi phai tu noi minh dang ep
    chieu nao.
    """
    keys = [k for k in woe.index if not is_special(k)]
    keys.sort(key=bin_order)
    if len(set(map(bin_order, keys))) != len(keys):
        raise ValueError(f"co hai bin cung thu tu: {keys}")
    if len(keys) < 2:
        return {}          # 0 hoac 1 bin thuong thi khong co gi de ep
    if direction == "tang":
        return _pava(keys, woe.to_dict(), counts)
    if direction == "giam":
        flipped = _pava(keys, {k: -woe[k] for k in keys}, counts)
        return {k: -x for k, x in flipped.items()}
    raise ValueError(f"direction khong hop le: {direction!r}")


def _design(fit, apply_, vars_, merge=None, monotone=None):
    merge = merge or {}
    monotone = monotone or {}
    if not isinstance(monotone, dict):
        raise TypeError("monotone phai la dict {ten bien: 'tang'|'giam'}, de chieu luon hien ra")
    A, B = fit.copy(), apply_.copy()
    for v, rule in merge.items():
        A[v] = A[v].replace(rule)
        B[v] = B[v].replace(rule)
    Xa = np.empty((len(A), len(vars_)))
    Xb = np.empty((len(B), len(vars_)))
    for j, v in enumerate(vars_):
        m = woe_of(A, v)
        if v in monotone:
            cnt = A.groupby(v)["target"].size().to_dict()
            for k, val in force_monotone(m, cnt, monotone[v]).items():
                m[k] = val
        Xa[:, j] = A[v].map(m).values
        Xb[:, j] = B[v].map(m).values
        if np.isnan(Xb[:, j]).any():
            # bin co o fold danh gia ma vang o 4 fold fit. Bao loi thay vi dien 0:
            # dien 0 nghia la "bin trung tinh" va no se lam ket qua dep len mot cach
            # gia tao dung o cac bin hiem, tuc dung cho can nhin ky nhat.
            raise ValueError(f"{v}: co bin trong fold danh gia ma fold fit chua thay")
    return Xa, Xb


def gini_cv(bins, vars_, merge=None, monotone=None, k=5, seed=None):
    """Gini tren tung fold, WOE tinh lai trong fold. Tra ve mang k phan tu.

    monotone: dict {ten bien: "tang"|"giam"}. Khong co mac dinh cho chieu.
    """
    seed = config.SEED if seed is None else seed
    tr = bins[bins.split == "train"].reset_index(drop=True)
    idx = np.arange(len(tr))
    np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    out = []
    for f in range(k):
        va = tr.iloc[folds[f]]
        fit = tr.iloc[np.concatenate([folds[i] for i in range(k) if i != f])]
        Xa, Xb = _design(fit, va, vars_, merge, monotone)
        lr = LogisticRegression(C=1e12, max_iter=2000).fit(Xa, 1 - fit.target.values)
        p = lr.predict_proba(Xb)[:, 1]
        out.append(2 * roc_auc_score(1 - va.target.values, p) - 1)
    return np.array(out)


def paired(base, other):
    """So sanh theo cap tren cung fold. Ghep cap moi tra loi dung cau hoi.

    Phuong sai giua cac fold (0,7005 den 0,7215 o bo nay) lon hon nhieu lan
    chenh lech giua hai phuong an, nen so hai trung binh doc lap se khong thay
    gi. Tru theo tung fold thi phan phuong sai chung do fold triet tieu.
    """
    d = np.asarray(other) - np.asarray(base)
    se = d.std(ddof=1) / np.sqrt(len(d))
    # t Student 97,5% voi k-1 bac tu do. Bao ca khoang chu khong chi bao d va t:
    # phan lon ket luan o day la ket luan NULL ("khong do duoc chenh lech nao"),
    # ma mot ket luan null chi co nghia khi noi kem no loai tru duoc den dau.
    tcrit = {2: 12.706, 3: 4.303, 4: 2.776, 5: 2.571, 9: 2.262}.get(len(d) - 1)
    if tcrit is None:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.975, len(d) - 1))
    return {
        "d": float(d.mean()),
        "se": float(se),
        "t": float(d.mean() / se) if se > 0 else float("nan"),
        "ktc_lo": float(d.mean() - tcrit * se),
        "ktc_hi": float(d.mean() + tcrit * se),
        "cung_dau": bool(np.all(d > 0) or np.all(d < 0)),
    }
