"""Tinh lai WOE/IV bang pandas de doi chieu voi ket qua cua sql/features.sql.

Day la mot cai dat DOC LAP, khong doc lai bang nao do SQL tao ra ngoai
`applications`. Muc dich la neu ca hai cung sai thi phai sai theo hai duong khac
nhau - doi chieu voi chinh minh thi khong chung minh duoc gi.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


def _ntile_cuts(s: pd.Series, k: int = 10) -> list:
    """Diem cat tuong duong NTILE(k) trong SQLite, da khu trung.

    SQLite chia n dong thanh k nhom: (n mod k) nhom dau co (n div k)+1 dong,
    cac nhom sau co (n div k) dong. Lay can tren tung nhom lam diem cat, bo
    trung, va bo diem cat cuoi de bin tren cung mo ve phia phai.
    """
    v = np.sort(s.dropna().values)
    n = len(v)
    big, n_big = n // k + 1, n % k
    ends = [(i * big if i <= n_big else n_big * big + (i - n_big) * (n // k)) - 1
            for i in range(1, k + 1)]
    return sorted({v[e] for e in ends})[:-1]


def _assign(values: pd.Series, cuts: list) -> np.ndarray:
    out = np.full(len(values), None, dtype=object)
    ok = values.notna().values
    if ok.any():
        idx = np.searchsorted(np.asarray(cuts), values[ok].values, side="left")
        out[ok] = [f"{i + 1:02d}" for i in idx]
    return out


def long_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Gan bin cho moi (dong, bien), dung dung quy tac cua sql/features.sql."""
    tr = df[df.split == "train"]
    frames = []

    cont = {
        "revolving_util": (
            tr.revolving_util.where(tr.flag_util_implausible == 0),
            df.revolving_util.where(df.flag_util_implausible == 0),
            np.where(df.flag_util_implausible == 1, "X_IMPLAUSIBLE", None)),
        "age": (tr.age, df.age, np.full(len(df), None)),
        "debt_ratio_valid": (
            tr.debt_ratio_valid, df.debt_ratio_valid,
            np.where(df.flag_debt_ratio_invalid == 1, "X_INVALID", None)),
        "monthly_income": (
            tr.monthly_income.where(tr.monthly_income > 0),
            df.monthly_income.where(df.monthly_income > 0),
            np.where(df.monthly_income.isna(), "X_MISSING",
                     np.where(df.monthly_income == 0, "X_ZERO", None))),
        "open_credit_lines": (tr.open_credit_lines, df.open_credit_lines,
                              np.full(len(df), None)),
    }
    for var, (train_v, all_v, special) in cont.items():
        b = _assign(all_v, _ntile_cuts(train_v))
        b = np.where(special != None, special, b)          # noqa: E711
        frames.append(pd.DataFrame({"variable": var, "bin": b,
                                    "split": df.split.values, "target": df.target.values}))

    sent = (df.flag_sentinel == 1).values
    disc = {
        "late_30_59": np.where(sent, "9_SENTINEL", np.select(
            [df.late_30_59 == 0, df.late_30_59 == 1, df.late_30_59 == 2, df.late_30_59 <= 4],
            ["0", "1", "2", "3-4"], "5+")),
        "late_60_89": np.where(sent, "9_SENTINEL", np.select(
            [df.late_60_89 == 0, df.late_60_89 == 1, df.late_60_89 == 2],
            ["0", "1", "2"], "3+")),
        "late_90": np.where(sent, "9_SENTINEL", np.select(
            [df.late_90 == 0, df.late_90 == 1, df.late_90 == 2, df.late_90 <= 4],
            ["0", "1", "2", "3-4"], "5+")),
        "real_estate_loans": np.select(
            [df.real_estate_loans == 0, df.real_estate_loans == 1, df.real_estate_loans == 2],
            ["0", "1", "2"], "3+"),
        "dependents": np.where(df.dependents.isna(), "9_MISSING", np.select(
            [df.dependents == 0, df.dependents == 1, df.dependents == 2],
            ["0", "1", "2"], "3+")),
    }
    for var, b in disc.items():
        frames.append(pd.DataFrame({"variable": var, "bin": b,
                                    "split": df.split.values, "target": df.target.values}))
    return pd.concat(frames, ignore_index=True)


def woe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bang WOE/IV tinh tren train."""
    long = long_bins(df)
    tr = long[long.split == "train"]

    # Tong good/bad phai lay tu bang GOC, khong phai tu bang long: bang long co
    # 10 dong cho moi dong goc (mot dong moi bien) nen tong o do gap 10 lan.
    # Sai so nay TRIET TIEU trong WOE (vi WOE la ti so pct_good/pct_bad) nhung
    # KHONG triet tieu trong IV (vi IV dung HIEU pct_good - pct_bad). Doi chieu
    # WOE khop khong chung minh duoc IV dung.
    wide_train = df[df.split == "train"]
    n_good = int((wide_train.target == 0).sum())
    n_bad = int(wide_train.target.sum())

    t = (tr.groupby(["variable", "bin"]).target
           .agg(n="size", n_bad="sum").reset_index())
    t["n_good"] = t.n - t.n_bad
    t["woe"] = np.log((t.n_good / n_good) / (t.n_bad / n_bad))
    t["iv_part"] = (t.n_good / n_good - t.n_bad / n_bad) * t.woe
    return t


def compare(db_path=None) -> dict:
    """So bang WOE cua pandas voi bang woe_lookup do SQL tao ra."""
    db_path = config.DB_PATH if db_path is None else db_path
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("SELECT * FROM applications", con)
        sql = pd.read_sql("SELECT variable, bin, n, n_good, n_bad, woe FROM woe_lookup", con)
        iv_sql = pd.read_sql("SELECT variable, iv FROM iv_summary", con).set_index("variable").iv

    pdt = woe_table(df)
    m = sql.merge(pdt, on=["variable", "bin"], how="outer",
                  suffixes=("_sql", "_pd"), indicator=True)
    iv_pd = pdt.groupby("variable").iv_part.sum()
    # iv_summary lam tron 4 chu so nen so o cung do chinh xac do
    iv_diff = (iv_sql.sort_index() - iv_pd.sort_index().round(4)).abs()

    return {
        "so_bin_sql": len(sql),
        "so_bin_pandas": len(pdt),
        "khop_het_bin": bool((m._merge == "both").all()),
        "lech_so_dong": int((m.n_sql - m.n_pd).abs().max()),
        "lech_so_bad": int((m.n_bad_sql - m.n_bad_pd).abs().max()),
        "lech_woe_lon_nhat": float((m.woe_sql - m.woe_pd).abs().max()),
        "lech_iv_lon_nhat": float(iv_diff.max()),
        "so_bien_lech_iv": int((iv_diff > 1e-4).sum()),
    }
