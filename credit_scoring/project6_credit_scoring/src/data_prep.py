"""Nap cs-training.csv, danh co cac gia tri dac biet, split va day vao SQLite.

Ba quy uoc cua buoc nay:

Khong impute va khong xoa dong chi vi gia tri "trong xau". O bo du lieu nay
missing co mang thong tin - nhom thieu monthly_income co bad rate 5,61% trong
khi nhom co khai la 6,95%. Impute median se lam nhoe mat chenh lech do. Cach
xu ly la danh co, roi den buoc binning cho missing thanh mot bin rieng va de
WOE cua no do chinh du lieu quyet dinh.

Split truoc khi tinh bat cu thu gi. Ranh gioi bin va gia tri WOE la tham so
hoc duoc chu khong phai buoc tien xu ly, nen tinh chung tren toan bo du lieu
la de nhan cua test tham gia vao viec dung feature cho chinh test.

Moi nguong o day deu co so lieu kem theo trong results/data_profile.md.

Chay:  python src/data_prep.py
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:          # khi chay nhu script doc lap
    import config


def load_raw() -> pd.DataFrame:
    """Doc CSV tho va doi ten cot. Cot index khong ten thanh `id`."""
    df = pd.read_csv(config.RAW_CSV)
    return df.rename(columns={"Unnamed: 0": "id", **config.COLUMN_MAP})


def check_index_leakage(df: pd.DataFrame) -> dict:
    """Xem `id` co du bao duoc target khong truoc khi bo cot nay di.

    Khong phai lo vo co: file du lieu thuong duoc sap xep theo thoi gian hoac
    theo nhan truoc khi xuat, va khi do so thu tu dong tro thanh mot feature
    du bao rat tot ma khong the dung khi trien khai.

    Voi n = 150.000, sai so chuan cua he so tuong quan vao khoang 0,0026 nen
    nguong 0,01 o day tuong duong khoang bon lan sai so chuan.
    """
    corr = float(np.corrcoef(df["id"], df[config.TARGET])[0, 1])
    blocks = df.groupby(pd.qcut(df["id"], 10, labels=False))[config.TARGET].mean()
    return {
        "corr_id_target": round(corr, 6),
        "bad_rate_per_index_block": [round(100 * v, 2) for v in blocks.tolist()],
        "verdict": "khong co dau hieu leakage theo thu tu dong" if abs(corr) < 0.01
                   else "CANH BAO: dieu tra truoc khi bo cot id",
    }


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Them cot co cho cac gia tri dac biet. Gia tri goc giu nguyen."""
    df = df.copy()

    # Sentinel 96/98: co o muc ban ghi, khong phai muc cot.
    sent = df[config.DELINQUENCY_COLS].isin(config.SENTINEL_VALUES).any(axis=1)
    df["flag_sentinel"] = sent.astype(int)

    # monthly_income va dependents: missing co mang thong tin nen tach ra thanh
    # co rieng thay vi impute. income = 0 tach khoi income missing vi hai nhom
    # nay co bad rate khac nhau (4,04% so voi 5,61%).
    df["flag_income_missing"] = df["monthly_income"].isna().astype(int)
    df["flag_income_zero"] = (df["monthly_income"] == 0).astype(int)
    df["flag_dependents_missing"] = df["dependents"].isna().astype(int)

    df["flag_util_implausible"] = (df["revolving_util"] > config.UTIL_IMPLAUSIBLE).astype(int)

    # debt_ratio doi don vi tuy theo mau so. Khi co thu nhap, no la ti le
    # (median 0,29). Khi thu nhap missing hoac bang 0, mau so hong va cot tro
    # thanh so tien tuyet doi (median 1.159 va 930). Nhan nguoc lai tren nhom
    # co thu nhap cho median 1.649 USD/thang, dung co mot khoan tra no hang
    # thang, nen gia thuyet nay dung vung.
    #
    # Neu binning thang tren cot goc thi cac bin cao se toan la nhom thieu thu
    # nhap, tuc bien do "khach co khai thu nhap khong" chu khong con do muc no.
    invalid = df["monthly_income"].isna() | (df["monthly_income"] == 0)
    df["flag_debt_ratio_invalid"] = invalid.astype(int)
    df["debt_ratio_valid"] = df["debt_ratio"].where(~invalid)

    return df


def drop_impossible(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bo cac ban ghi khong the ton tai. Hien tai chi co dung mot: age = 0.

    Bao cao them so dong trung feature de tien theo doi, nhung khong bo chung.
    Kiem tra cho thay day la ho so thin file (98,1% co debt_ratio = 0, 85%
    thieu thu nhap) nen trung khop la chuyen binh thuong, va 37 nhom trong so
    do co nhan mau thuan nen chung la nhung nguoi khac nhau chu khong phai ban
    sao cua cung mot ho so.
    """
    bad_age = df["age"] <= 0
    report = {
        "dropped_age_le_0": int(bad_age.sum()),
        "duplicate_feature_rows_kept": int(
            df.drop(columns=["id", config.TARGET]).duplicated(keep=False).sum()
        ),
    }
    return df.loc[~bad_age].reset_index(drop=True), report


def stratified_split(y: np.ndarray,
                     ratios: tuple = config.SPLIT_RATIOS,
                     seed: int = config.SEED) -> np.ndarray:
    """Chia train / test / oot phan tang theo nhan, tai lap tu seed.

    Phan tang de bad rate ba tap gan bang nhau. Neu khong, mot phan chenh lech
    Gini giua cac tap se den tu ti le duong khac nhau chu khong phai tu model.

    Doi lai, phan tang lam tap oot nay KHAC mot tap out-of-time that: mot tap
    out-of-time that lay tu giai doan sau, va viec bad rate giai doan sau lech
    di chinh la thu can phat hien. Ep bad rate bang nhau la chu dong bo di bien
    dong do. He qua: PSI, bad rate va calibration tren oot deu se dep theo
    thiet ke chu khong phai theo nang luc model.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out = np.empty(len(y), dtype=object)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(ratios[0] * n))
        n_test = int(round(ratios[1] * n))
        out[idx[:n_train]] = "train"
        out[idx[n_train:n_train + n_test]] = "test"
        out[idx[n_train + n_test:]] = "oot"
    return out


def write_sqlite(df: pd.DataFrame, db_path=None, table=None) -> None:
    """Ghi mot bang duy nhat co cot `split`.

    Mot bang kem cot split thay vi ba bang rieng la de buoc WOE tinh dung cho:
    tinh thi loc `WHERE split = 'train'`, ap thi `LEFT JOIN` bang tra vao toan
    bo bang. Khong co tap nao ton tai rieng de co the vo tinh fit len.

    Doc config luc goi chu khong phai luc import, de test va notebook doi duoc
    duong dan ma khong phai sua ham.
    """
    db_path = config.DB_PATH if db_path is None else db_path
    table = config.TABLE if table is None else table
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_split ON {table}(split)")
        con.commit()


def build(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Chay ca chuoi: nap, danh co, bo dong loi, split. Tra ve df va bao cao."""
    raw = load_raw()
    report = {"n_rows_raw": len(raw), "index_check": check_index_leakage(raw)}

    df = add_flags(raw)
    df, drop_report = drop_impossible(df)
    report.update(drop_report)

    df["split"] = stratified_split(df[config.TARGET].values)

    lead = ["id", "split", config.TARGET]
    df = df[lead + [c for c in df.columns if c not in lead]]

    report["n_rows_final"] = len(df)
    report["bad_rate_overall"] = round(100 * df[config.TARGET].mean(), 4)
    report["split_summary"] = (
        df.groupby("split")[config.TARGET]
        .agg(n="size", bad="sum")
        .assign(bad_rate_pct=lambda d: (100 * d.bad / d.n).round(4))
        .to_dict("index")
    )
    if verbose:
        for k, v in report.items():
            print(f"{k}: {v}")
    return df, report


def main() -> None:
    """Chay ca chuoi roi ghi ra SQLite."""
    df, _ = build()
    write_sqlite(df)
    print(f"\nDa ghi {len(df):,} dong vao {config.DB_PATH} (bang '{config.TABLE}')")


if __name__ == "__main__":
    main()
