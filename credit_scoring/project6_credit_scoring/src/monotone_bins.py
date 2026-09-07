"""Gop bin de bang WOE don dieu theo chieu nghiep vu, roi chay lai WOE bang SQL.

Chay:  python src/monotone_bins.py

Phan cong viec giua Python va SQL o buoc nay:

  Python quyet dinh GOP NHUNG BIN NAO. Thuat toan la pool-adjacent-violators,
  vong lap tren bang WOE cua train, khong phai thu viet duoc bang SQL cho ra hon.

  SQL TINH LAI WOE tren cac bin da gop. Van la mot cau GROUP BY va mot CROSS JOIN
  voi dong tong, y het khoi 2. Nghia la khong co con so WOE nao ra doi ngoai SQL.

Cau noi giua hai ben la bang `bin_map` (variable, bin, bin_mono): Python ghi ra,
SQL doc vao. Nho vay cach gop bin thanh mot artefact nhin duoc va kiem duoc, thay
vi nam an trong mot ham.

Vi sao lam buoc nay: khoi 3 do duoc rang ep don dieu dung chieu ton duoi 0,002
Gini (KTC 95%), trong khi bang diem don dieu la thu lam ma ly do noi duoc thanh
cau. Doi mot con so khong do duoc lay mot thu giai thich duoc thi nen doi.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

try:
    import config, cv_check
    from scorecard import _ro_uri
except ImportError:  # khi import tu notebook
    from src import config, cv_check
    from src.scorecard import _ro_uri


def build_map(woe: pd.DataFrame, directions=None) -> pd.DataFrame:
    """Bang anh xa bin goc -> bin da gop, mot dong moi bin goc.

    Nhan cua bin gop la cac bin thanh phan noi bang dau '+', vi du '01+02+03'.
    Chon kieu nhan nay chu khong danh so lai tu 1 de bang diem va ma ly do doc
    len van truy nguoc duoc ve bin goc cua khoi 2. `cv_check.bin_order` van doc
    duoc thu tu tu nhan nay vi no lay cum so dau tien.

    Bin dac biet (X_*, 9_*) khong nam tren truc gia tri nen di thang qua, khong
    tham gia phep gop.
    """
    directions = config.MONOTONE_DIRECTION if directions is None else directions
    rows = []
    for v, g in woe.groupby("variable"):
        if v not in directions:
            raise KeyError(f"chua khai bao chieu don dieu cho bien {v!r}")
        m = g.set_index("bin").woe
        cnt = g.set_index("bin").n.to_dict()
        merged = cv_check.force_monotone(m, cnt, directions[v])
        groups: dict = {}
        for b in sorted((k for k in m.index if not cv_check.is_special(k)),
                        key=cv_check.bin_order):
            # merged.get(...) chu khong merged[...]: force_monotone tra ve {} khi
            # bien co duoi 2 bin thuong, khi do phai giu nguyen WOE goc
            groups.setdefault(round(merged.get(b, m[b]), 9), []).append(b)
        label = {b: "+".join(grp) for grp in groups.values() for b in grp}
        for b in m.index:
            rows.append({"variable": v, "bin": b,
                         "bin_mono": label.get(b, b)})   # bin dac biet giu nguyen
    return pd.DataFrame(rows).sort_values(["variable", "bin"]).reset_index(drop=True)


def write_map(bmap: pd.DataFrame, db_path=None) -> None:
    """Ghi bang bin_map vao credit.db. Day la lenh ghi DUY NHAT cua khoi 4."""
    db_path = config.DB_PATH if db_path is None else db_path
    with sqlite3.connect(db_path) as con:
        bmap.to_sql(config.TABLE_BIN_MAP, con, if_exists="replace", index=False)
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_bin_map "
                    f"ON {config.TABLE_BIN_MAP}(variable, bin)")
        con.commit()


def summary(bmap: pd.DataFrame) -> pd.DataFrame:
    """Bien nao bi gop bao nhieu, de doc nhanh truoc khi chay SQL."""
    out = (bmap.groupby("variable")
           .agg(bin_goc=("bin", "size"), bin_mono=("bin_mono", "nunique")))
    out["gop"] = out.bin_goc - out.bin_mono
    return out.sort_values("gop", ascending=False)


def main() -> None:
    import build_features

    # doc read-only; lenh ghi duy nhat cua file nay nam trong write_map()
    con = sqlite3.connect(_ro_uri(config.DB_PATH), uri=True)
    try:
        woe = pd.read_sql("SELECT variable, bin, n, woe FROM woe_lookup", con)
    finally:
        con.close()
    woe = woe[~woe.variable.isin(config.SCORECARD_DROP)]

    bmap = build_map(woe)
    s = summary(bmap)
    print(s.to_string())
    print(f"\ntong bin: {s.bin_goc.sum()} -> {s.bin_mono.sum()}")

    write_map(bmap)
    print(f"da ghi bang {config.TABLE_BIN_MAP} ({len(bmap)} dong)\n")

    iv = build_features.execute_sql(config.SQL_DIR / "features_mono.sql",
                                    fetch="iv_summary_mono")
    print(iv.to_string(index=False))


if __name__ == "__main__":
    main()
