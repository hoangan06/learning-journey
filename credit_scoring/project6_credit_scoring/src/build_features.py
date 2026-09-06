"""Chay sql/features.sql len data/credit.db.

Dung Python thay vi goi CLI `sqlite3` vi may Windows thuong khong co san CLI do.
Script khong tinh gi ca, moi phep tinh nam trong file SQL - o day chi doc file,
thuc thi, roi in ra bang IV de kiem nhanh.

Chay:  python src/build_features.py
"""
from __future__ import annotations

import math
import sqlite3

import pandas as pd

try:
    from . import config
except ImportError:
    import config


def _ensure_ln(con: sqlite3.Connection) -> str:
    """Bao dam ham LN() dung duoc trong SQL.

    LN() thuoc math extension cua SQLite (co tu 3.35) nhung chi ton tai neu ban
    SQLite duoc bien dich voi SQLITE_ENABLE_MATH_FUNCTIONS. Nhieu ban Python tren
    Windows (ke ca trong conda) khong bat co nay, va khi do features.sql chet o
    buoc tinh WOE voi loi "no such function: LN".

    Neu thieu thi dang ky mot ham LN viet bang Python. Ket qua giong het ban
    native: ca hai deu goi log tu nhien cua C, chi khac duong di.

    Tra ve 'native' hoac 'python' de biet da di duong nao.
    """
    try:
        con.execute("SELECT LN(2.0)").fetchone()
        return "native"
    except sqlite3.OperationalError:
        con.create_function("LN", 1, lambda x: math.log(x) if x is not None and x > 0 else None)
        return "python"


def run(db_path=None, sql_path=None) -> pd.DataFrame:
    db_path = config.DB_PATH if db_path is None else db_path
    sql_path = (config.SQL_DIR / "features.sql") if sql_path is None else sql_path
    script = sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        print(f"LN(): {_ensure_ln(con)}")
        con.executescript(script)
        con.commit()
        return pd.read_sql("SELECT * FROM iv_summary", con)


def check(db_path=None) -> dict:
    """Ba phep kiem sau khi chay: dung so dong, khong co WOE NULL, bin khop train."""
    db_path = config.DB_PATH if db_path is None else db_path
    with sqlite3.connect(db_path) as con:
        n_rows = con.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
        n_vars = con.execute("SELECT COUNT(DISTINCT variable) FROM row_bins").fetchone()[0]
        out = {
            "n_features_woe": con.execute("SELECT COUNT(*) FROM features_woe").fetchone()[0],
            "n_row_bins": con.execute("SELECT COUNT(*) FROM row_bins").fetchone()[0],
            "n_row_bins_ky_vong": n_rows * n_vars,
            # WOE NULL nghia la co bin xuat hien o test/oot ma train chua tung thay.
            # LEFT JOIN o muc 6 cua features.sql cot y de loi nay lo ra thay vi
            # lam bien mat dong do trong im lang.
            "n_dong_woe_null": con.execute("""
                SELECT COUNT(*) FROM features_woe WHERE
                  woe_revolving_util IS NULL OR woe_age IS NULL OR woe_debt_ratio IS NULL
                  OR woe_monthly_income IS NULL OR woe_open_credit_lines IS NULL
                  OR woe_late_30_59 IS NULL OR woe_late_60_89 IS NULL OR woe_late_90 IS NULL
                  OR woe_real_estate_loans IS NULL OR woe_dependents IS NULL""").fetchone()[0],
        }
    out["ok"] = (out["n_features_woe"] == n_rows
                 and out["n_row_bins"] == out["n_row_bins_ky_vong"]
                 and out["n_dong_woe_null"] == 0)
    return out


def main() -> None:
    iv = run()
    print(iv.to_string(index=False))
    print()
    for k, v in check().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
