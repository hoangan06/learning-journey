"""Hang so dung chung cho ca project.

De tap trung o mot cho vi 01_eda, 02_woe, 03_scorecard... deu can seed va
duong dan giong nhau. Neu moi notebook tu dinh nghia lai thi som muon cung
lech, va bang WOE tinh ra tu mot split khac voi split luc train la loi im lang.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SQL_DIR = PROJECT_ROOT / "sql"

RAW_CSV = DATA_DIR / "cs-training.csv"
DB_PATH = DATA_DIR / "credit.db"
TABLE = "applications"

SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / test / oot-proxy

TARGET = "target"                   # bad = 1, la lop thieu so (6,68%)

# Ten cot goc co dau gach ngang (NumberOfTime30-59DaysPastDueNotWorse), viet
# trong SQL phai quote nen doi het sang snake_case ngay tu buoc nap.
COLUMN_MAP = {
    "SeriousDlqin2yrs":                     "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_util",
    "age":                                  "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio":                            "debt_ratio",
    "MonthlyIncome":                        "monthly_income",
    "NumberOfOpenCreditLinesAndLoans":      "open_credit_lines",
    "NumberOfTimes90DaysLate":              "late_90",
    "NumberRealEstateLoansOrLines":         "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents":                   "dependents",
}

DELINQUENCY_COLS = ["late_30_59", "late_60_89", "late_90"]

# 96 va 98 khong phai so lan tre han. Chung xuat hien dong thoi o ca ba cot dem
# tren dung 269 dong, tuc la ma trang thai cua ca ban ghi. Nhom nay co bad rate
# 54,65% so voi 6,60% phan con lai nen giu lai va cho thanh mot bin rieng.
SENTINEL_VALUES = (96, 98)

# Nguong chon theo nghia cua bien: revolving_util la du no chia han muc, gia tri
# 10 nghia la dung 1000% han muc. Khong do nguong theo target - lam vay la de
# nhan dan dat mot quyet dinh tien xu ly. Du lieu xac nhan sau: 223 dong co
# util > 100 chi co bad rate 4,93%, thap hon base rate 6,684%.
UTIL_IMPLAUSIBLE = 10.0

IV_THRESHOLDS = {"useless": 0.02, "weak": 0.10, "medium": 0.30, "strong": 0.50}
PSI_THRESHOLDS = {"stable": 0.10, "watch": 0.25}

# --- Scorecard -------------------------------------------------------------
# Ba hang so quy dinh thang diem. Chung KHONG anh huong gi den do chinh xac cua
# model: diem la ham tuyen tinh cua log-odds nen moi bo (PDO, SCORE_BASE,
# ODDS_BASE) deu cho cung mot thu tu xep hang va cung mot Gini. Chon theo thong
# le nganh (Siddiqi 2017) de con so quen mat voi nguoi doc.
PDO = 20          # so diem lam odds gap doi
SCORE_BASE = 600  # diem quy chieu
ODDS_BASE = 50    # odds good:bad tai SCORE_BASE

# Bo khoi model cuoi. Ly do: dong gop bien te do bang 5-fold CV trong train la
# +0,00004 Gini, tuc bo di thi Gini con nhich len. He so am cua no trong model da
# bien KHONG phai ly do thu hai doc lap (z = -0,68, khong phan biet duoc voi 0),
# no la cung mot su viec nhin tu goc khac. Xem results/scorecard.md muc 3.
SCORECARD_DROP = ["open_credit_lines"]

# Tien to cua nhan bin dac biet: ma trang thai chu khong phai khoang gia tri.
# Chung khong nam tren truc gia tri nen phai dung ngoai moi phep ep don dieu.
SPECIAL_BIN_PREFIXES = ("X_", "9_")
