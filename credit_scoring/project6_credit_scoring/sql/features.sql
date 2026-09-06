-- =============================================================================
-- Feature engineering WOE bang SQL  --  Give Me Some Credit
--
-- Chay:  python src/build_features.py
--
-- Khong chay bang CLI `sqlite3` truc tiep tru khi ban SQLite do co math extension:
-- muc 4 dung LN(), ma ham nay chi ton tai khi SQLite duoc bien dich voi
-- SQLITE_ENABLE_MATH_FUNCTIONS. Nhieu ban Python tren Windows khong bat co do,
-- nen build_features.py tu dang ky mot ham LN thay the khi can.
--
-- Nguyen tac chi phoi toan bo file: MOI THAM SO HOC DUOC CHI DUOC TINH TREN
-- split = 'train'. Diem cat bin va gia tri WOE deu la tham so hoc duoc, khong
-- phai buoc tien xu ly, nen tinh chung tren ca bang la de nhan cua test dung
-- feature cho chinh test.
--
-- Bang tao ra:
--   row_values    long format: moi dong x moi bien, kem ma dac biet
--   bin_cuts      diem cat lay tu NTILE(10) TREN TRAIN, chong trung gia tri
--   row_bins      gan bin cho TOAN BO cac dong bang cach ap bin_cuts
--   woe_lookup    bang tra WOE/IV, tinh tren TRAIN
--   iv_summary    IV mot dong moi bien
--   features_woe  bang rong: mot cot WOE moi bien, cho TOAN BO cac dong
--
-- SQL su dung: CTE, window function (NTILE, ROW_NUMBER, COUNT OVER), GROUP BY,
--              CASE, LEFT JOIN, CROSS JOIN, range join.
-- =============================================================================

DROP TABLE IF EXISTS row_values;
DROP TABLE IF EXISTS bin_cuts;
DROP TABLE IF EXISTS row_bins;
DROP TABLE IF EXISTS woe_lookup;
DROP TABLE IF EXISTS iv_summary;
DROP TABLE IF EXISTS features_woe;


-- -----------------------------------------------------------------------------
-- 1. LONG FORMAT
--
-- Moi dong goc thanh 10 dong, mot dong moi bien. Cot `val` chi dung cho bien
-- lien tuc; `special` la ma bin cho bien roi rac va cho cac gia tri dac biet
-- (missing, zero, sentinel, ngoai nguong). Dong nao co `special` thi bo qua
-- buoc cat bin o muc 3.
--
-- Bien roi rac dung CASE chu khong dung NTILE. Ly do o results/data_profile.md
-- muc 10: NTILE khong gop diem cat trung nhau ma chia deu so dong bat ke gia
-- tri bang nhau, nen tren cot 94% gia tri 0 no tao ra chin bin deu chi chua
-- gia tri 0 va tra ve mot IV trong rat hop ly.
-- -----------------------------------------------------------------------------
CREATE TABLE row_values AS
    -- ---- lien tuc -----------------------------------------------------------
    SELECT id, split, target, 'revolving_util' AS variable,
           CASE WHEN flag_util_implausible = 1 THEN NULL ELSE revolving_util END AS val,
           CASE WHEN flag_util_implausible = 1 THEN 'X_IMPLAUSIBLE' END AS special
    FROM applications
UNION ALL
    SELECT id, split, target, 'age', age, NULL FROM applications
UNION ALL
    SELECT id, split, target, 'debt_ratio_valid', debt_ratio_valid,
           CASE WHEN flag_debt_ratio_invalid = 1 THEN 'X_INVALID' END
    FROM applications
UNION ALL
    SELECT id, split, target, 'monthly_income',
           CASE WHEN monthly_income IS NULL OR monthly_income = 0 THEN NULL
                ELSE monthly_income END,
           CASE WHEN monthly_income IS NULL THEN 'X_MISSING'
                WHEN monthly_income = 0    THEN 'X_ZERO' END
    FROM applications
UNION ALL
    SELECT id, split, target, 'open_credit_lines', open_credit_lines, NULL FROM applications

    -- ---- roi rac ------------------------------------------------------------
UNION ALL
    SELECT id, split, target, 'late_30_59', NULL,
           CASE WHEN flag_sentinel = 1 THEN '9_SENTINEL'
                WHEN late_30_59 = 0    THEN '0'
                WHEN late_30_59 = 1    THEN '1'
                WHEN late_30_59 = 2    THEN '2'
                WHEN late_30_59 <= 4   THEN '3-4'
                ELSE '5+' END
    FROM applications
UNION ALL
    SELECT id, split, target, 'late_60_89', NULL,
           CASE WHEN flag_sentinel = 1 THEN '9_SENTINEL'
                WHEN late_60_89 = 0    THEN '0'
                WHEN late_60_89 = 1    THEN '1'
                WHEN late_60_89 = 2    THEN '2'
                ELSE '3+' END
    FROM applications
UNION ALL
    SELECT id, split, target, 'late_90', NULL,
           CASE WHEN flag_sentinel = 1 THEN '9_SENTINEL'
                WHEN late_90 = 0       THEN '0'
                WHEN late_90 = 1       THEN '1'
                WHEN late_90 = 2       THEN '2'
                WHEN late_90 <= 4      THEN '3-4'
                ELSE '5+' END
    FROM applications
UNION ALL
    SELECT id, split, target, 'real_estate_loans', NULL,
           CASE WHEN real_estate_loans = 0 THEN '0'
                WHEN real_estate_loans = 1 THEN '1'
                WHEN real_estate_loans = 2 THEN '2'
                ELSE '3+' END
    FROM applications
UNION ALL
    SELECT id, split, target, 'dependents', NULL,
           CASE WHEN dependents IS NULL THEN '9_MISSING'
                WHEN dependents = 0     THEN '0'
                WHEN dependents = 1     THEN '1'
                WHEN dependents = 2     THEN '2'
                ELSE '3+' END
    FROM applications;

CREATE INDEX idx_row_values ON row_values(variable, split);


-- -----------------------------------------------------------------------------
-- 2. DIEM CAT BIN, LAY TU TRAIN
--
-- NTILE(10) chia deu SO DONG, nen khi mot gia tri xuat hien nhieu lan no bi cat
-- ngang giua hai nhom. Vi vay khong dung truc tiep so nhom cua NTILE lam bin, ma
-- lay CAN TREN cua tung nhom lam diem cat, roi DISTINCT de bo cac diem cat trung
-- nhau. Nho vay moi gia tri bang nhau luon roi vao cung mot bin, va so bin thuc
-- te co the it hon 10 voi bien roi rac.
--
-- Diem cat cuoi (gia tri lon nhat) bi bo di de bin tren cung mo ve phia phai,
-- nho do gia tri lon hon bat ky gia tri nao tung thay trong train van gan duoc bin.
-- -----------------------------------------------------------------------------
CREATE TABLE bin_cuts AS
WITH ntiles AS (
    SELECT variable, val,
           NTILE(10) OVER (PARTITION BY variable ORDER BY val) AS nt
    FROM row_values
    WHERE split = 'train' AND special IS NULL AND val IS NOT NULL
),
group_max AS (
    SELECT variable, nt, MAX(val) AS hi
    FROM ntiles
    GROUP BY variable, nt
),
distinct_cuts AS (
    SELECT DISTINCT variable, hi FROM group_max
),
ranked AS (
    SELECT variable, hi,
           ROW_NUMBER() OVER (PARTITION BY variable ORDER BY hi) AS cut_no,
           COUNT(*)     OVER (PARTITION BY variable)             AS n_cuts
    FROM distinct_cuts
)
SELECT variable, cut_no, hi AS upper_bound
FROM ranked
WHERE cut_no < n_cuts;

CREATE INDEX idx_bin_cuts ON bin_cuts(variable);


-- -----------------------------------------------------------------------------
-- 3. GAN BIN CHO TOAN BO CAC DONG
--
-- Ap dung diem cat cua train len ca train, test va oot. So bin = 1 cong so diem
-- cat ma gia tri vuot qua. Dong co `special` giu nguyen ma dac biet.
-- -----------------------------------------------------------------------------
CREATE TABLE row_bins AS
SELECT r.id, r.split, r.target, r.variable,
       COALESCE(
           r.special,
           printf('%02d', 1 + SUM(CASE WHEN r.val > c.upper_bound THEN 1 ELSE 0 END))
       ) AS bin
FROM row_values r
LEFT JOIN bin_cuts c ON c.variable = r.variable
GROUP BY r.id, r.variable;

CREATE INDEX idx_row_bins ON row_bins(variable, bin, split);


-- -----------------------------------------------------------------------------
-- 4. BANG TRA WOE, TINH TREN TRAIN
--
--   g = so good trong bin / TONG so good toan tap train
--   b = so bad  trong bin / TONG so bad  toan tap train
--   WOE = ln(g / b)  = log-odds cua bin  -  log-odds cua quan the
--
-- Mau so la tong toan tap chu khong phai tong dong trong bin. WOE duong nghia la
-- bin an toan hon muc nen.
--
-- CROSS JOIN voi mot dong tong de moi bin deu chia cho cung mot mau so.
-- -----------------------------------------------------------------------------
CREATE TABLE woe_lookup AS
WITH totals AS (
    SELECT SUM(CASE WHEN target = 0 THEN 1 ELSE 0 END) AS n_good,
           SUM(target)                                 AS n_bad
    FROM applications
    WHERE split = 'train'
),
agg AS (
    SELECT variable, bin,
           COUNT(*)                  AS n,
           SUM(target)               AS n_bad,
           COUNT(*) - SUM(target)    AS n_good
    FROM row_bins
    WHERE split = 'train'
    GROUP BY variable, bin
)
SELECT a.variable,
       a.bin,
       a.n,
       a.n_good,
       a.n_bad,
       ROUND(100.0 * a.n / (SELECT COUNT(*) FROM applications WHERE split='train'), 3) AS pct_rows,
       ROUND(100.0 * a.n_bad / a.n, 3)                     AS bad_rate_pct,
       ROUND(1.0 * a.n_good / t.n_good, 6)                 AS pct_good,
       ROUND(1.0 * a.n_bad  / t.n_bad,  6)                 AS pct_bad,
       ROUND(LN((1.0 * a.n_good / t.n_good) /
                (1.0 * a.n_bad  / t.n_bad)), 6)            AS woe,
       ROUND((1.0 * a.n_good / t.n_good - 1.0 * a.n_bad / t.n_bad) *
             LN((1.0 * a.n_good / t.n_good) /
                (1.0 * a.n_bad  / t.n_bad)), 6)            AS iv_part
FROM agg a
CROSS JOIN totals t
ORDER BY a.variable, a.bin;

CREATE INDEX idx_woe_lookup ON woe_lookup(variable, bin);


-- -----------------------------------------------------------------------------
-- 5. IV MOI BIEN
-- -----------------------------------------------------------------------------
CREATE TABLE iv_summary AS
SELECT variable,
       COUNT(*)                  AS n_bins,
       ROUND(SUM(iv_part), 4)    AS iv,
       CASE WHEN SUM(iv_part) < 0.02 THEN 'vo dung'
            WHEN SUM(iv_part) < 0.10 THEN 'yeu'
            WHEN SUM(iv_part) < 0.30 THEN 'trung binh'
            WHEN SUM(iv_part) < 0.50 THEN 'manh'
            ELSE 'rat manh - kiem leakage' END AS muc
FROM woe_lookup
GROUP BY variable
ORDER BY iv DESC;


-- -----------------------------------------------------------------------------
-- 6. AP WOE NGUOC LEN TOAN BO CAC DONG
--
-- LEFT JOIN chu khong INNER JOIN: neu mot bin xuat hien o test hoac oot ma khong
-- co trong train thi WOE se la NULL va lo ra ngay o buoc kiem tra, thay vi dong
-- do bi bien mat khoi ket qua trong im lang.
-- -----------------------------------------------------------------------------
CREATE TABLE features_woe AS
SELECT b.id,
       MAX(b.split)  AS split,
       MAX(b.target) AS target,
       MAX(CASE WHEN b.variable = 'revolving_util'    THEN w.woe END) AS woe_revolving_util,
       MAX(CASE WHEN b.variable = 'age'               THEN w.woe END) AS woe_age,
       MAX(CASE WHEN b.variable = 'debt_ratio_valid'  THEN w.woe END) AS woe_debt_ratio,
       MAX(CASE WHEN b.variable = 'monthly_income'    THEN w.woe END) AS woe_monthly_income,
       MAX(CASE WHEN b.variable = 'open_credit_lines' THEN w.woe END) AS woe_open_credit_lines,
       MAX(CASE WHEN b.variable = 'late_30_59'        THEN w.woe END) AS woe_late_30_59,
       MAX(CASE WHEN b.variable = 'late_60_89'        THEN w.woe END) AS woe_late_60_89,
       MAX(CASE WHEN b.variable = 'late_90'           THEN w.woe END) AS woe_late_90,
       MAX(CASE WHEN b.variable = 'real_estate_loans' THEN w.woe END) AS woe_real_estate_loans,
       MAX(CASE WHEN b.variable = 'dependents'        THEN w.woe END) AS woe_dependents
FROM row_bins b
LEFT JOIN woe_lookup w
       ON w.variable = b.variable AND w.bin = b.bin
GROUP BY b.id;

CREATE INDEX idx_features_woe ON features_woe(split);
