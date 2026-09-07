-- =============================================================================
-- Chia bin don dieu  --  Give Me Some Credit, khoi 4
--
-- Chay:  python src/monotone_bins.py   (script do ghi bang bin_map truoc, roi goi file nay)
--
-- File nay KHONG quyet dinh gop bin nao. Viec do do pool-adjacent-violators lam
-- trong src/monotone_bins.py va ket qua nam san trong bang `bin_map`. O day chi
-- ap bang anh xa do roi TINH LAI WOE, bang dung mot cau GROUP BY va mot CROSS
-- JOIN voi dong tong y het muc 4 cua features.sql. Nghia la khong con so WOE nao
-- ra doi ngoai SQL.
--
-- Bang khoi 2 giu nguyen khong dung toi, de con doi chieu truoc/sau.
--
-- Bang tao ra:
--   row_bins_mono     bin da gop, cho TOAN BO cac dong
--   woe_lookup_mono   bang tra WOE/IV tren bin da gop, tinh tren TRAIN
--   iv_summary_mono   IV mot dong moi bien
--   features_woe_mono ma tran model: mot cot WOE moi bien
-- =============================================================================

DROP TABLE IF EXISTS row_bins_mono;
DROP TABLE IF EXISTS woe_lookup_mono;
DROP TABLE IF EXISTS iv_summary_mono;
DROP TABLE IF EXISTS features_woe_mono;


-- -----------------------------------------------------------------------------
-- 1. AP BANG ANH XA
--
-- JOIN chu khong LEFT JOIN, va day la mot bo loc CO Y: bin_map chi chua 9 bien
-- duoc giu lai, nen open_credit_lines roi ra o buoc nay. So dong ky vong la
-- 149.999 x 9 = 1.349.991, va notebook 04 kiem dung con so do. Neu dung LEFT
-- JOIN thi bien bi loai se di tiep voi bin_mono NULL va lo ra muon hon.
-- -----------------------------------------------------------------------------
CREATE TABLE row_bins_mono AS
SELECT b.id, b.split, b.target, b.variable, m.bin_mono AS bin
FROM row_bins b
JOIN bin_map m ON m.variable = b.variable AND m.bin = b.bin;

CREATE INDEX idx_row_bins_mono ON row_bins_mono(variable, bin, split);


-- -----------------------------------------------------------------------------
-- 2. BANG TRA WOE TREN BIN DA GOP, TINH TREN TRAIN
--
-- Y het muc 4 cua features.sql: mau so la TONG good/bad cua ca tap train, lay
-- bang CROSS JOIN voi mot dong tong.
--
-- Chu y: WOE cua bin gop KHONG bang trung binh WOE cac bin thanh phan. No duoc
-- tinh lai tu so good/bad cong don, va do moi la con so dung. Gia tri trung binh
-- co trong so ma PAVA tra ve chi dung de xac dinh NHOM nao gop voi nhom nao.
-- -----------------------------------------------------------------------------
CREATE TABLE woe_lookup_mono AS
WITH totals AS (
    SELECT SUM(CASE WHEN target = 0 THEN 1 ELSE 0 END) AS n_good,
           SUM(target)                                 AS n_bad
    FROM applications
    WHERE split = 'train'
),
agg AS (
    SELECT variable, bin,
           COUNT(*)               AS n,
           SUM(target)            AS n_bad,
           COUNT(*) - SUM(target) AS n_good
    FROM row_bins_mono
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

CREATE INDEX idx_woe_lookup_mono ON woe_lookup_mono(variable, bin);


-- -----------------------------------------------------------------------------
-- 3. IV MOI BIEN
-- -----------------------------------------------------------------------------
CREATE TABLE iv_summary_mono AS
SELECT variable,
       COUNT(*)               AS n_bins,
       ROUND(SUM(iv_part), 4) AS iv,
       CASE WHEN SUM(iv_part) < 0.02 THEN 'vo dung'
            WHEN SUM(iv_part) < 0.10 THEN 'yeu'
            WHEN SUM(iv_part) < 0.30 THEN 'trung binh'
            WHEN SUM(iv_part) < 0.50 THEN 'manh'
            ELSE 'rat manh - kiem leakage' END AS muc
FROM woe_lookup_mono
GROUP BY variable
ORDER BY iv DESC;


-- -----------------------------------------------------------------------------
-- 4. AP WOE NGUOC LEN TOAN BO CAC DONG
--
-- LEFT JOIN nhu khoi 2: bin nao xuat hien o test/oot ma train chua thay se thanh
-- NULL va lo ra o buoc kiem, thay vi lam bien mat dong do trong im lang.
--
-- Ten cot lan nay dat theo dung ten bien (woe_debt_ratio_valid), khong theo ten
-- cot goc nhu features_woe cua khoi 2 (woe_debt_ratio). Cai ten cu la mot vet
-- xuoc: no buoc scorecard.py phai co mot nhanh doi ten rieng cho mot bien.
-- -----------------------------------------------------------------------------
CREATE TABLE features_woe_mono AS
SELECT b.id,
       MAX(b.split)  AS split,
       MAX(b.target) AS target,
       MAX(CASE WHEN b.variable = 'revolving_util'    THEN w.woe END) AS woe_revolving_util,
       MAX(CASE WHEN b.variable = 'age'               THEN w.woe END) AS woe_age,
       MAX(CASE WHEN b.variable = 'debt_ratio_valid'  THEN w.woe END) AS woe_debt_ratio_valid,
       MAX(CASE WHEN b.variable = 'monthly_income'    THEN w.woe END) AS woe_monthly_income,
       MAX(CASE WHEN b.variable = 'late_30_59'        THEN w.woe END) AS woe_late_30_59,
       MAX(CASE WHEN b.variable = 'late_60_89'        THEN w.woe END) AS woe_late_60_89,
       MAX(CASE WHEN b.variable = 'late_90'           THEN w.woe END) AS woe_late_90,
       MAX(CASE WHEN b.variable = 'real_estate_loans' THEN w.woe END) AS woe_real_estate_loans,
       MAX(CASE WHEN b.variable = 'dependents'        THEN w.woe END) AS woe_dependents
FROM row_bins_mono b
LEFT JOIN woe_lookup_mono w
       ON w.variable = b.variable AND w.bin = b.bin
GROUP BY b.id;

CREATE INDEX idx_features_woe_mono ON features_woe_mono(split);
