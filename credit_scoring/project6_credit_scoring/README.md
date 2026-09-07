# Chấm điểm tín dụng (Give Me Some Credit) — Scorecard + WOE bằng SQL

Pipeline ước lượng PD: feature engineering WOE bằng SQL, scorecard hồi quy logistic quy đổi ra thang
điểm cộng được bằng tay, đối chiếu với XGBoost, rồi calibration và PSI. Bối cảnh là một tổ chức cho
vay vừa cần xếp hạng rủi ro chính xác, vừa phải nêu được lý do cụ thể cho từng hồ sơ bị từ chối.

## Dữ liệu

Give Me Some Credit (Kaggle): 150.000 hồ sơ, target `SeriousDlqin2yrs` (trễ 90+ ngày trong 2 năm),
bad rate 6,68%. Mười biến số, không biến hạng mục, không có cột thời gian.

Ba cái bẫy, cả ba đều đổi cách làm sạch. Mã 96 và 98 trong các cột đếm số lần trễ hạn là cờ ở mức
bản ghi, không phải số lần trễ: 269 dòng mang nó ở cả ba cột cùng lúc, bad rate nhóm đó 54,7% so với
6,6% phần còn lại. `DebtRatio` là hai biến khác đơn vị chung một cột, median 0,29 khi có thu nhập và
1.159 khi thu nhập missing, nhân ngược ra median 1.649 USD một tháng. Và missing mang thông tin ngược
trực giác: người không khai thu nhập có bad rate 5,6% so với 7,0%, vì nhóm đó nghiêng về người đã
nghỉ hưu. Cả ba được giữ và đánh cờ để bước binning định giá, không impute và không xoá.

## Pipeline

```
cs-training.csv
  -> data_prep.py       lam sach bang co, split 70/15/15 phan tang, 1 bang SQLite kem cot split
  -> features.sql       NTILE(10) tren train -> bin_cuts -> row_bins -> woe_lookup -> features_woe
  -> scorecard.py       logistic khong phat tren 9 cot WOE -> bang diem PDO 20 / 600 = odds 50:1
  -> monotone_bins.py   PAVA gop bin theo chieu nghiep vu khai tay -> 70 bin con 60
  -> tree_model.py      ba model XGBoost, moi buoc doi dung mot thu
  -> calibration.py, psi.py
```

Mọi tham số học được, gồm cả điểm cắt bin và giá trị WOE, chỉ tính trên `split = 'train'` rồi áp
sang test và oot bằng `LEFT JOIN`; một bảng duy nhất kèm cột `split` thay vì ba bảng. Chọn biến bằng
5-fold CV trong train với WOE tính lại trong từng fold, và `oot` chỉ được đọc nhãn ở bước cuối.

## Kết quả

| | train Gini | test Gini | test KS | oot Gini | oot Brier | PD TB oot |
|---|---|---|---|---|---|---|
| scorecard đơn điệu | 0,7162 | 0,7006 | 0,5440 | 0,7124 | 0,0506 | 6,90% |
| XGBoost, đủ biến | 0,7500 | 0,7208 | 0,5752 | 0,7297 | 0,0490 | 6,88% |

Bad rate thực trên oot 6,68%; PSI của điểm số 0,0012 và CSI chín biến đều dưới 0,0006. Khoảng tin
cậy 95% của Gini ở cỡ mẫu này là ±0,025, nên mọi so sánh đều làm bằng CV ghép cặp trong train, không bằng cách nhìn hai con số test.

Trên cùng 9 biến, XGBoost hơn scorecard 0,0143 Gini. Tách bằng một cây `max_depth=1`, vốn cộng tính
nhưng hình dạng tự do, thì phần lớn nằm ở tương tác giữa các biến, không ở chỗ cắt bin: cột WOE đã là log-odds
thực nghiệm của từng bin nên hình dạng cộng tính tối ưu nằm sẵn trong đó.

## Bốn phát hiện đáng kể nhất

**IV nói dối theo hai kiểu.** `real_estate_loans` có IV 0,0052, dưới ngưỡng 0,02 mà giáo trình dùng
để loại biến, nhưng bỏ nó khỏi model mất 0,0029 Gini, hơn cả hai biến có IV gấp nó 14 lần: hệ số của
nó là 1,00 khi đứng một mình và 1,88 trong model đủ, tức suppression, mà IV là đại lượng đơn biến nên
không thấy. Kiểu thứ hai là IV thuộc về cặp biến và cách chia bin: cùng một cột,
ba cách chia cho IV 0,0000, 0,5864 và 0,8783, trong đó cách cho 0,5864 chia thành chín bin đều chỉ
chứa giá trị 0 rồi trả về một con số trông hợp lý.

**84,7% hồ sơ bị từ chối ở ngưỡng 580 nhận cùng một mã lý do.** `revolving_util` có biên độ 57 điểm
trên thang 385 đến 640 nên hầu như luôn thắng khi xếp hạng nguyên nhân mất điểm. Đúng kỹ thuật,
nhưng một thông báo từ chối mà 5 trên 6 người nhận nội dung giống nhau thì không phục vụ được mục
đích của Regulation B; chia bin đơn điệu kéo xuống 74,3% mà không sửa được nguyên nhân.

**Model xếp hạng tốt nhưng nói sai odds.** PD dự báo bị nén về giữa: decile rủi ro nhất dự báo 34,6%
so với thực tế 35,4%, decile an toàn nhất 0,85% so với 0,36%. Trung bình thì khớp, nhưng mọi model fit bằng
hợp lý cực đại đều có tính chất đó, theo phương trình chuẩn tắc.

**PSI báo động trong khi model vẫn nói đúng PD.** Lấy mẫu lại oot thiên về nhóm `revolving_util` cao
với trọng số chỉ phụ thuộc X: PSI lên 0,41 và bad rate thực tăng từ 6,7% lên 10,7%, nhưng PD dự báo
trung bình bám sát ở 11,0%; thứ xuống cấp là xếp hạng, Gini rơi từ 0,713 xuống 0,660. Trên chính
quần thể đó, để mỗi kỳ tự chia decile thay vì dùng ranh giới đóng băng từ train thì PSI ra 0,000004.

## Giới hạn

Không có cột thời gian nên không có vintage analysis và không có out-of-time thật. Tập "oot" là một
lát cắt ngẫu nhiên phân tầng, nên PSI và calibration trên nó đẹp theo thiết kế: đó là mức nền và
phép kiểm pipeline, không phải bằng chứng model bền với thời gian.

Split cắt theo dòng, không theo nhóm feature, nên 261 dòng ở test và oot có bản sao feature y hệt
trong train. Điểm cắt bin cũng không được tính lại trong từng fold CV, chỉ WOE mới được. Reject
inference nằm ngoài phạm vi vì dữ liệu chỉ có hồ sơ đã được duyệt.

Giả thuyết ghi trước khi chạy để các bước sau chấm, và kết quả ghi lại kể
cả khi sai: hai dự đoán về cách chia bin đã kiểm ở khối 3, một đúng nửa và một sai; bốn giả thuyết ở
`notes_credit_scoring.md` §10 đã chấm hết, hai đúng, một sai vì ước lượng cao, một rộng tay về con số.

Việc tiếp theo: chia bin lại cho `open_credit_lines` rồi cân nhắc đưa trở lại scorecard, vì biến này
bị loại do đóng góp biên bằng không nhưng đo lại thì mang một hiệu ứng chính đáng 0,002 đến 0,003
Gini mà dạng hàm một hệ số cho mỗi cột WOE không dùng được.

## Chạy lại

```
pip install -r requirements.txt
# tai cs-training.csv tu Kaggle vao data/
python src/data_prep.py          # CSV -> bang applications trong SQLite
python src/build_features.py     # applications -> 6 bang WOE
python src/scorecard.py          # he so, bang diem 70 dong, round-trip
python src/monotone_bins.py      # bin_map + 4 bang *_mono
python src/scorecard.py --mono   # bang diem 60 dong don dieu
```

Rồi năm notebook theo thứ tự `01` đến `05`.

## File

- `notes_credit_scoring.md` — kiến thức nền: PD, leakage, WOE/IV, Gini/KS, calibration, PSI, scorecard so với cây
- `sql/` — feature engineering WOE bằng SQL: CTE, window function, GROUP BY, CASE, JOIN
- `src/` — `config`, `data_prep`, `build_features`, `woe_check`, `scorecard`, `cv_check`, `monotone_bins`, `tree_model`, `calibration`, `psi`, `doi_chieu_so`
- `notebooks/01` đến `05` — khảo sát; WOE bằng SQL; scorecard; đơn điệu và XGBoost; calibration và PSI
- `results/` — `data_profile`, `iv_report`, `scorecard`, `model_comparison`, `calibration_psi`: số liệu đầy đủ và lý lẽ cho từng quyết định
- `requirements.txt` — thư viện và mốc phiên bản, kèm lý do cho từng mốc
- `data/` — không track; tải `cs-training.csv` từ Kaggle

`src/doi_chieu_so.py` đối chiếu mọi con số trong `results/*.md` với output đang lưu trong notebook.
Nó tồn tại vì đã có lần một bảng số trong báo cáo không còn tái lập được bằng code trong repo.

## Công nghệ

Python, pandas, numpy, SQLite, scikit-learn, scipy, xgboost.
