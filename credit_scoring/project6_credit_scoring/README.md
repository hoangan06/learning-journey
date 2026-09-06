# Chấm điểm tín dụng (Give Me Some Credit) — Scorecard + WOE bằng SQL

> **Đang làm.** Đã xong: nền tảng khái niệm, EDA & làm sạch, split, nạp SQLite, feature engineering WOE bằng SQL, scorecard. Chưa làm: XGBoost, calibration, PSI, đánh giá trên OOT.

## Mục tiêu
Xây pipeline ước lượng PD (probability of default): feature engineering **WOE bằng SQL** (SQLite), scorecard hồi quy logistic đối chiếu với XGBoost, đánh giá bằng Gini/KS, calibration và PSI, đặt trong bối cảnh một tổ chức cho vay vừa cần xếp hạng rủi ro chính xác, vừa phải giải thích được từng quyết định từ chối.

## Dữ liệu
Give Me Some Credit (Kaggle): 150.000 hồ sơ, target `SeriousDlqin2yrs` (trễ 90+ ngày trong 2 năm), bad rate **6,684%**. Mười biến số, không có biến hạng mục, **không có cột thời gian**. Feature là hành vi tín dụng của khách đã có quan hệ tín dụng, nên đây là behavioral scorecard chứ không phải application scorecard.

## Quy trình (phần đã xong)
1. **Khảo sát dữ liệu:** mọi quyết định làm sạch đều kèm số liệu chạy trực tiếp trên file, không lấy từ mô tả của Kaggle.
2. **Làm sạch bằng cờ, không impute và không xoá:** missing ở bộ này *mang thông tin* nên được đánh cờ và để dành cho bước binning định giá, thay vì bị impute median làm nhoè.
3. **Kiểm cột index trước khi bỏ:** `corr(id, target) = 0,0028`, bad rate phẳng qua 10 khối liên tiếp, không có leakage theo thứ tự dòng.
4. **Split trước khi tính bất cứ tham số nào:** 70/15/15 phân tầng, seed 42, tái lập được. Ranh giới bin và WOE là *tham số học được*, tính trên toàn bộ dữ liệu là để nhãn của test dựng feature cho chính test.
5. **Một bảng SQLite kèm cột `split`,** không tách ba bảng: WOE tính bằng `WHERE split='train'` và áp bằng `LEFT JOIN` lên toàn bộ bảng, nên không có tập nào tồn tại riêng để vô tình fit lên.
6. **Feature engineering WOE bằng SQL** (`sql/features.sql`): điểm cắt lấy từ `NTILE(10)` trên train rồi khử trùng để không cắt ngang các giá trị bằng nhau, `CASE` cho biến đếm và mã đặc biệt, `GROUP BY` tính WOE/IV, `LEFT JOIN` bảng tra ngược lên toàn bộ 149.999 dòng. Đối chiếu độc lập bằng pandas: lệch WOE 4,9e-07, lệch IV 0,0.
7. **Scorecard:** logistic không phạt trên 9 cột WOE, quy đổi ra thang điểm PDO 20 / 600 điểm = odds 50:1, sinh bảng điểm 70 dòng và mã lý do. Chọn biến bằng 5-fold CV trong train với WOE **tính lại trong từng fold** (điểm cắt bin thì không, chúng vẫn của cả train, nên phép đo lạc quan hơn thực tế một chút). `test` không tham gia quyết định nào, chỉ để đối chiếu dự đoán đã ghi trước và để báo cáo. Gini test 0,6984, KS 0,5472.

## Phát hiện chính
- **Mã 96/98 là cờ ở mức bản ghi, không phải số lần trễ hạn:** cùng 269 dòng mang nó ở *cả ba* cột đếm (269/269), bad rate **54,65% so với 6,60%**. Giữ và đánh cờ để thành một bin WOE riêng; drop hoặc impute là thay tín hiệu mạnh nhất tập bằng giá trị trung bình.
- **`DebtRatio` là hai biến khác đơn vị chung một cột:** median 0,29 khi có thu nhập, 1.159 khi thu nhập missing. Nhân ngược lại cho median 1.649 USD/tháng, đúng cỡ một khoản trả nợ hàng tháng: xác nhận mẫu số hỏng thì cột chuyển thành số tiền tuyệt đối.
- **Missing mang thông tin và theo chiều ngược trực giác:** người *không* khai thu nhập có bad rate **5,61% so với 6,95%**; nhóm này có tuổi trung vị 57 so với 51 và 30,3% từ 65 tuổi trở lên so với 18,5%, tức nghiêng về nhóm hưu trí.
- **`NTILE` hỏng im lặng trên cột 94% giá trị 0:** cùng một biến, ba cách chia bin cho **IV 0,0000 (`qcut`) / 0,5864 (`NTILE`) / 0,8783 (`CASE`)**. `qcut` gộp hết còn một bin nên hỏng *ồn ào*; `NTILE` chia bừa thành chín bin đều chỉ chứa giá trị 0 rồi trả về một con số trông hợp lý. IV là thuộc tính của cặp *(biến, cách chia bin)* chứ không phải của riêng biến.

- **Hình dạng không đơn điệu chỉ đáng giữ khi cơ chế đằng sau nó không nằm trong biến khác, và ở bộ này thì không biến nào đạt điều kiện đó.** Đo đơn biến, ép đơn điệu tốn 0,0062 đến 0,0214 Gini nên khối 2 quyết định giữ nguyên hình. Đo lại trong **model đa biến**, ép đúng chiều mà vẫn giữ 8 trên 10 mức của `revolving_util` và 6 trên 10 của `debt_ratio_valid` thì cái giá xấu nhất còn tương thích với dữ liệu ở mức tin cậy 95% là 0,0013 Gini, tức nhỏ hơn một bậc so với phép đo đơn biến và nhỏ hơn nhiều so với ±0,028 sai số của chính Gini. Nhánh trái của `open_credit_lines` là nhóm bị hạn chế tín dụng có utilization trung vị 0,439 so với 0,137 ở đáy, thông tin đó đã nằm sẵn trong `revolving_util`; `real_estate_loans` thì thực chất là câu hỏi nhị phân "có từ ba khoản bất động sản trở lên hay không". Dự đoán tôi ghi trước ("chữ U sống sót") **sai**. Hệ quả: scorecard có thể làm đơn điệu mà không mất gì đo được, và đó là việc đầu tiên của khối 4.
- **Chiều của phép ép đơn điệu quan trọng hơn bản thân phép ép, vì ép sai chiều không phải là ép mà là xoá biến.** Với `revolving_util`, PAVA sai chiều gộp cả 10 bin thành một mức và làm mất 0,05133 Gini, gần bằng đúng 0,05109 của việc bỏ hẳn cột khỏi model. Hai đường đi khác hẳn nhau cho cùng một con số tới chữ số thứ tư, nên đó cũng là một phép kiểm chéo cho toàn bộ bộ máy CV.
- **Sức mạnh đơn biến không phải là đóng góp trong model.** Đo trên cùng 5 fold: `monthly_income` có Gini đơn biến 0,153, cao nhất nhóm biến yếu, nhưng bỏ khỏi model chỉ mất 0,00017; `real_estate_loans` có Gini đơn biến 0,119 lại mất 0,0030, gấp gần 20 lần. Tổng Gini đơn biến của 10 biến là 2,41, gấp hơn ba lần Gini model đủ, nên Gini không cộng được. `open_credit_lines` bị loại vì bỏ nó đi Gini *tăng* 0,00004.
- **Model xếp hạng tốt nhưng nói sai odds.** Gini test 0,6984, còn PD dự báo thì bị nén về giữa: decile thấp nhất dự báo 34,44% trong khi thực tế 35,30%, decile cao nhất dự báo 0,85% trong khi thực tế 0,36%. Trung bình vẫn khớp hoàn hảo, nhưng đó là tính chất của hợp lý cực đại chứ không phải thành tích. Hình dạng lặp lại y hệt trên test nên đây là dạng hàm chứ không phải overfit.
- **84,5% hồ sơ bị từ chối nhận cùng một mã lý do.** `revolving_util` có biên độ 57 điểm trên tổng 384 đến 640 nên hầu như luôn thắng khi xếp hạng nguyên nhân mất điểm. Đúng về kỹ thuật, nhưng một thông báo từ chối mà 5 trên 6 người nhận nội dung giống nhau thì không phục vụ được mục đích của Regulation B.

Ngoài ra: giữ 646 dòng trùng feature (là hồ sơ *thin file*, và 37 nhóm có **nhãn mâu thuẫn** nên đó là những người khác nhau chứ không phải bản sao); bỏ đúng 1 dòng `age = 0`.

## Giới hạn (nêu trước, không giấu)
Dataset **không có cột thời gian** nên không có vintage analysis và không có out-of-time thật. Tập "OOT" ở đây là lát cắt ngẫu nhiên, và việc phân tầng nó làm nó *khác* một OOT thật chứ không giống hơn. PSI, bad rate và calibration trên tập này sẽ đẹp **theo thiết kế**, sẽ được báo cáo như hệ quả của cách cắt tập chứ không phải bằng chứng về model. OOT có 1.504 ca dương nên khoảng tin cậy 95% của Gini là **±0,028**; mọi chênh lệch dưới ~0,03 sẽ ghi là chưa kết luận được. Reject inference nằm ngoài phạm vi vì dữ liệu chỉ có hồ sơ đã được duyệt.

Giả thuyết được ghi trước khi chạy để các bước sau kiểm chứ không giải thích hồi tố, và kết quả ghi lại kể cả khi sai: hai dự đoán về cách chia bin (`results/iv_report.md` §2) đã kiểm ở khối 3, cả hai đều không đúng như tôi nghĩ; bốn giả thuyết còn lại (`notes_credit_scoring.md` §10) để dành cho khối 4 và 5.

## File
- `notes_credit_scoring.md` — tổng hợp kiến thức credit scoring (PD, leakage, WOE/IV, Gini/KS, calibration, PSI, imbalance, scorecard vs cây)
- `notebooks/01_eda.ipynb` — khảo sát, làm sạch, split, đối chiếu
- `src/config.py`, `src/data_prep.py` — hằng số dùng chung và pipeline
- `sql/features.sql` — feature engineering WOE (CTE, window, GROUP BY, CASE, JOIN)
- `notebooks/02_woe_sql.ipynb` — chạy SQL, kiểm toàn vẹn, đối chiếu pandas, kiểm đa cộng tuyến
- `src/build_features.py`, `src/woe_check.py` — chạy SQL và cài đặt lại WOE/IV độc lập để đối chiếu
- `notebooks/03_scorecard.ipynb` — hồi quy logistic, chọn biến, thang điểm, mã lý do, ngưỡng cắt
- `src/scorecard.py`, `src/cv_check.py` — dựng bảng điểm và kiểm chọn biến bằng CV trong train
- `results/data_profile.md` — nhật ký khảo sát dữ liệu kèm số liệu
- `results/iv_report.md` — bảng WOE/IV đầy đủ và lý lẽ chọn cách chia bin
- `results/scorecard.md` — hệ số, bảng điểm 70 dòng, calibration, mã lý do, ngưỡng cắt
- `data/` — dữ liệu (không track; tải `cs-training.csv` từ Kaggle)

## Công nghệ
Python, pandas, numpy, SQLite, scikit-learn, xgboost, matplotlib.
