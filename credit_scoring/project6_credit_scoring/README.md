# Chấm điểm tín dụng (Give Me Some Credit) — Scorecard + WOE bằng SQL

> **Đang làm.** Đã xong: nền tảng khái niệm, EDA & làm sạch, split, nạp SQLite, feature engineering WOE bằng SQL, scorecard, chia bin đơn điệu, đối chiếu XGBoost. Chưa làm: calibration, PSI, đánh giá trên OOT.

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
8. **Chia bin đơn điệu và XGBoost:** PAVA gộp bin theo chiều nghiệp vụ **khai báo tay** trong `config.MONOTONE_DIRECTION` (Python quyết định gộp bin nào, SQL tính lại WOE, cầu nối là bảng `bin_map`), 70 bin còn 60. Rồi ba model cây, mỗi bước chỉ đổi đúng một thứ, để tách phần XGBoost vượt scorecard thành từng nguyên nhân.

## Phát hiện chính
- Mã 96 và 98 là cờ ở mức bản ghi chứ không phải số lần trễ hạn: cùng 269 dòng mang nó ở *cả ba* cột đếm, và bad rate của nhóm đó là **54,65%** so với 6,60% phần còn lại. Giữ và đánh cờ để thành một bin WOE riêng; drop hoặc impute là thay tín hiệu mạnh nhất tập bằng giá trị trung bình.
- **`DebtRatio` là hai biến khác đơn vị chung một cột:** median 0,29 khi có thu nhập, 1.159 khi thu nhập missing. Nhân ngược lại cho median 1.649 USD/tháng, đúng cỡ một khoản trả nợ hàng tháng: xác nhận mẫu số hỏng thì cột chuyển thành số tiền tuyệt đối.
- **Missing mang thông tin và theo chiều ngược trực giác:** người *không* khai thu nhập có bad rate **5,61% so với 6,95%**; nhóm này có tuổi trung vị 57 so với 51 và 30,3% từ 65 tuổi trở lên so với 18,5%, tức nghiêng về nhóm hưu trí.
- Cùng một biến, ba cách chia bin cho IV **0,0000** (`qcut`), **0,5864** (`NTILE`) và **0,8783** (`CASE`). `qcut` gộp hết còn một bin nên hỏng *ồn ào*; `NTILE` chia bừa thành chín bin đều chỉ chứa giá trị 0 rồi trả về một con số trông hợp lý. IV là thuộc tính của cặp *(biến, cách chia bin)* chứ không phải của riêng biến.
- **Hình dạng không đơn điệu chỉ đáng giữ khi cơ chế đằng sau nó không nằm trong biến khác, và ở bộ này thì không biến nào đạt điều kiện đó.** Đo đơn biến, ép đơn điệu đúng chiều tốn 0,0063 đến 0,0205 Gini nên khối 2 quyết định giữ nguyên hình. Đo lại trong **model đa biến** thì khác hẳn. Ép đúng chiều vẫn giữ được 8 trên 10 mức của `revolving_util` và 6 trên 10 của `debt_ratio_valid`, và ép cả chín biến cùng lúc thì cái giá xấu nhất ở mức tin cậy 95% chỉ là 0,0019 Gini. Nhỏ hơn một bậc so với phép đo đơn biến, và nhỏ hơn nhiều so với ±0,028 sai số của chính Gini. Nhánh trái của `open_credit_lines` là nhóm bị hạn chế tín dụng có utilization trung vị 0,439 so với 0,137 ở đáy, thông tin đó đã nằm sẵn trong `revolving_util`; `real_estate_loans` thì thực chất là câu hỏi nhị phân "có từ ba khoản bất động sản trở lên hay không". Dự đoán tôi ghi trước ("chữ U sống sót") **sai**. Hệ quả: scorecard có thể làm đơn điệu mà không mất gì đo được, và đó là việc đầu tiên của khối 4.
- **Chiều của phép ép đơn điệu quan trọng hơn bản thân phép ép, vì ép sai chiều không phải là ép mà là xoá biến.** Với `revolving_util`, PAVA sai chiều gộp cả 10 bin thường thành một mức và làm mất 0,05119 Gini, gần bằng đúng 0,05107 của việc bỏ hẳn cột. Hai con số trùng nhau không phải tình cờ: một cột hằng số thì bị intercept hấp thụ hết, nên ép sai chiều bằng bỏ cột theo cấu tạo.
- Đo trên cùng 5 fold thì sức mạnh đơn biến và đóng góp trong model nói hai chuyện khác nhau: `monthly_income` có Gini đơn biến 0,153, cao nhất nhóm biến yếu, nhưng bỏ khỏi model chỉ mất 0,00017; `real_estate_loans` có Gini đơn biến 0,119 lại mất 0,0030, gấp gần 20 lần. Tổng Gini đơn biến của 10 biến là 2,41, gấp hơn ba lần Gini model đủ, nên Gini không cộng được. `open_credit_lines` bị loại vì bỏ nó đi Gini *tăng* 0,00004.
- Model xếp hạng tốt nhưng nói sai odds. Gini test 0,6985, còn PD dự báo thì bị nén về giữa: decile thấp nhất dự báo 34,44% trong khi thực tế 35,30%, decile cao nhất dự báo 0,85% trong khi thực tế 0,36%. Trung bình vẫn khớp hoàn hảo, nhưng đó là tính chất của hợp lý cực đại chứ không phải thành tích. Hình dạng lặp lại y hệt trên test nên đây là dạng hàm chứ không phải overfit.
- Ở ngưỡng 580, **84,7%** hồ sơ bị từ chối nhận cùng một mã lý do số 1. `revolving_util` có biên độ 57 điểm trên tổng 385 đến 640 nên hầu như luôn thắng khi xếp hạng nguyên nhân mất điểm. Đúng về kỹ thuật, nhưng một thông báo từ chối mà 5 trên 6 người nhận nội dung giống nhau thì không phục vụ được mục đích của Regulation B.
- **Một biến "vô dụng" theo IV lại là biến quan trọng thứ sáu, vì IV là đại lượng đơn biến.** Sau khi gộp bin cho đơn điệu, `real_estate_loans` còn IV **0,0052**, dưới ngưỡng 0,02 mà mọi giáo trình dùng để loại biến, nhưng bỏ nó khỏi model mất **0,0029 Gini** (t = −4,58, năm fold cùng dấu), trên cả hai biến có IV gấp nó 14 lần. Hệ số của nó là 1,000 khi đứng một mình và 1,877 trong model đủ: đó là **suppression**, thu nhập cao của nhóm nhiều bất động sản (trung vị 9.032 so với 5.166) che mất rủi ro đòn bẩy. Bảng 2×2 tách được binning khỏi suppression, và cho thấy **dấu của hiệu chỉnh đa biến đảo ngược theo cách chia bin**.
- **Phần XGBoost vượt scorecard chủ yếu là tương tác, không phải chỗ cắt bin.** Trên cùng 9 biến, XGBoost hơn scorecard **0,0143 Gini** (t = 19,2). Tách bằng cây `max_depth=1`, vốn là model cộng tính có hình dạng tự do: tự do hình dạng đáng **+0,0005** và không phân biệt được với 0, còn tương tác đáng **+0,0091** (t = 9,9), gấp 17 lần. Cột WOE đã là log-odds thực nghiệm của từng bin nên hình dạng cộng tính tối ưu đã nằm sẵn trong đó; đó là một lập luận bênh vực cách mã hoá WOE. Phần còn lại chia gần đều cho chỗ cắt bin (+0,0044) và biến bị loại (+0,0035).
- Một biến mà scorecard không dùng được chưa chắc đã vô dụng. Khối 3 loại `open_credit_lines` vì đóng góp biên bằng không (+0,00004). Giữ nguyên bộ bin của khối 2 và chỉ đổi **số tham số** dành cho nó: một hệ số cho +0,00004, chín cột dummy cho **+0,0021** (t = 3,9), và một cây cộng tính `depth=1` trên chính cột WOE đó cho +0,0023. Vậy giá trị của nó là **hiệu ứng chính**, thứ giết nó là **ràng buộc một hệ số cho cả cột WOE** chứ không phải cách chia bin và cũng không phải thiếu tương tác. Tôi đã kết luận sai hai lần trước khi đo bằng một phép chỉ đổi một thứ.
- **Đơn điệu hoá mua được lời giải thích ở biến này và làm hỏng lời giải thích ở biến khác.** 70 bin còn 60, cái giá đo bằng CV là 0,0019 Gini ở cận trên, tức không đo được, và mã lý do bớt tập trung từ 84,7% xuống 74,3%. Nhưng với `real_estate_loans`, bảng điểm đơn điệu tách nhóm 0 khoản (bad rate thô 8,33%) và nhóm 3+ khoản (8,46%) ra **15 điểm**, trong khi bảng điểm cũ chấm chúng chênh nhau 1 điểm. Câu bảo vệ là câu đa biến, và nó không nói được với khách hàng bị từ chối.
- Bật `scale_pos_weight` thì thứ hạng không tốt lên mà calibration hỏng. Gini chênh −0,0008, xa dưới ngưỡng 0,01, nhưng PD dự báo trung bình nhảy từ 6,69% lên 31,98%, tức **4,8 lần** bad rate thực.
- **Early stopping trên một eval set nhỏ tự tạo ra 0,005 Gini nhiễu, gấp 13 lần số vòng cố định.** Cùng code cùng seed chạy trên Windows và trên Linux cho kết quả khác nhau ở chữ số thứ ba và **đổi cả cấu hình thắng**. Nguyên nhân: eval set chỉ ~16.800 dòng với ~1.120 ca dương nên AUC trên nó rất nhiễu, và có fold dừng ở vòng **8** với `learning_rate = 0,05`. Thay bằng số vòng khai trước theo quy tắc `lr × n_estimators ≈ 25` thì biên độ giữa các seed rơi từ 0,00458 xuống 0,00036, cây được fit trên trọn 80% train đúng bằng scorecard, và bảng CV đọc chéo được với bảng test. Phát hiện này đã lật ba kết luận tôi viết trước đó.

Ngoài ra: giữ 646 dòng trùng feature (là hồ sơ *thin file*, và 37 nhóm có **nhãn mâu thuẫn** nên đó là những người khác nhau chứ không phải bản sao); bỏ đúng 1 dòng `age = 0`.

## Giới hạn
Dataset **không có cột thời gian** nên không có vintage analysis và không có out-of-time thật. Tập "OOT" ở đây là lát cắt ngẫu nhiên, và việc phân tầng nó làm nó *khác* một OOT thật chứ không giống hơn. PSI, bad rate và calibration trên tập này sẽ đẹp **theo thiết kế**, sẽ được báo cáo như hệ quả của cách cắt tập chứ không phải bằng chứng về model. OOT có 1.504 ca dương nên khoảng tin cậy 95% của Gini là **±0,028**; mọi chênh lệch dưới ~0,03 sẽ ghi là chưa kết luận được. Reject inference nằm ngoài phạm vi vì dữ liệu chỉ có hồ sơ đã được duyệt.

Giả thuyết được ghi trước khi chạy để các bước sau kiểm chứ không giải thích hồi tố, và kết quả ghi lại kể cả khi sai: hai dự đoán về cách chia bin (`results/iv_report.md` §2) đã kiểm ở khối 3, cả hai đều không đúng như tôi nghĩ; giả thuyết về `scale_pos_weight` đã kiểm ở khối 4 và đúng; ba giả thuyết còn lại (`notes_credit_scoring.md` §10) để dành cho khối 5.

## File
- `notes_credit_scoring.md` — tổng hợp kiến thức credit scoring (PD, leakage, WOE/IV, Gini/KS, calibration, PSI, imbalance, scorecard vs cây)
- `notebooks/01_eda.ipynb` — khảo sát, làm sạch, split, đối chiếu
- `src/config.py`, `src/data_prep.py` — hằng số dùng chung và pipeline
- `sql/features.sql` — feature engineering WOE (CTE, window, GROUP BY, CASE, JOIN)
- `notebooks/02_woe_sql.ipynb` — chạy SQL, kiểm toàn vẹn, đối chiếu pandas, kiểm đa cộng tuyến
- `src/build_features.py`, `src/woe_check.py` — chạy SQL và cài đặt lại WOE/IV độc lập để đối chiếu
- `notebooks/03_scorecard.ipynb` — hồi quy logistic, chọn biến, thang điểm, mã lý do, ngưỡng cắt
- `notebooks/04_monotone_xgboost.ipynb` — chia bin đơn điệu, nghịch lý IV, XGBoost và SHAP
- `sql/features_mono.sql`, `src/monotone_bins.py` — gộp bin cho đơn điệu rồi tính lại WOE bằng SQL
- `src/tree_model.py` — ba model XGBoost đối chiếu với scorecard trên cùng bộ fold
- `src/scorecard.py`, `src/cv_check.py` — dựng bảng điểm và kiểm chọn biến bằng CV trong train
- `results/data_profile.md` — nhật ký khảo sát dữ liệu kèm số liệu
- `results/iv_report.md` — bảng WOE/IV đầy đủ và lý lẽ chọn cách chia bin
- `results/scorecard.md` — hệ số, bảng điểm 70 dòng, calibration, mã lý do, ngưỡng cắt
- `results/model_comparison.md` — chia bin đơn điệu, nghịch lý IV, phân rã phần XGBoost vượt scorecard
- `data/` — dữ liệu (không track; tải `cs-training.csv` từ Kaggle)

## Công nghệ
Python, pandas, numpy, SQLite, scikit-learn, xgboost, matplotlib.
