# Chấm điểm tín dụng (Give Me Some Credit) — Scorecard + WOE bằng SQL

> **Đang làm.** Đã xong: nền tảng khái niệm, EDA & làm sạch, split, nạp SQLite. Chưa làm: WOE bằng SQL, scorecard, XGBoost, đánh giá.

## Mục tiêu
Xây pipeline ước lượng PD (probability of default): feature engineering **WOE bằng SQL** (SQLite), scorecard hồi quy logistic đối chiếu với XGBoost, đánh giá bằng Gini/KS, calibration và PSI — đặt trong bối cảnh một tổ chức cho vay vừa cần xếp hạng rủi ro chính xác, vừa phải giải thích được từng quyết định từ chối.

## Dữ liệu
Give Me Some Credit (Kaggle): 150.000 hồ sơ, target `SeriousDlqin2yrs` (trễ 90+ ngày trong 2 năm), bad rate **6,684%**. Mười biến số, không có biến hạng mục, **không có cột thời gian**. Feature là hành vi tín dụng của khách đã có quan hệ tín dụng, nên đây là behavioral scorecard chứ không phải application scorecard.

## Quy trình (phần đã xong)
1. **Khảo sát dữ liệu:** mọi quyết định làm sạch đều kèm số liệu chạy trực tiếp trên file, không lấy từ mô tả của Kaggle.
2. **Làm sạch bằng cờ, không impute và không xoá:** missing ở bộ này *mang thông tin* nên được đánh cờ và để dành cho bước binning định giá, thay vì bị impute median làm nhoè.
3. **Kiểm cột index trước khi bỏ:** `corr(id, target) = 0,0028`, bad rate phẳng qua 10 khối liên tiếp — không có leakage theo thứ tự dòng.
4. **Split trước khi tính bất cứ tham số nào:** 70/15/15 phân tầng, seed 42, tái lập được. Ranh giới bin và WOE là *tham số học được*, tính trên toàn bộ dữ liệu là để nhãn của test dựng feature cho chính test.
5. **Một bảng SQLite kèm cột `split`,** không tách ba bảng: WOE tính bằng `WHERE split='train'` và áp bằng `LEFT JOIN` lên toàn bộ bảng, nên không có tập nào tồn tại riêng để vô tình fit lên.

## Phát hiện chính
- **Mã 96/98 là cờ ở mức bản ghi, không phải số lần trễ hạn:** cùng 269 dòng mang nó ở *cả ba* cột đếm (269/269), bad rate **54,65% so với 6,60%**. Giữ và đánh cờ để thành một bin WOE riêng; drop hoặc impute là thay tín hiệu mạnh nhất tập bằng giá trị trung bình.
- **`DebtRatio` là hai biến khác đơn vị chung một cột:** median 0,29 khi có thu nhập, 1.159 khi thu nhập missing. Nhân ngược lại cho median 1.649 USD/tháng — đúng cỡ một khoản trả nợ hàng tháng, xác nhận mẫu số hỏng thì cột chuyển thành số tiền tuyệt đối.
- **Missing mang thông tin và theo chiều ngược trực giác:** người *không* khai thu nhập có bad rate **5,61% so với 6,95%**; nhóm này có tuổi trung vị 57 so với 51 và 30,3% từ 65 tuổi trở lên so với 18,5%, tức nghiêng về nhóm hưu trí.
- **`NTILE` hỏng im lặng trên cột 94% giá trị 0:** cùng một biến, ba cách chia bin cho **IV 0,0000 (`qcut`) / 0,5864 (`NTILE`) / 0,8783 (`CASE`)**. `qcut` gộp hết còn một bin nên hỏng *ồn ào*; `NTILE` chia bừa thành chín bin đều chỉ chứa giá trị 0 rồi trả về một con số trông hợp lý. IV là thuộc tính của cặp *(biến, cách chia bin)* chứ không phải của riêng biến.

Ngoài ra: giữ 646 dòng trùng feature (là hồ sơ *thin file*, và 37 nhóm có **nhãn mâu thuẫn** nên đó là những người khác nhau chứ không phải bản sao); bỏ đúng 1 dòng `age = 0`.

## Giới hạn (nêu trước, không giấu)
Dataset **không có cột thời gian** nên không có vintage analysis và không có out-of-time thật. Tập "OOT" ở đây là lát cắt ngẫu nhiên, và việc phân tầng nó làm nó *khác* một OOT thật chứ không giống hơn — PSI, bad rate và calibration trên tập này sẽ đẹp **theo thiết kế**, sẽ được báo cáo như hệ quả của cách cắt tập chứ không phải bằng chứng về model. OOT có 1.504 ca dương nên khoảng tin cậy 95% của Gini là **±0,028**; mọi chênh lệch dưới ~0,03 sẽ ghi là chưa kết luận được. Reject inference nằm ngoài phạm vi vì dữ liệu chỉ có hồ sơ đã được duyệt.

Bốn giả thuyết đã ghi trước khi mô hình hoá (`notes_credit_scoring.md` §10) để các bước sau kiểm chứ không giải thích hồi tố.

## File
- `notes_credit_scoring.md` — tổng hợp kiến thức credit scoring (PD, leakage, WOE/IV, Gini/KS, calibration, PSI, imbalance, scorecard vs cây)
- `notebooks/01_eda.ipynb` — khảo sát, làm sạch, split, đối chiếu
- `src/config.py`, `src/data_prep.py` — hằng số dùng chung và pipeline
- `results/data_profile.md` — nhật ký khảo sát dữ liệu kèm số liệu
- `data/` — dữ liệu (không track; tải `cs-training.csv` từ Kaggle)

## Công nghệ
Python, pandas, numpy, SQLite, scikit-learn, xgboost, matplotlib.
