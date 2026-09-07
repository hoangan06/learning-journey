# Calibration, PSI và CSI

Kết quả khối 5. Mọi con số lấy từ output đã lưu trong `notebooks/05_calibration_psi.ipynb`.

Dựng lại: `python src/calibration.py` và `python src/psi.py` chạy được độc lập, còn toàn bộ bảng
trong file này sinh ra từ notebook 05.

Ba khối trước đo xếp hạng. Khối này đo hai thứ khác: PD có phải con số thật không, và model
còn dùng được không khi chưa có nhãn. Đây cũng là lần đầu tập `oot` được đọc nhãn trong cả dự án.

Hai quyết định chốt trước khi chạy phép đo nào. Lớp hiệu chỉnh **fit trên `test`, đánh giá trên
`oot`**, vì test đã dùng ở khối 3 và 4 nhưng chỉ để báo cáo, không để chọn gì; cách này giữ
`oot` sạch và cũng đúng hình dạng thật của việc triển khai. Mười bốn dự đoán ghi ở
`notes/du_doan_khoi5.md` trước khi đọc nhãn `oot` lần nào: **mười trúng, ba trượt, một trúng một
nửa**.

---

## 1. Calibration của scorecard

| tập | n | PD TB | bad rate | lệch | Brier | reliability | resolution |
|---|---|---|---|---|---|---|---|
| train | 104.999 | 6,6839% | 6,6839% | 0,0000 | 0,05032 | 0,000042 | 0,01048 |
| test | 22.500 | 6,5985% | 6,6844% | −0,0859 | 0,05021 | 0,000045 | 0,01041 |
| oot | 22.500 | 6,8981% | 6,6844% | +0,2137 | 0,05059 | 0,000054 | 0,01037 |

Trên train hai con số bằng nhau tới chữ số thứ sáu, đúng như phương trình chuẩn tắc của hợp lý
cực đại đòi hỏi. Trên `oot` model dự báo cao hơn thực tế 0,21 điểm phần trăm; đó là dao động lấy
mẫu, không phải thiên lệch, vì `test` lệch theo chiều ngược lại.

### Phân rã Murphy có một phần dư không được bỏ qua

Sách viết `Brier = Reliability − Resolution + Uncertainty`, với Uncertainty = `ȳ(1−ȳ)` = 0,0623.
Đẳng thức đó chỉ đúng **khi p là hằng số trong từng nhóm**. Chia theo decile thì p biến thiên bên
trong nhóm, và phần chênh ra là một số hạng thứ tư:

| k | Brier | reliability | resolution | dư |
|---|---|---|---|---|
| 5 | 0,05059 | 0,000012 | 0,00769 | −0,004110 |
| 10 | 0,05059 | 0,000054 | 0,01037 | −0,001472 |
| 20 | 0,05059 | 0,000182 | 0,01191 | −0,000053 |
| 50 | 0,05059 | 0,000527 | 0,01244 | +0,000134 |
| 100 | 0,05059 | 0,000694 | 0,01253 | +0,000048 |

Phần dư đó tách được chính xác: `dư = Σ wₖ [varₖ(p) − 2·covₖ(p,y)]`. Ở k = 10 hai vế là +0,004252
và −0,005723, cộng lại đúng bằng −0,001472. Vậy **dư âm là do số hạng hiệp phương sai lấn**: trong
cùng một decile, ai bị chấm PD cao hơn thì thật sự hay vỡ nợ hơn, tức bên trong nhóm model vẫn còn
xếp hạng đúng. Nếu chia đủ mịn để p gần như hằng trong nhóm thì cả hai số hạng cùng tắt, và bảng
trên cho thấy đúng như vậy từ k = 20.

Hai điều rút ra, và cả hai là lý do không nên trích một con số Reliability trần trụi. Phần dư ở
k = 5 gấp hơn ba trăm lần chính Reliability, nên bỏ qua nó là đọc Reliability sai mấy bậc độ lớn.
Và Reliability tăng theo k, từ 0,000012 lên 0,000694. Phần lớn là vì chia mịn thì mỗi nhóm còn
ít người và bad rate quan sát nhiễu hơn, nhưng câu đó phải đo mới được nói. Cách đo:
sinh nhãn giả từ chính PD của model, khi đó model calibrated tuyệt đối theo định nghĩa nên
Reliability đo được trên nhãn giả chính là mức nhiễu nền của phép chia ở độ mịn đó.

| k | Reliability đo được | mức nhiễu nền | tỉ lệ |
|---|---|---|---|
| 10 | 0,000054 | 0,000016 | 3,27 |
| 50 | 0,000527 | 0,000109 | 4,85 |
| 100 | 0,000694 | 0,000222 | 3,12 |

Sàn nhiễu tăng theo `k` gần như tỉ lệ thuận, đúng cơ chế đã nói. Nhưng Reliability đo được nằm
**trên** sàn ở cả ba mức chia, nên phần vượt sàn là độ lệch thật, và nó khớp với độ lệch +0,21 điểm
phần trăm ở bảng đầu. Nếu tỉ lệ đó xấp xỉ 1 thì kết luận phải đổi hẳn: Reliability khi ấy chỉ là
sản phẩm của phép chia nhóm. Dù thế nào, Reliability chỉ so sánh được giữa các model **trên cùng
một k**.

### Hình dạng lệch, và một nửa dự đoán sai

| decile | PD dự báo | bad rate thực | lệch |
|---|---|---|---|
| 0 (an toàn nhất) | 0,88% | 0,40% | +0,48 |
| 4 | 2,27% | 1,64% | +0,63 |
| 8 | 10,86% | 12,44% | −1,59 |
| 9 (rủi ro nhất) | 36,51% | 35,38% | +1,13 |

Dự đoán A1 nói model nén về giữa, tức đầu an toàn dự báo cao hơn thực và đầu rủi ro dự báo
thấp hơn thực, đúng như đã thấy ở train và test của khối 3. Đầu an toàn thì đúng; đầu rủi ro thì
sai chiều. Trên `oot` model dự báo cao hơn ở 8 trên 10 decile, tức dịch lên gần đều, không nén.

---

## 2. Hai cách hiệu chỉnh, và một câu lý thuyết phải sửa

`notes_credit_scoring.md` §5.3 viết: *"Cả ba đều đơn điệu nên không bao giờ đổi Gini/AUC/KS."*
Đó là phép kiểm có đường fail rõ nhất khối này, và nó đã fail một nửa.

| phương án | PD TB | Brier | Gini | lệch Gini | số mức PD | Spearman |
|---|---|---|---|---|---|---|
| gốc | 6,8981% | 0,050593 | 0,712407 | | 10.589 | |
| Platt | 6,9875% | 0,050610 | 0,712407 | 0,000000 | 10.589 | 1,000000 |
| isotonic | 6,9819% | 0,050167 | 0,710655 | −0,001752 | 74 | 0,996762 |

**Platt giữ Gini nguyên vẹn tới chữ số cuối và Spearman đúng bằng 1.** Đúng như lý thuyết.

Một chi tiết cài đặt phải nói vì khối 3 đã trả giá cho đúng lỗi đó: bản đầu của lớp Platt dùng
`C=1e6` với `lbfgs` và `tol` mặc định, và nó dừng ở `|gradient| = 1,65`. Phép bắt được là phép
kiểm MLE, trung bình dự báo phải bằng bad rate trên chính tập fit: bản đó cho 6,68201% so với
6,68444%. Đổi sang `C=inf`, `newton-cholesky`, `tol=1e-12` thì `|gradient|` xuống 4,4e-12 và hai
con số bằng nhau tuyệt đối. Hệ số đổi từ (1,0073; 0,0326) sang (1,0072; 0,0327), đủ nhỏ để không
kết luận nào đổi, nhưng để nguyên thì bài học của khối 3 chỉ nằm trên giấy.

**Isotonic thì không.** Cột "số mức PD" chỉ ra ngay nguyên nhân: nó nén 10.589 giá trị PD phân
biệt xuống còn 74. Isotonic là hàm đơn điệu không nghiêm ngặt, nên nó bảo toàn thứ tự theo
nghĩa yếu: hai người khác điểm trước đó có thể thành bằng điểm sau khi hiệu chỉnh. AUC tính mỗi
cặp hoà là 0,5, nên Gini giảm.

Câu trong ghi chú nền tảng vì thế phải sửa: **Platt không bao giờ đổi thứ hạng; isotonic thì có,
qua đường tạo hoà.** Cái giá ở đây là 0,0018 Gini, nhỏ nhưng có thật và đo được.

Cùng chỗ tạo hoà đó buộc phải kèm một cảnh báo khi đọc phân rã Murphy của isotonic. Bảng decile
chia theo thứ hạng, mà đầu ra isotonic chỉ có 74 mức nên **59,98% số dòng nằm trong một khối hoà
bị biên decile cắt ngang**: hai dòng giống hệt nhau rơi vào hai nhóm khác nhau chỉ vì thứ tự dòng.
Brier không bị ảnh hưởng vì nó tính trên từng dòng, nhưng con số Reliability của riêng isotonic
phải đọc kèm điều này. Đó là lý do bảng trên chỉ so Brier.

### Vì sao Platt làm Brier tệ đi

Kết quả ngược với dự đoán A5: Platt làm Brier tệ đi 0,000017, còn isotonic làm tốt lên 0,000426.

Nguyên nhân mới là chỗ đáng nói. Trên `test` model dự báo thấp hơn bad rate (6,599% so với
6,684%), nên Platt học được phép kéo lên (`a` = 1,0072, `b` = +0,0327). Trên `oot` model đã dự báo
**cao hơn** (6,898%), nên kéo lên nữa là kéo sai chiều.

**Một lớp hiệu chỉnh fit trên kỳ này rồi áp lên kỳ khác chỉ giúp nếu độ lệch giữ nguyên dấu.** Ở
đây nó đổi dấu, và cả hai lần lệch đều chỉ là dao động quanh 0. Hiệu chỉnh một model vốn đã
calibrated chỉ thêm nhiễu.

---

## 3. Bất đối xứng: calibration sửa được sau, xếp hạng thì không

| model | Gini oot | Brier oot | reliability | resolution | PD TB |
|---|---|---|---|---|---|
| scorecard đơn điệu | 0,712407 | 0,050593 | 0,000054 | 0,010365 | 6,898% |
| XGBoost M4 | 0,729663 | 0,049028 | 0,000019 | 0,011375 | 6,876% |

XGBoost trội ở cả hai mặt cùng lúc, đúng như dự đoán A6. Nhưng phân rã cho biết vì sao, và câu
trả lời làm giảm giá trị của Brier ở bộ này: Reliability của hai model đều rất nhỏ (0,000019 so
với 0,000054), tức cả hai calibrate tốt như nhau. **Toàn bộ chênh lệch Brier đến từ Resolution**,
mà Resolution chính là khả năng phân biệt, tức đúng thứ Gini đã đo.

Nói cách khác, ở bộ này Brier không mang thông tin nào mà Gini chưa có. Nó chỉ hữu ích khi
Reliability khác nhau đáng kể giữa hai model, và điều đó chỉ xảy ra khi có một model bị phá
calibration, ví dụ bằng `scale_pos_weight` như khối 4 đã đo.

---

## 4. PSI

### 4.1 Mức nền, và vì sao một con số PSI đơn lẻ không nói lên điều gì

| | PSI |
|---|---|
| train → test | 0,000372 |
| train → oot | 0,001155 |
| xấp xỉ lý thuyết `(k−1)(1/n₁+1/n₂)` | 0,000486 |

Hai tập đều là mẫu ngẫu nhiên từ cùng quần thể mà PSI chênh nhau ba lần, nên câu hỏi đúng không
phải "PSI bằng bao nhiêu" mà **"PSI dao động cỡ nào khi không có drift"**. Cắt ngẫu nhiên chính
train 200 lần, cùng cỡ mẫu với `oot`:

```
trung bình = 0,000510   trung vị = 0,000465   p95 = 0,000956   max = 0,001579
lý thuyết cho đúng cặp cỡ mẫu này (82.499 so với 22.500) = 0,000509
PSI của oot (0,001155) nằm ở phân vị 98%
```

**Trung bình mô phỏng 0,000510 so với lý thuyết 0,000509, lệch 0,3%.** Đây mới là phép so đúng, và
nó đòi phải cẩn thận hai chỗ. Một, xấp xỉ phải tính với đúng cặp cỡ mẫu của chính phép mô phỏng
(82.499 so với 22.500), không phải cặp của dòng `train → oot` ở bảng trên. Hai, xấp xỉ đó xấp xỉ
**kỳ vọng**, mà PSI xấp xỉ một chi-bình phương 9 bậc tự do chia cho cỡ mẫu (Yurdakul 2018) nên
phân phối của nó
lệch phải; trung vị vì thế phải thấp hơn trung bình đúng bằng độ lệch của chi-bình phương, tỉ lệ
lý thuyết 0,927 và đo được 0,911.

Phân phối rộng: từ dưới 0,0002 lên tới 0,0016 chỉ vì lấy mẫu. PSI của `oot` nằm ở phân vị 98%, hơi
cao nhưng vẫn trong dải mà chỉ nhiễu cũng tạo ra được. Một lưu ý về chiều của kết luận: mẫu tham
chiếu trong mô phỏng nhỏ hơn train thật (82.499 so với 104.999) nên mức nền mô phỏng hơi CAO hơn
mức nền thật, tức phân vị 98% là con số thận trọng.

**Một con số PSI đơn lẻ không nói lên điều gì nếu không biết mức nền của chính hệ thống đó.** Ngưỡng 0,1
an toàn ở đây vì nó cách mức nền hai bậc độ lớn, không phải vì nó có cơ sở lý thuyết.

CSI của cả chín biến đều dưới 0,0006, cao nhất là `late_60_89` với 0,000536. Điều này xác nhận
thứ nó sinh ra để xác nhận: ranh giới bin của khối 2 và bảng tra WOE được áp sang `oot` đúng như
đã áp cho train. Biến nào vượt 0,02 sẽ là bug ở bước JOIN, không phải drift, vì `oot` không thể
drift so với train khi nó là lát cắt ngẫu nhiên.

### 4.2 OOT dịch nhân tạo: PSI báo động trong khi model vẫn đúng

`oot` không đo được drift nên phần trên không chứng minh được PSI bắt được drift. Cách kiểm:
lấy mẫu lại `oot` thiên về nhóm `revolving_util` cao với trọng số `w = exp(α · phân vị(util))`,
lấy không hoàn lại 12.000 trên 22.500 dòng. Điều quan trọng nhất về thiết kế: trọng số **chỉ phụ
thuộc X, tuyệt đối không nhìn y**, nên phép lấy mẫu đổi P(X) mà giữ nguyên P(y|X).

| α | PSI | bad rate | PD dự báo | lệch | Gini | util trung vị |
|---|---|---|---|---|---|---|
| 0 | 0,0023 | 6,66% | 6,90% | +0,24 | 0,7134 | 0,159 |
| 2 | 0,0921 | 8,98% | 9,17% | +0,18 | 0,6941 | 0,344 |
| 4 | 0,2660 | 9,98% | 10,51% | +0,53 | 0,6848 | 0,479 |
| 6 | 0,4145 | 10,70% | 11,02% | +0,32 | 0,6595 | 0,521 |

Bảng này xác nhận dự đoán B4 ở cả bốn vế, và nó là kết quả đáng giá nhất khối 5.

**PSI báo động:** ở α = 6 nó lên 0,41, vượt xa ngưỡng điều tra 0,25 và gấp hơn ba trăm lần mức nền.

**Quần thể đúng là rủi ro hơn thật:** bad rate đi từ 6,66% lên 10,70%, gấp 1,6 lần.

**Nhưng model không hề nói dối:** PD dự báo trung bình bám sát bad rate thực ở mọi mức dịch, độ
lệch không bao giờ quá 0,53 điểm phần trăm. Lý do là phép lấy mẫu chỉ đụng P(X), còn calibration
là phát biểu về P(y|X).

**Cái mất là xếp hạng:** Gini rơi từ 0,713 xuống 0,660, vì quần thể mới tập trung vào một vùng hẹp
của biến mạnh nhất nên còn ít thứ để phân biệt.

Kết luận vận hành, vừa là lý do PSI tồn tại vừa là giới hạn của nó: **PSI vượt ngưỡng có nghĩa là
"quần thể đã đổi, hãy đi xem", không có nghĩa là "model đã sai".** Ở đây model vẫn nói đúng PD của
từng người; thứ xuống cấp là khả năng xếp hạng, mà PSI không đo được điều đó. Trong công thức PSI
không có `y` ở bất kỳ đâu.

### 4.3 Cái bẫy bin không đóng băng

Cùng quần thể α = 6 đã chứng minh là dịch mạnh, cùng công thức, ba cách chia bin:

| cách chia | PSI |
|---|---|
| (1) ranh giới đóng băng từ train | 0,414452 |
| (2) mỗi kỳ tự chia decile của mình | 0,000004 |
| (3) lấy ranh giới kỳ mới, áp cho cả hai bên | 0,405777 |

Cách (2) là cái bẫy: khi mỗi bên tự chia decile theo phân phối của mình thì cả tỉ trọng tham chiếu
lẫn tỉ trọng kỳ mới đều bằng 0,10 ở mọi bin **theo đúng cấu tạo**, nên PSI không còn đo được gì.
Con số 0,000004 thấp hơn mức nền của một quần thể không hề dịch chuyển gần ba trăm lần, nên chỉ
số giám sát vẫn nằm yên trong ngưỡng an toàn trong khi quần thể đã dịch chuyển hoàn toàn.

Cách (3) đáng chú ý vì nó vẫn bắt được. Nghĩa là cái bẫy không nằm ở chỗ tính lại ranh giới,
mà ở chỗ **để mỗi bên tự chia theo phân phối của mình**. Chừng nào hai bên còn dùng chung một bộ
ranh giới thì PSI vẫn đo được thứ nó cần đo.

Đây cũng là chỗ tôi cài sai lần đầu: tôi lấy ranh giới của kỳ mới rồi áp cho cả hai bên, tức làm
ra cách (3), và ra 0,41 thay vì 0 như dự đoán B5. Con số không khớp dự đoán là thứ chỉ ra
tôi mô phỏng sai, nên tôi giữ cả ba dòng thay vì xoá dòng sai.

Ranh giới bin là tham số của hệ thống giám sát, và tham số thì đóng băng cùng model. Đó là lý do
`psi()` trong `src/psi.py` bắt buộc nhận `cuts` từ ngoài.

---

## 5. Thống nhất khoảng tin cậy của Gini

Khối 3 dùng ±0,028 cho `oot`, khối 4 dùng ±0,025 cho `test`. Hai tập có đúng cùng số ca dương nên
hai con số không thể khác nhau vì dữ liệu.

| tập | Gini | ca dương | Hanley–McNeil | bootstrap 500 lần |
|---|---|---|---|---|
| test | 0,7006 | 1.504 | ±0,0247 | ±0,0215 |
| oot | 0,7124 | 1.504 | ±0,0243 | ±0,0198 |

**Chốt ±0,025** cho cả dự án, tính bằng Hanley–McNeil, ghi kèm bootstrap ±0,021 làm mốc thứ hai.

Chênh lệch với ±0,028 của khối 3 có nguyên nhân rõ, không phải hai cách tính khác nhau: bảng ở
`notes_credit_scoring.md` §4.9 viết lúc chưa có model nên đặt **AUC giả định 0,78**, và Hanley–McNeil
cho SE nhỏ dần khi AUC lớn dần. Cùng công thức, cùng 1.504 ca dương: ở AUC 0,78 ra ±0,0281, ở AUC
0,856 mà scorecard đạt thật thì ra ±0,0243.

Mọi chênh lệch Gini dưới 0,025 trên một tập giữ riêng ở dự án này là **chưa kết luận được**, và đó
là lý do so sánh scorecard với XGBoost phải làm bằng CV ghép cặp trong train.

---

## Chấm mười bốn dự đoán

| # | dự đoán | kết quả |
|---|---|---|
| A1 | PD TB lệch dưới 0,3 điểm %; hình dạng nén về giữa | trúng một nửa: lệch 0,21 nhưng đầu rủi ro sai chiều |
| A2 | Brier trong 0,048 đến 0,053 | trúng: 0,0506 |
| A3 | XGBoost Brier thấp hơn 0,0005 đến 0,0020 | trúng: thấp hơn 0,00157 |
| A4 | Platt và isotonic không đổi Gini | trượt: isotonic mất 0,0018. Chẩn đoán ghi sẵn ("tạo hoà") thì đúng |
| A5 | Platt cải thiện Brier; isotonic không hơn Platt trên oot | trượt cả hai vế |
| A6 | XGBoost trội scorecard ở cả Gini lẫn Brier | trúng |
| B1 | PSI nền 0,0003 đến 0,0012 | trúng: 0,001155, sát mép trên |
| B2 | CSI chín biến đều dưới 0,01 | trúng: cao nhất 0,000536 |
| B3 | PSI trên OOT dịch trên 0,25, chỉnh vào 0,3 đến 0,6 | trúng: 0,4145 |
| B4 | PSI kêu, bad rate tăng 1,5 đến 3 lần, PD vẫn khớp dưới 0,5 điểm %, Gini giảm 0,03 đến 0,10 | **trúng cả bốn vế** |
| B5 | Bin không đóng băng cho PSI dưới 0,001 | trúng: 0,000004, sau khi sửa cách mô phỏng |
| B6 | Gini oot trong 0,68 đến 0,71 | trượt: 0,7124, cao hơn cận trên |
| D1 | Giả thuyết số 2 của khối 0 sẽ bị chấm là sai | trúng |
| D2 | KTC Gini ra khoảng ±0,025 | trúng: ±0,0243 |

Ba cái trượt đáng nhìn hơn mười cái trúng. A4 và A5 đều nói sai về hiệu chỉnh, và cùng một nguyên
nhân: tôi đọc câu lý thuyết "cả ba đều đơn điệu" như một đảm bảo mà không hỏi đơn điệu nghiêm ngặt
hay không.

B6 trượt vì đoán Gini bi quan, **lần thứ tư liên tiếp** trong dự án, dù lần này tôi đã ghi rõ
trong chính file dự đoán rằng mình biết thói quen đó và đã cố đặt khoảng cao hơn. Biết một thiên
lệch của mình không đủ để sửa nó.

---

## Giới hạn đã biết

- `oot` là lát cắt ngẫu nhiên phân tầng, không phải out-of-time thật. PSI, bad rate và calibration
  trên nó đẹp theo thiết kế. Mọi con số ở mục 4.1 vì thế là mức nền và phép kiểm pipeline,
  không phải bằng chứng model bền với thời gian.
- OOT dịch nhân tạo chỉ dịch P(X) và giữ nguyên P(y|X). Drift thật ngoài đời thường đổi cả hai, và
  loại drift đổi P(y|X) thì PSI không thấy được còn calibration sẽ hỏng thật. Bài kiểm ở đây vì
  vậy chứng minh được PSI bắt được dịch chuyển đầu vào, không chứng minh được nó đủ để giám
  sát.
- Lớp hiệu chỉnh fit trên `test` mà `test` chỉ có 1.504 ca dương, nên chính lớp hiệu chỉnh cũng
  nhiễu. Với isotonic thì đó là nhiễu của 74 bậc thang.
- Phân rã Murphy phụ thuộc số nhóm; mọi con số Reliability và Resolution ở đây chỉ so sánh được
  trong cùng một `k`.
- Ngưỡng PSI 0,1 và 0,25 là quy ước kinh nghiệm. Mục 4.1 cho mức nền của hệ thống này, nhưng không
  cho một ngưỡng có cơ sở thống kê.

## Việc để lại cho khối 6

- Chia bin lại cho `open_credit_lines` rồi cân nhắc đưa nó trở lại scorecard. Khối 4 đã đo được nó
  mang hiệu ứng chính đáng 0,002 đến 0,003 Gini mà dạng hàm của scorecard không dùng được.
- Ba nguyên nhân mất calibration còn lại ở `notes_credit_scoring.md` §5.2 chưa được kiểm bằng số
  trên bộ này: regularization, bản chất thuật toán, dịch chuyển quần thể theo thời gian.

## Nguồn

- Murphy, A. H. (1973), "A New Vector Partition of the Probability Score", *Journal of Applied
  Meteorology*, 12(4), 595–600 — phân rã Brier thành reliability, resolution, uncertainty.
- Platt, J. (1999), "Probabilistic Outputs for Support Vector Machines and Comparisons to
  Regularized Likelihood Methods", *Advances in Large Margin Classifiers*.
- Zadrozny, B. & Elkan, C. (2002), "Transforming Classifier Scores into Accurate Multiclass
  Probability Estimates", *KDD '02* — isotonic regression cho calibration.
- Hanley, J. A. & McNeil, B. J. (1982), "The Meaning and Use of the Area under a ROC Curve",
  *Radiology*, 143(1), 29–36.
- Siddiqi, N. (2017), *Intelligent Credit Scoring*, 2nd ed., Wiley — PSI, CSI và ngưỡng quy ước.
- Yurdakul, B. (2018), *Statistical Properties of Population Stability Index*, luận án tiến sĩ,
  Western Michigan University — PSI xấp xỉ một chi-bình phương chia cho cỡ mẫu, và phân phối của
  nó khi hai mẫu thật sự cùng phân phối. Đây là nguồn cho mục 4.1.
