# Chia bin đơn điệu và đối chiếu scorecard với XGBoost

Kết quả khối 4. Mọi con số dưới đây lấy từ output đã lưu trong
`notebooks/04_monotone_xgboost.ipynb`, trừ bốn nhóm được ghi rõ nguồn tại chỗ: hai chỉ số train KS
lấy từ `python src/scorecard.py` và `python src/scorecard.py --mono`; phép chẩn đoán early
stopping và phép kiểm hội tụ của cây `depth=1` chạy ngoài notebook, trên một máy; và các con số
**trước khi sửa** của bảng ở mục 4, giữ lại để đối chiếu chứ không sinh ra được từ code hiện tại.

Dựng lại từ đầu: `python src/monotone_bins.py` (tạo `bin_map` và bốn bảng `*_mono`), rồi
`python src/scorecard.py --mono`, rồi chạy notebook 04. Cần thêm `pip install xgboost`.

Chín dự đoán được ghi vào `notes/du_doan_khoi4.md` **trước khi chạy phép đo nào**.
Kết quả: **năm trúng, một trúng một phần, hai trượt sát mép, một không kết luận được**. Dự đoán
A2 nêu đích danh sáu biến sẽ "không gộp bin nào hoặc chỉ gộp một cặp"; năm biến đúng, còn
`monthly_income` gộp hai cặp, nên chấm là trúng một phần chứ không phải trúng.

Ngoài ra khối này có **ba kết luận tôi viết ra rồi phải rút lại**, và cả ba đáng ghi hơn bản
thân kết luận đúng: giá trị của `open_credit_lines` nằm ở tương tác (mục 4), mã hoá WOE của khối 2
giết 60% tín hiệu của nó (mục 4), và tự do hình dạng đáng −0,0008 (mục 3). Hai cái đầu cùng một
kiểu lỗi: suy từ một chênh lệch nhiều-thứ-đổi ra kết luận về một thứ. Cái thứ ba là hệ quả của
một lỗi cài đặt mà tôi đã kịp viết một lời giải thích cơ học nghe hợp lý cho nó.

---

## 1. Chia bin đơn điệu

### Vì sao làm

Khối 3 đo được rằng ép đơn điệu đúng chiều tốn dưới 0,002 Gini ở cận trên khoảng tin cậy 95%,
trong khi bảng điểm đơn điệu là thứ làm mã lý do nói được thành câu: "điểm của anh thấp vì tỉ lệ
sử dụng hạn mức cao", không kèm ngoại lệ "trừ khi anh dùng quá ít".

### Chiều là tham số khai báo

Ở bộ này ép sai chiều `revolving_util` làm mất **0,0513 Gini**, gần bằng đúng cái giá của việc
bỏ hẳn biến khỏi model (0,0511). Ép đúng chiều thì `d = +0,00006`, tức không đo được chênh lệch
nào, và cái giá xấu nhất mà khoảng tin cậy còn cho phép là 0,0013. Chiều quyết định gần như toàn
bộ kết quả, còn bản thân phép ép gần như không tốn gì. Bản đầu của
`force_monotone` ở khối 3 tự đoán chiều bằng dấu hiệp phương sai có trọng số, đoán sai ở đúng
biến đang cần phân xử, và cho ra kết luận ngược hoàn toàn.

Nên chiều nằm trong `config.MONOTONE_DIRECTION`, mỗi dòng kèm một câu nghiệp vụ, và
`force_monotone` **bắt buộc** nhận `direction`; truyền tuple thì raise. Bảy biến khai `"giam"`
(rủi ro tăng theo giá trị), `age` và `monthly_income` khai `"tang"`.

### Phân công Python và SQL

Python quyết định **gộp bin nào**: pool-adjacent-violators là thuật toán vòng lặp, viết bằng SQL
không cho ra hơn. SQL **tính lại WOE** trên bin đã gộp, vẫn bằng một `GROUP BY` và một
`CROSS JOIN` với dòng tổng y hệt khối 2, nên không con số WOE nào ra đời ngoài SQL. Cầu nối là
bảng `bin_map`, nhờ vậy cách gộp bin là một bảng nhìn được và kiểm được thay vì nằm ẩn trong một hàm.

Một chi tiết dễ sai: **WOE của bin gộp không bằng trung bình WOE các bin thành phần.** Nó được
tính lại từ số good/bad cộng dồn; giá trị trung bình có trọng số mà PAVA trả về chỉ dùng để xác
định nhóm nào gộp với nhóm nào.

### Kết quả

| biến | chiều | bin | gộp |
|---|---|---|---|
| `debt_ratio_valid` | giảm | 11 → 7 | 02+03+04+05+06 |
| `revolving_util` | giảm | 11 → 9 | 01+02+03 |
| `real_estate_loans` | giảm | 4 → 2 | 0+1+2 |
| `monthly_income` | tăng | 12 → 10 | 01+02, 09+10 |
| `age`, `dependents`, ba biến `late_*` | | không đổi | đã đơn điệu sẵn |

Tổng 70 → 60 bin. Cái giá đo bằng 5-fold CV trong train: `d = −0,00019`, KTC 95%
`[−0,00186; +0,00148]`, tức **cái giá xấu nhất mà dữ liệu còn cho phép là 0,0019 Gini**.

Nhắc lại cách đọc vì rất dễ lấy nhầm đầu: `d = Gini(phương án) − Gini(model đủ)` nên cái giá là
`−d`, và cái giá xấu nhất là **đầu trái** của khoảng. Đầu phải là mức được lợi tối đa.

| | bin gốc | đơn điệu |
|---|---|---|
| số bin | 70 | 60 |
| train Gini / KS | 0,7165 / 0,5576 | 0,7162 / 0,5531 |
| test Gini / KS | 0,6984 / 0,5474 | 0,7006 / 0,5440 |
| tổng điểm | 385–640 | 376–636 |
| lý do số 1 tập trung | 84,7% | 74,3% |

**0,7006 so với 0,6984 không được đọc là cải thiện.** CV nói `d` không phân biệt được với 0, và
chênh lệch 0,0022 nằm sâu trong khoảng tin cậy của chính Gini trên tập test (±0,025; xem mục 7).

---

## 2. Nghịch lý IV

Sau khi gộp, `real_estate_loans` còn 2 bin và IV rơi từ 0,0564 xuống **0,0052**, dưới ngưỡng
0,02 mà mọi giáo trình dùng để loại biến. Nhưng đo bằng CV trên chính bộ bin đó, bỏ nó ra khỏi
model mất **0,00290 Gini** (t = −4,58, năm fold cùng dấu), xếp thứ sáu trên chín biến, trên cả
`debt_ratio_valid` và `monthly_income` là hai biến có IV gấp nó 14 lần.

Hệ số của nó nhảy từ 0,55 lên **1,88**, vượt xa kỳ vọng "quanh 1" mà cả dự án dùng làm quy tắc
kiểm. Đó là con số lạ nên phải kiểm bằng số:

| model | hệ số | z |
|---|---|---|
| `real_estate_loans` một mình | **1,000** | 6,14 |
| + `monthly_income` | 1,758 | 10,54 |
| + `age` | 1,166 | 7,10 |
| + `monthly_income` + `age` | 1,806 | 10,75 |
| đầy đủ 9 biến | 1,877 | 9,66 |

Một mình thì hệ số đúng bằng **1,000**, tức quy tắc "quanh 1" không sai, và nó đúng tới ba chữ
số chứ không phải xấp xỉ. Con số 1,88 là **suppression**, và gần như toàn bộ mức nhảy đến từ
`monthly_income`.

Cơ chế kiểm được: nhóm có từ 3 khoản bất động sản trở lên có thu nhập trung vị **9.032** so với
**5.166** của phần còn lại, tương quan WOE giữa hai biến là **−0,165**. Thu nhập cao che mất rủi
ro đòn bẩy, nên đo đơn biến thì hai hiệu ứng triệt tiêu nhau một phần; đưa thu nhập vào model
rồi thì hiệu ứng đòn bẩy hiện ra gần gấp đôi.

Nhưng câu "nhảy từ 0,55 lên 1,88" so hai con số ở **hai bộ bin khác nhau**, nên chưa phải
suppression thuần. Bảng 2×2 tách được:

| | bin cũ (4 bin) | bin mới (2 bin) |
|---|---|---|
| **đơn biến** | 1,000 | 1,000 |
| **đầy đủ 9 biến** | 0,548 | 1,877 |

Hàng trên bằng nhau **chính xác**: đơn biến thì hệ số đúng bằng 1,000 ở cả hai bộ bin, đúng như
lý thuyết WOE nói, nên gộp bin không tự nó làm hệ số to lên. (Bản đầu của bảng này ghi 1,003 và
1,019; hai số đó là hệ quả của lỗi hội tụ trong `fit_logit`, và sau khi sửa thì hàng trên khít
lại đúng như lý thuyết.) Hàng dưới mới là chỗ có nội dung, và **dấu của hiệu chỉnh đa biến đảo
ngược theo cách chia bin**: trên bin cũ, thêm 8 biến kia vào làm hệ số **co lại** (1,000 →
0,548) vì cột WOE chữ U trùng lặp với biến khác; trên bin mới, thêm đúng 8 biến đó làm hệ số
**nở ra** (1,000 → 1,877) vì contrast còn lại bị `monthly_income` che. Gộp bin bỏ đi
phần trùng lặp, và phần còn lại hoá ra là phần bị che.

**Đây là lý do IV không nhìn thấy: IV là đại lượng đơn biến, nó không có cơ chế nào để thấy một
hiệu ứng bị biến khác che.**

Bảng đóng góp biên xáo trộn mạnh sau khi đổi bin, và chỗ này tôi suýt nói sai. Ba thay đổi lớn
nhất: `revolving_util` từ −0,05107 xuống −0,06177, `debt_ratio_valid` từ −0,00631 lên −0,00218,
`monthly_income` từ −0,00016 xuống −0,00134. **Không được cộng chúng lại** rồi đối chiếu với `d`
tổng −0,00019: đóng góp leave-one-out không cộng được, đúng như chính khối 3 đã chứng minh bằng
số khi tổng Gini đơn biến của mười biến ra 2,41. Cộng thử ra khoảng −0,0105, tức lệch hơn năm
chục lần so với `d` tổng, và đó
là bằng chứng cho tính không cộng được chứ không phải cho một sai sót.

Điều đọc được: **toàn bộ bảng đóng góp biên xáo trộn mạnh trong khi Gini tổng đứng yên.** "Biến
này quan trọng thứ mấy" không phải thuộc tính của biến, nó là thuộc tính của cặp (biến, cách chia
bin) đặt trong một model cụ thể, y hệt như IV.

---

## 3. XGBoost: phần vượt đến từ đâu

Khối 3 ghi trước rằng nếu XGBoost vượt scorecard thì phần vượt chỉ có thể đến từ ba chỗ. Cách
kiểm là dựng ba model cây, **mỗi bước chỉ đổi đúng một thứ**:

| model | đầu vào | bước này thêm gì |
|---|---|---|
| scorecard | 9 cột WOE đơn điệu | mốc so |
| M2 | cùng 9 cột WOE đó | cây thay hàm cộng tính |
| M3 | 9 biến gốc + cờ | cây tự chọn điểm cắt, tức **chỗ cắt bin** |
| M4 | thêm `open_credit_lines`, `debt_ratio` | biến scorecard không dùng |

Cả ba dùng **đúng bộ fold** của `cv_check.make_folds`, nên so theo cặp còn nghĩa: phương sai do
fold (biên độ 0,021 Gini) triệt tiêu khi trừ theo từng fold. Cả ba cũng cùng thắng ở một cấu
hình (`max_depth=3`, `lr=0,10`, 250 vòng), nên ba bước chỉ đổi đầu vào chứ không lẫn "đổi cấu
hình".

**Một thiên lệch cố ý:** siêu tham số của cây chọn bằng chính bộ 5-fold đó, nên con số CV của
cây lạc quan hơn thực tế, còn scorecard không được ưu ái gì. Tôi để nguyên vì nó nghiêng về phía
cây, tức ngược với kết luận tôi chờ đợi.

### Một lỗi cài đặt phải sửa trước khi đọc bất cứ con số nào

Bản đầu của `tree_model.cv_gini` dùng `early_stopping_rounds=30` trên 20% cuối của fold-fit. Cùng
một notebook chạy trên Windows và trên Linux cho kết quả khác nhau **ở chữ số thứ ba**, và **cấu
hình thắng đổi theo nền tảng**: M3 thắng ở `depth=4` bên này, `depth=5` bên kia. XGBoost
`tree_method="hist"` cộng dồn gradient theo thứ tự phụ thuộc số luồng, nên chênh lệch dấu phẩy
động là chuyện bình thường; cái không bình thường là nó bị khuếch đại lên tới chữ số thứ ba. Chẩn đoán, trên cấu hình
`depth=5, lr=0,05`:

| | kết quả |
|---|---|
| có early stopping | `best_iteration` mỗi fold = 31, 75, **8**, 98, 156; đổi seed → biên độ **0,00458** |
| số vòng cố định | đổi seed → biên độ **0,00036** |

Một fold dừng ở vòng **8** với `learning_rate = 0,05` là một model chưa học gì. Eval set chỉ
khoảng 16.800 dòng với ~1.120 ca dương nên AUC trên nó rất nhiễu, và 30 vòng quá ngắn để phân
biệt một đỉnh giả với một đỉnh thật. Early stopping ở đây **không chống overfit mà biến nhiễu của
eval set thành nhiễu của kết quả, gấp 13 lần.**

Bỏ nó đi được thêm hai thứ. Cây fit trên **trọn 80%** mỗi fold, đúng bằng scorecard, nên mất một
thiên lệch mà trước đó chỉ khai được chứ không tách được. Và bảng test dùng đúng số vòng của bảng
CV, nên hai bảng đọc chéo nhau được. Cái giá: số vòng thành một siêu tham số nữa, khai trước theo
quy tắc `learning_rate × n_estimators ≈ 25` chứ không dò tìm.

Mọi con số dưới đây là của phiên bản đã sửa. Ghi lại cả chuyện này vì nó đã sinh ra ba kết luận
sai mà tôi viết ra rồi phải rút.

### Bước scorecard → M2 đổi hai thứ, không phải một

Scorecard có đúng **một hệ số cho mỗi biến**, tức hình dạng của biến bị khoá ở giá trị WOE đơn
biến. Cây trên cùng các cột WOE đó gán được một giá trị riêng cho **từng bin**, tức thêm khoảng
51 tham số hình dạng. Vậy phần chênh giữa hai bên có thể là tương tác, có thể là tự do hình dạng,
hoặc cả hai. Tách bằng cây `max_depth=1`, vốn là model **cộng tính có hình dạng tự do**:

| | Gini CV | |
|---|---|---|
| scorecard (logistic trên cột WOE) | 0,71507 | |
| XGB `depth=1` trên đúng cột WOE đó | 0,71559 | tự do hình dạng = **+0,0005**, t = 1,18, KTC [−0,0007; +0,0017] |
| XGB `depth=4` trên đúng cột WOE đó | 0,72466 | tương tác = **+0,0091**, t = 9,9 |

Tự do hình dạng **không phân biệt được với 0**; tương tác gấp 17 lần nó. Lý do cơ học và đáng
nhớ: **cột WOE đã là log-odds thực nghiệm của từng bin**, nên model tuyến tính theo WOE vốn đã
mang sẵn hình dạng cộng tính tối ưu, sai khác đúng một hệ số co giãn cho mỗi biến; một cây
`depth=1` học lại hình dạng đó từ đầu không có gì để thêm. Đây là một lập luận bênh vực cách mã
hoá WOE mà tôi không có trước phép kiểm này.

Số vòng của cây `depth=1` để 1200 và đã kiểm hội tụ: 600 vòng cho 0,71589, 1200 cho 0,71562,
2000 cho 0,71569.

### Mức nhiễu

Đổi `random_state` mà giữ nguyên bộ fold và siêu tham số, Gini CV của M3 dao động với **độ lệch
chuẩn 0,00043 và biên độ 0,00110**.

Từ đó tôi lấy **0,002** làm ngưỡng đọc được. Con số này **không suy ra được** từ 0,00110; nó là
biên độ làm tròn lên gần gấp đôi, chọn cho an toàn. Nói vậy cho đúng thay vì viết "vậy nên".

Phép đo nhiễu này chạy tại đúng cấu hình đã thắng vòng chọn siêu tham số, nên nó đo nhiễu của
**một lần fit** chứ không đo nhiễu của cả quy trình "chọn rồi báo cáo". Khi ngưỡng 0,002 và kiểm
định `t` nói ngược nhau thì **không đọc con số đó**.

### Bảng phân rã

| model | Gini CV | tăng so với dòng trên | t | nguồn |
|---|---|---|---|---|
| scorecard (logistic, WOE đơn điệu) | 0,71507 | | | |
| XGB `depth=1` trên cùng cột WOE | 0,71559 | +0,0005 | 1,18 | tự do hình dạng, không đọc được |
| M2 XGB `depth=3` trên cùng cột WOE | 0,72492 | **+0,0099** | 13,0 | **tương tác** |
| M3 XGB trên biến gốc | 0,72935 | +0,0044 | 10,0 | chỗ cắt bin |
| M4 XGB trên tất cả | 0,73280 | +0,0035 | 8,26 | biến scorecard không dùng |

Dòng M2 so với dòng ngay trên nó là +0,0099; phần thuần tương tác, tức so với cây `depth=4` trên
cùng cột WOE, là +0,0091. Hai con số chênh nhau đúng phần tự do hình dạng không đọc được.

**So sánh chính, trên cùng 9 biến: XGBoost hơn scorecard 0,0143 Gini** (t = 19,2, KTC
[+0,0122; +0,0163]), gấp mười mấy lần mức nhiễu 0,0011, nên đây là kết luận chắc.

### Cái gì ổn định và cái gì không

**Ổn định:**

- XGBoost hơn scorecard **0,0143** trên cùng 9 biến.
- **Tương tác là kênh lớn nhất, +0,0091**; tự do hình dạng không đọc được.
- Chỗ cắt bin và biến bị loại mỗi kênh đáng khoảng **+0,004**, cả hai đều đọc được.

**Ít ổn định hơn:** *thứ tự* giữa hai kênh sau. Cố định một bộ siêu tham số cho cả ba model:

| cấu hình | chỗ cắt bin (M2→M3) | biến bị loại (M3→M4) |
|---|---|---|
| `max_depth=4`, `lr=0,10`, 250 vòng | +0,00349 (t = 6,18) | +0,00439 (t = 8,99) |
| `max_depth=5`, `lr=0,05`, 500 vòng | +0,00383 (t = 5,52) | +0,00409 (t = 10,31) |

Bốn con số nằm trong khoảng hẹp 0,0035 đến 0,0044 và cả bốn đều có `t` lớn, nên **độ lớn của hai
kênh là kết luận đọc được**. Cái không đọc được chỉ là kênh nào lớn hơn: hai kênh gần bằng nhau
và chênh lệch giữa chúng (0,0009 và 0,0003) nhỏ hơn mức phân biệt được.

Câu đúng là: **phần vượt của cây chủ yếu là tương tác (+0,0091), phần còn lại chia gần đều cho
chỗ cắt bin và biến bị loại, mỗi kênh khoảng +0,004.**

Bản trước của mục này kết luận rằng tỉ lệ giữa hai kênh "là thuộc tính của cặp (dữ liệu, sức chứa
của model)", với quan sát rằng cây nông thì phần lợi rơi vào chỗ cắt bin còn cây sâu thì rơi vào
biến được thêm. **Kết luận đó phải rút lại:** nó dựa trên bảng chạy với early stopping, và sau khi
sửa thì hai kênh gần bằng nhau ở cả hai cấu hình. Cái đảo thứ tự quan sát được trước đây là nhiễu
của early stopping chứ không phải một quy luật.

Ở `results/scorecard.md` tôi viết "nếu chênh lệch chỗ cắt bin lớn hơn chênh lệch tương tác thì
kết luận là chỗ cắt bin quan trọng hơn"; nó không lớn hơn ở bất kỳ cấu hình nào (0,004 so với
0,0091), nên kết luận đó vẫn đứng. Dự đoán B1 và B2 đều **trúng**, và tổng phần vượt nằm xa dưới
ngưỡng 0,04 tôi đặt làm mốc nghi cài sai hoặc leakage.

---

## 4. Câu hỏi riêng: `open_credit_lines`

M4 thêm hai biến cùng lúc nên phải tách:

| phương án | d | t | 5 fold cùng dấu |
|---|---|---|---|
| M3 + `open_credit_lines` | **+0,00303** | 7,41 | có |
| M3 + `debt_ratio` thô | +0,00007 | 0,20 | không |
| M3 + cả hai | +0,00346 | 8,26 | có |

Toàn bộ phần tăng của M4 là của `open_credit_lines`; `debt_ratio` thô không phân biệt được với 0.

### Hai kết luận tôi rút ra ở đây rồi phải rút lại

**Lần một.** Tôi viết "toàn bộ giá trị của biến này nằm ở tương tác", lấy lý do nó đáng +0,00004
với model cộng tính ở khối 3 và +0,0030 với cây ở đây. Nhưng hai con số đó khác nhau ở **ba** thứ
cùng lúc: lớp hàm, dạng biến, và mốc so. Chỉ thứ nhất là tương tác.

**Lần hai.** Sửa bằng cây `depth=1` (khi đó còn early stopping) thì thấy biến gốc đáng +0,0037 còn cột WOE của khối 2 chỉ
+0,0014, nên tôi kết luận "mã hoá WOE của khối 2 giết 60% tín hiệu". Sau khi sửa early stopping,
hai con số đó thành +0,0029 và +0,0023, tức **chênh lệch nằm trong nhiễu**. Kết luận "bin xấu"
cũng phải rút.

### Phép kiểm chỉ đổi một thứ

Cùng bộ bin của khối 2, cùng logistic, cùng bộ fold, chỉ đổi **số tham số** dành cho biến đó.

Bảng này ban đầu sinh ra từ một cell gọi `LogisticRegression(C=1e12, max_iter=2000)`, tức đúng
cấu hình không hội tụ đã làm hỏng `fit_logit` ở khối 3. Đo lại bằng solver đã sửa: dòng chín
dummy gần như không đổi (+0,00207 thành +0,00206) nhưng t tăng từ 3,88 lên 4,07, còn dòng một hệ
số tụt từ +0,00024 (t = 2,36) xuống +0,00016 (t = 1,82). Kết luận không đổi, nhưng bản không hội
tụ đã đẩy dòng một hệ số lên sát mức phân biệt được với 0, tức lệch về phía làm kết luận yếu đi.

| cách đưa `open_credit_lines` vào model | đóng góp | t |
|---|---|---|
| biến gốc, cộng tính tự chọn hình dạng (`depth=1`) | +0,0029 | 3,48 |
| cột WOE khối 2, cộng tính (`depth=1`) | +0,0023 | 4,37 |
| cột WOE khối 2, logistic với **9 cột dummy** | +0,0021 | 4,07 |
| cột WOE khối 2, logistic với **1 hệ số**, cùng phép đo với dòng trên | **+0,00016** | 1,82 |
| cột WOE khối 2, logistic với **1 hệ số**, đo bằng đóng góp biên ở khối 3 | **+0,00004** | 1,14 |

Hai dòng cuối là cùng một mô hình đo theo hai đường: dòng áp chót dùng đúng baseline và đúng bộ
fold của dòng chín dummy nên nó mới là phép so chỉ đổi một thứ, dòng chót là con số khối 3 đã
dùng để loại biến. Hai đường cho cùng một kết luận, và đó là điều đáng ghi.

Ba dòng đầu nói cùng một chuyện qua ba đường khác nhau; hai dòng cuối rơi xuống gần 0. Thủ phạm
**không phải cách chia bin và không phải thiếu tương tác**, mà là **ràng buộc một hệ số cho cả
cột WOE**, tức chính dạng hàm của scorecard cổ điển. Cột WOE của biến này có hình chữ U, mà khối
3 đã chứng minh nhánh trái chỉ là bóng của `revolving_util`; nhân cả cột với một hằng số thì phần
còn có ích không tách ra được.

Kết luận: **quyết định loại `open_credit_lines` ở khối 3 đúng với dạng hàm của scorecard**, và
cách sửa ở khối 5 là chia bin lại sao cho cột WOE đơn điệu (khi đó một hệ số là đủ), hoặc cho
biến này nhiều hơn một tham số. Điểm mù của quy trình "chia bin một lần rồi chọn biến" vẫn đứng,
phát biểu lại cho đúng: **một biến mà scorecard không dùng được chưa chắc là một biến vô dụng; có
thể chỉ là scorecard không đủ tham số cho nó.**

Ranh giới giữa hai lớp hàm không nằm ở "có tương tác hay không" như tôi tưởng lần đầu, cũng không
ở "bin tốt hay xấu" như tôi tưởng lần hai, mà ở **số tham số mà model cho phép biến đó có**.

Vì lý do đó so sánh chính ở mục 3 dùng M3 chứ không dùng M4: hai bên phải cùng bộ biến, nếu không
thì phần vượt đó bị gán nhầm cho "cây mạnh hơn" trong khi thực ra là "cây được cho thêm một biến".

---

## 5. `scale_pos_weight`

Giả thuyết số 1 ghi ở `notes_credit_scoring.md` §10 từ khối 0: bật và tắt chênh nhau dưới 0,01
Gini. **Đúng**: chênh lệch là **−0,00080** (t = −1,44, KTC [−0,0023; +0,0007]), xa dưới ngưỡng.

Nhưng calibration vỡ hẳn: PD dự báo trung bình nhảy từ **6,685%** lên **31,981%**, tức **4,78
lần** bad rate thực. Hệ số cân bằng lớp đổi độ nhạy lấy một phép dịch intercept, lượng dịch liên
quan tới `−ln(w) = 2,64`, và mọi con số PD mất nghĩa theo. (Mức dịch quan sát được nhỏ hơn
một phép dịch logit thuần vì `base_score` và cấu trúc cây hấp thụ một phần; không khẳng định con
số chính xác vì chưa tách được hai phần đó.)

Kết luận triển khai: **không bật `scale_pos_weight`** cho bài toán này. Thứ tự xếp hạng không
tốt lên, còn PD thì hỏng, mà PD mới là thứ scorecard tồn tại để cung cấp.

---

## 6. SHAP, và một dự đoán đặt sai

Giá trị SHAP interaction lấy thẳng từ XGBoost (`pred_interactions=True`) chứ không qua gói
`shap`, vì `shap` 0.49 chưa đọc được `base_score` dạng mảng của XGBoost 3.x. Tính trên mẫu 4.000
dòng train.

Dự đoán B4 (cặp mạnh nhất là `revolving_util` với một trong ba biến `late_*`) **không kết luận
được**, và lý do đáng ghi hơn bản thân dự đoán. Ba lần chạy khác nhau cho **ba cặp đứng đầu khác
nhau**: `revolving_util` với `debt_ratio_valid`, rồi với `late_30_59`, rồi với `age`. Ở lần chạy
được lưu, bốn cặp đầu bảng nằm trong khoảng hẹp 0,039 đến 0,048, tức cách nhau ít hơn mức phân
biệt được.

**Một dự đoán mà phán quyết đổi theo hạt giống ngẫu nhiên thì không phải một dự đoán đặt đúng.**
Cách đặt đúng lẽ ra là "cặp `revolving_util × late_*` nằm trong ba cặp mạnh nhất", một mệnh đề đủ
thô để nhiễu không lật được, và mệnh đề đó đúng ở cả ba lần chạy. Đây là lỗi thiết kế dự đoán chứ
không phải lỗi đo.

Hai điều ổn định hơn và đáng chú ý hơn:

`open_credit_lines` là **đóng góp chính mạnh thứ sáu** trong bảng SHAP ở cả ba lần chạy, và có
mặt ở hai đến ba trong mười cặp tương tác mạnh nhất, dù đóng góp của nó trong scorecard bằng
không. SHAP và CV nói cùng một chuyện qua hai đường khác nhau.

Tương tác chiếm **28,7%** tổng |SHAP| nhưng kênh tương tác chỉ quy ra +0,0091 Gini. Hai con số
không mâu thuẫn, và phải nói kèm hai điều. Thứ nhất chúng **của hai model khác nhau**: 28,7% đo
trên M4 (biến thô, thêm hai biến), còn +0,0091 đo trên M2 (cột WOE). Thứ hai, ngay cả trong cùng
một model thì phần lớn tương tác là tinh chỉnh quanh một hiệu ứng chính rất mạnh chứ không tạo ra
thứ tự xếp hạng mới.

---

## 7. Tập test, dùng đúng một lần

| model | train Gini | test Gini | test KS | chênh train-test |
|---|---|---|---|---|
| scorecard đơn điệu | 0,7162 | 0,7006 | 0,5440 | 0,0155 |
| M2 XGB trên cột WOE | 0,7332 | 0,7139 | 0,5732 | 0,0193 |
| M3 XGB trên biến gốc | 0,7455 | 0,7137 | 0,5692 | 0,0317 |
| M4 XGB trên tất cả | 0,7500 | 0,7208 | 0,5752 | 0,0292 |

Scorecard có khoảng cách train trừ test hẹp nhất; ba model cây rộng hơn, đúng kiểu của model nhiều
tham số hơn. Bảng này giờ **đọc chéo được với bảng CV** vì cả hai dùng cùng cấu hình và
cùng số vòng, điều mà bản trước không làm được.

**Thứ tự giữa ba model cây thì không đọc được**, và ở đây có một ví dụ trực tiếp: CV nói M3
(0,72935) hơn M2 (0,72492) một cách rõ ràng (t = 10,0), còn trên test thì hai model gần như bằng
nhau và M2 nhỉnh hơn (0,7139 so với 0,7137). Chênh lệch đó nằm sâu trong khoảng tin cậy của chính
Gini trên test.

Con số nên tin vẫn là CV chứ không phải test: test chỉ có 1.504 ca dương nên khoảng tin cậy 95%
của Gini theo Hanley–McNeil là **±0,025**, bootstrap 300 lần cho ±0,021, cả hai rộng hơn toàn bộ
chênh lệch 0,020 giữa scorecard và model cây tốt nhất. (Con số ±0,028 dùng ở khối 3 là cho tập
OOT với cùng số ca dương nhưng tính bảo thủ hơn; hai chỗ nên thống nhất ở khối 5.)

PD dự báo trung bình của cả bốn model đều quanh 6,55 đến 6,60% so với bad rate thực 6,684%.

---

## Một hệ quả của đơn điệu chưa nói ở mục 1

Lý do duy nhất để ép đơn điệu là để giải thích được. Với `revolving_util` thì đúng. Với
`real_estate_loans` thì nó **tạo ra** một vấn đề giải thích mới:

| nhóm | bad rate thô | điểm (bin gốc) | điểm (đơn điệu) |
|---|---|---|---|
| 0 khoản bất động sản | 8,33% | 59 | 64, cao nhất biến |
| 2 khoản | 5,60% | 65 | 64, cùng nhóm |
| 3+ khoản | 8,46% | 58 | 49 |

Bảng điểm cũ chấm nhóm 8,33% và nhóm 8,46% gần bằng nhau (59 và 58), đúng như bad rate thô của
họ. Bảng điểm "giải thích được" mới tách hai nhóm gần như cùng rủi ro thô ra **15 điểm**, và cho
nhóm 0 khoản điểm tối đa.

Về thống kê thì bảo vệ được: khối 3 đã đo rằng rủi ro thừa của nhóm 0 khoản đã nằm trong các biến
khác. Nhưng đó là câu trả lời **đa biến**, và nó không nói được với một khách hàng bị từ chối,
trong khi giải thích được cho khách hàng chính là lý do cả bước đơn điệu hoá tồn tại. Ghi lại làm
giới hạn đã biết.

---

## Giới hạn đã biết

- Siêu tham số của cây chọn bằng chính bộ CV dùng để báo cáo, nên con số CV của cây lạc quan.
  Thiên lệch này nghiêng về phía cây, tức ngược với kết luận "cây chỉ hơn 0,0143".
- Cấu hình thắng hơn cấu hình nhì 0,00026 ở M2, 0,00121 ở M3, 0,00027 ở M4, tức cùng bậc với mức
  nhiễu 0,00110. Cấu hình thắng vì thế gần như tuỳ ý; điều cứu phép so sánh là cả ba model cùng
  thắng ở một cấu hình và các cấu hình đứng đầu cho gần cùng một con số.
- Số vòng boosting khai trước theo quy tắc `lr × n_estimators ≈ 25` chứ không dò tìm, nên không
  tối ưu cho từng cấu hình. Đây là cái giá phải trả để bỏ early stopping.
- Kết quả XGBoost vẫn không tái lập bit-wise giữa hai nền tảng; sau khi bỏ early stopping thì
  phần dư đó nhỏ hơn nhiều, nhưng mọi con số cây vẫn chỉ nêu tới chữ số chịu được.
- Bin của scorecard (cắt `NTILE` và nhóm PAVA) học trên toàn bộ train rồi dùng lại trong mọi fold,
  trong khi cây học mọi thứ trong fold. Riêng nhóm PAVA được sinh ra **từ WOE, tức từ nhãn**, nên
  lý lẽ đã dùng ở khối 3 để tha cho điểm cắt `NTILE` (phân vị của biến, không nhìn nhãn) không mở
  rộng sang đây được. Thiên lệch này có lợi cho scorecard, tức ngược chiều với thiên lệch đầu tiên.
- "Cùng bộ fold" giữa scorecard và cây dựa vào giả định hai bảng có cùng thứ tự dòng theo `id`.
  Notebook có một `assert` cho chuyện này; trước khi thêm nó thì đây là một giả định không được
  kiểm.
- Điểm cắt bin vẫn không được tính lại trong từng fold, chỉ WOE mới được.
- Mã lý do vẫn tập trung 74,3% vào một biến duy nhất.
- Tập `oot` vẫn chưa được đọc nhãn ở bất kỳ phép đo nào.

## Việc để lại cho khối 5

- Calibration: Platt và isotonic cho cả scorecard lẫn XGBoost, so bằng Brier và reliability diagram.
- PSI và CSI trên tập OOT, và trên một OOT dịch nhân tạo để PSI có cái để bắt.
- Chia bin lại cho `open_credit_lines` rồi cân nhắc đưa nó trở lại scorecard.
- Thống nhất cách tính khoảng tin cậy của Gini giữa khối 3 (±0,028) và khối 4 (±0,025 / ±0,021).
- Ba giả thuyết còn lại ở `notes_credit_scoring.md` §10.

---

## Nguồn

- Barlow, R. E. và cộng sự (1972), *Statistical Inference Under Order Restrictions*, Wiley.
- Siddiqi, N. (2017), *Intelligent Credit Scoring*, 2nd ed., Wiley — chuẩn mực đơn điệu trong scorecard.
- Chen, T. & Guestrin, C. (2016), "XGBoost: A Scalable Tree Boosting System", *KDD '16*, 785–794.
- Lundberg, S. M., Erion, G. & Lee, S.-I. (2018), "Consistent Individualized Feature Attribution
  for Tree Ensembles", arXiv:1802.03888 — SHAP interaction values cho model cây.
- Dietterich, T. G. (1998), "Approximate statistical tests for comparing supervised classification
  learning algorithms", *Neural Computation*, 10(7), 1895–1923.
