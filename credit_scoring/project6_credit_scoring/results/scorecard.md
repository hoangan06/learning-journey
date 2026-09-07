# Báo cáo scorecard

Hồi quy logistic trên các cột WOE của khối 2, quy đổi ra thang điểm và mã lý do.
Hệ số ước lượng trên `split = 'train'` (104.999 dòng, 7.018 bad, bad rate 6,6839%).
Mọi quyết định chọn biến ra đời bằng 5-fold CV **bên trong train**. `test` không tham gia
quyết định nào, chỉ để đối chiếu dự đoán đã ghi trước và để báo cáo Gini, calibration, ngưỡng
cắt. `oot` chưa động tới.

Chạy lại: `python src/scorecard.py`, chi tiết trong `notebooks/03_scorecard.ipynb`.

---

## 1. Model đầy đủ 10 biến

Model log-odds của **good** (`y = 1 - target`), không regularization (`C=1e12`).
WOE định nghĩa là `ln(pct_good / pct_bad)` nên WOE cao là bin an toàn, và quy ước này cho
một phép kiểm một dòng: mọi hệ số phải dương. Sai số chuẩn tính tay từ Hessian,
`cov = (Xᵀ W X)⁻¹` với `W = diag(p(1−p))`, vì scikit-learn không trả về.

|                   |   hệ số |     SE |        z |
|:------------------|--------:|-------:|---------:|
| (intercept)       |  2.6053 | 0.0152 | 171.8246 |
| age               |  0.3784 | 0.0319 |  11.8617 |
| debt_ratio_valid  |  0.7891 | 0.0531 |  14.8491 |
| dependents        |  0.2577 | 0.0758 |   3.4022 |
| late_30_59        |  0.5201 | 0.0155 |  33.4908 |
| late_60_89        |  0.3676 |  0.017 |  21.5634 |
| late_90           |  0.5158 | 0.0139 |  37.1745 |
| monthly_income    |  0.0947 | 0.0543 |   1.7432 |
| open_credit_lines | -0.0462 | 0.0584 |  -0.7911 |
| real_estate_loans |  0.5578 | 0.0643 |   8.6698 |
| revolving_util    |   0.627 | 0.0151 |  41.4085 |

| | Gini | KS |
|---|---|---|
| train | 0.7165 | 0.5571 |
| test | 0.6985 | 0.5484 |

Hệ số quanh 1 là kỳ vọng có lý do: WOE đã ở đơn vị log-odds, nên một biến là nguồn thông tin
duy nhất sẽ có hệ số đúng bằng 1. Nhỏ hơn 1 nghĩa là thông tin đó đã có phần nằm trong biến
khác nên model chiết khấu bớt.

Hai chỗ cần dừng lại.

`open_credit_lines` có hệ số **âm** (-0,0462), z = -0,79. Âm trong quy ước này nghĩa là model
đảo ngược thứ tự WOE của biến: bin mà phân tích đơn biến gọi là an toàn thì trong bối cảnh
đa biến lại xấu hơn. Nhưng z nhỏ hơn 1 nên hệ số không phân biệt được với 0, và câu đúng là
model **không tìm thấy gì** ở biến này sau khi đã có chín biến kia. Cần đo đóng góp biên chứ
không đọc dấu.

Gini test 0,6985 nằm ngoài khoảng 0,60 đến 0,68 tôi đoán trước khi fit, cao hơn cận
trên. Đoán sai vì bi quan: tôi trừ hao phần chồng lấn giữa các biến nhiều hơn thực tế. Vẫn
dưới ngưỡng báo động 0,72 nên không đổi kết luận về leakage. Con số này là kỳ vọng tôi đặt lúc
mở khối 3 chứ không nằm trong bảng giả thuyết ở `notes_credit_scoring.md` §10, nên nó không có
cùng sức nặng với hai dự đoán ghi trong `results/iv_report.md`.

---

## 2. Chọn biến

Đo bằng 5-fold CV trong train, **WOE tính lại trong từng fold** (`src/cv_check.py`). Nếu dùng
sẵn cột WOE trong `features_woe`, vốn tính trên toàn bộ train, thì fold đánh giá đã góp phần
tạo ra feature của chính nó và chênh lệch đo được sẽ lạc quan.

So sánh **theo cặp trên cùng fold**: Gini giữa các fold dao động 0,7004 đến 0,7215, biên độ
khoảng 0,021, lớn gấp nhiều lần mọi chênh lệch cần đo. Trừ theo từng fold thì phần phương sai
do fold triệt tiêu.

Một ngoại lệ phải nói ra: **điểm cắt bin thì không tính lại trong fold**, chúng lấy sẵn từ
`bin_cuts` của khối 2, vốn tính trên toàn bộ train. Điểm cắt cũng là tham số học được nên lập
luận ở trên áp dụng y nguyên cho chúng. Tôi để vậy vì điểm cắt là phân vị của biến, không nhìn
vào nhãn, và phân vị rất ổn định khi bỏ đi 20% dữ liệu; nhưng đó là lập luận về mức ảnh hưởng
chứ không phải lời bào chữa, và phép đo dưới đây lạc quan hơn đúng một chút vì lý do này.

Gini CV của model đủ: **0,71513**.

### Đóng góp biên của từng biến

Cột "Gini đơn biến" là model chỉ có một biến đó; "Gini khi bỏ" là model thiếu nó; `d` là hiệu
so với model đủ, nên âm nhiều là quan trọng. Hai cột đo trên cùng năm fold nên so được trực tiếp.

| biến             |   Gini đơn biến |   Gini khi bỏ |        d |      se |         t |   ktc_lo |   ktc_hi | 5 fold cùng dấu   |
|:------------------|----------------:|--------------:|---------:|--------:|----------:|---------:|---------:|:------------------|
| revolving_util    |         0.56001 |       0.66406 | -0.05107 | 0.00178 | -28.7422  | -0.056   | -0.04614 | True              |
| late_30_59        |         0.38181 |       0.68307 | -0.03206 | 0.00194 | -16.5432  | -0.03744 | -0.02668 | True              |
| late_90           |         0.31247 |       0.68827 | -0.02686 | 0.00178 | -15.0876  | -0.0318  | -0.02192 | True              |
| late_60_89        |         0.24133 |       0.70283 | -0.0123  | 0.00127 |  -9.71849 | -0.01581 | -0.00879 | True              |
| debt_ratio_valid  |         0.14836 |       0.70882 | -0.00631 | 0.00141 |  -4.48936 | -0.01021 | -0.00241 | True              |
| age               |         0.26733 |       0.71152 | -0.0036  | 0.00074 |  -4.88669 | -0.00565 | -0.00156 | True              |
| real_estate_loans |         0.11936 |       0.71213 | -0.003   | 0.0009  |  -3.34144 | -0.00549 | -0.00051 | True              |
| monthly_income    |         0.15349 |       0.71497 | -0.00016 | 5e-05   |  -3.43585 | -0.00029 | -3e-05   | True              |
| dependents        |         0.09925 |       0.71506 | -6e-05   | 0.00023 |  -0.27604 | -0.0007  |  0.00058 | False             |
| open_credit_lines |         0.12249 |       0.71518 |  6e-05   | 4e-05   |   1.46102 | -5e-05   |  0.00017 | False             |

Đặt cạnh nhau thì hai cột nói hai chuyện khác nhau, và đây là bài học chính của cả khối:

| biến | Gini đơn biến | đóng góp biên |
|---|---|---|
| `monthly_income` | 0,15349 | −0,00016 |
| `debt_ratio_valid` | 0,14836 | −0,00631 |
| `open_credit_lines` | 0,12249 | +0,00006 |
| `real_estate_loans` | 0,11936 | −0,00300 |
| `dependents` | 0,09925 | −0,00006 |

`monthly_income` mạnh nhất nhóm này khi đứng một mình nhưng đóng góp gần bằng không, còn
`real_estate_loans` gần yếu nhất lại đóng góp gấp gần 20 lần. Sức mạnh đơn biến đo *biến đó
biết gì*; đóng góp biên đo *biến đó biết gì mà chín biến kia không biết*. Chọn biến bằng bảng
IV là dùng đại lượng thứ nhất để trả lời câu hỏi thứ hai.

Tổng Gini đơn biến của mười biến là 2,41, gấp hơn ba lần Gini của model đủ. Gini không
cộng được, và bảng này là cách nhìn thấy điều đó bằng số.

### Ép đơn điệu: đo cả hai chiều

"Ép đơn điệu" **không phải một phương án duy nhất**. PAVA cần một chiều, và với biến hình chữ U
thì hai chiều gộp phẳng hai nhánh khác nhau. WOE ở đây định hướng theo good, nên "giảm" nghĩa
là rủi ro tăng dần theo bin. Cột "mức còn lại" cho biết sau khi ép thì còn mấy mức WOE phân
biệt trên tổng số bin thường; nó quan trọng ngang cột `d`, vì nó là cái giá phải trả về mặt
cấu trúc.

| biến              | chiều   | mức còn lại   |   Gini CV |        d |      se |         t |   ktc_lo |   ktc_hi | 5 fold cùng dấu   |
|:------------------|:--------|:--------------|----------:|---------:|--------:|----------:|---------:|---------:|:------------------|
| revolving_util    | tang    | 1/10          |   0.6638  | -0.05133 | 0.00178 | -28.8042  | -0.05628 | -0.04639 | True              |
| revolving_util    | giam    | 8/10          |   0.7152  |  6e-05   | 0.0005  |   0.1276  | -0.00133 |  0.00146 | False             |
| open_credit_lines | tang    | 4/10          |   0.71535 |  0.00022 | 0.00014 |   1.58924 | -0.00016 |  0.0006  | False             |
| open_credit_lines | giam    | 2/10          |   0.71507 | -6e-05   | 0.00013 |  -0.45649 | -0.00042 |  0.0003  | False             |
| real_estate_loans | tang    | 2/4           |   0.71272 | -0.00241 | 0.00062 |  -3.91617 | -0.00412 | -0.0007  | True              |
| real_estate_loans | giam    | 2/4           |   0.71519 |  6e-05   | 0.00058 |   0.10414 | -0.00154 |  0.00166 | False             |
| debt_ratio_valid  | tang    | 1/10          |   0.70885 | -0.00628 | 0.00141 |  -4.45348 | -0.01019 | -0.00237 | True              |
| debt_ratio_valid  | giam    | 6/10          |   0.71598 |  0.00085 | 0.00049 |   1.72682 | -0.00052 |  0.00222 | False             |

### Các phép gộp bin và rủi ro leakage

| phương án                     |   Gini CV |        d |      se |         t |   ktc_lo |   ktc_hi | 5 fold cùng dấu   |
|:------------------------------|----------:|---------:|--------:|----------:|---------:|---------:|:------------------|
| gộp bin 01 của revolving_util |   0.71525 |  0.00011 | 0.00048 |   0.23916 | -0.00122 |  0.00145 | False             |
| bỏ open_credit_lines          |   0.71517 |  4e-05   | 4e-05   |   1.13826 | -6e-05   |  0.00015 | False             |
| real_estate: gộp 0 vào 1      |   0.71339 | -0.00174 | 0.00058 |  -3.00856 | -0.00335 | -0.00013 | False             |
| real_estate: gộp 3+ vào 2     |   0.71357 | -0.00156 | 0.00044 |  -3.56973 | -0.00277 | -0.00035 | True              |
| real_estate: gộp 0,1,2        |   0.7152  |  7e-05   | 0.00058 |   0.11301 | -0.00154 |  0.00167 | False             |
| real_estate: gộp hết (1 bin)  |   0.71213 | -0.003   | 0.00089 |  -3.36639 | -0.00547 | -0.00053 | True              |
| gộp sentinel vào bin 0        |   0.71278 | -0.00235 | 0.00064 |  -3.68247 | -0.00412 | -0.00058 | True              |
| bỏ late_90                    |   0.68826 | -0.02687 | 0.00177 | -15.1418  | -0.0318  | -0.02194 | True              |
| bỏ cả ba biến late_*          |   0.5905  | -0.12463 | 0.00218 | -57.2259  | -0.13068 | -0.11858 | True              |

---

## 3. Hai dự đoán ghi trước

Mục này viết lại hai lần, vì hai lỗi khác nhau trong cùng một hàm mười dòng.
Cả hai ghi ở cuối mục.

### Với hai biến nhiều bin, ép sai chiều không phải là ép mà là xoá biến

Cột "mức còn lại" ở chiều tăng: `revolving_util` và `debt_ratio_valid` đều còn **1 mức trên
10**, tức PAVA gộp toàn bộ bin thường thành một hằng số. Đó không còn là một biến. Bảng đóng
góp biên xác nhận:

| biến | ép sai chiều (tăng) | bỏ hẳn biến |
|---|---|---|
| `revolving_util` | −0,05119 | −0,05107 |
| `debt_ratio_valid` | −0,00647 | −0,00631 |

Hai cột gần bằng nhau, qua hai đường đi khác hẳn: một bên PAVA gộp bin rồi fit lại, một bên bỏ
cột khỏi ma trận.

Bản đầu tôi gọi đây là "một phép kiểm chéo tình cờ có được cho cả bộ máy CV". Nói vậy là quá
tay, và nó vi phạm đúng tiêu chuẩn mà cả khối này dựng lên. Ép sai chiều gộp **toàn bộ** bin
thường thành một hằng số, mà một cột hằng số thì bị intercept hấp thụ hoàn toàn, nên "ép sai
chiều" bằng "bỏ cột" **theo cấu tạo** chứ không phải theo dữ liệu. Phép kiểm này gần như không
có đường để fail. Cái nó thật sự xác nhận chỉ là code chạy đúng như mô tả.

(Sau khi sửa lỗi hội tụ ở `fit_logit`, `gini_cv` bỏ hẳn cột hằng số trước khi fit, vì Hessian
của nó suy biến. Chênh lệch còn lại giữa hai cột là do bin đặc biệt vẫn nằm ngoài phép ép.)

### Ép đúng chiều thì mất bao nhiêu

Cột `d` ở chiều giảm, kèm khoảng tin cậy 95% tính từ SE của năm fold (`d ± 2,776·se`, `t` bảng
với 4 bậc tự do):

| biến | mức còn lại | d | KTC 95% của d | cái giá xấu nhất |
|---|---|---|---|---|
| `revolving_util` | 8/10 | +0,00006 | [−0,00135; +0,00146] | 0,00135 |
| `debt_ratio_valid` | 6/10 | +0,00083 | [−0,00052; +0,00218] | 0,00052 |
| `real_estate_loans` | 2/4 | +0,00004 | [−0,00157; +0,00164] | 0,00157 |
| `open_credit_lines` | 2/10 | +0,00070 | [−0,00067; +0,00207] | 0,00067 |

Cột cuối rất dễ lấy nhầm đầu, và tôi đã lấy nhầm một lần. `d = Gini(phương án) − Gini(model
đủ)`, nên **cái giá là −d**, và cái giá xấu nhất mà dữ liệu còn cho phép là **−ktc_lo**, tức đầu
*trái* của khoảng. Đầu phải là mức được lợi tối đa: với `debt_ratio_valid` đầu phải là 0,0022
trong khi cái giá xấu nhất thật ra chỉ 0,0005, nên lấy nhầm đầu vừa sai chiều vừa tự làm yếu
kết luận của mình.

Câu đúng **không** phải "tốn 0,000 Gini", vì dữ liệu không nói được điều đó. Câu đúng là: cái
giá xấu nhất còn tương thích với dữ liệu ở mức tin cậy 95% là **0,0016 Gini** trên cả bốn biến,
và **0,0014** nếu chỉ tính hai biến đáng đem đi dùng. Một chỗ phải nói rõ: bốn dòng trên đo
**từng biến một**, mà max của bốn cái giá riêng lẻ không phải cận trên của cái giá khi ép cả
bốn cùng lúc. Ép đồng thời thì phải đo đồng thời, và khối 4 đã làm: ép cả chín biến cho
`d = −0,00024` với cái giá xấu nhất **0,0019**. Kết luận định tính không đổi, nhưng con số
đúng để trích là 0,0019 chứ không phải 0,0015. Con số đó nhỏ hơn một bậc so với 0,0063
đến 0,0205 mà khối 2 đo đơn biến, và nhỏ hơn nhiều so với ±0,028 là khoảng tin cậy của chính
Gini trên tập OOT. Ở quy mô dữ liệu này nó không đo được.

Cột mức còn lại tách bảng thành hai nhóm rất khác nhau, và đây mới là chỗ có nội dung.

**`revolving_util` và `debt_ratio_valid` giữ được 8 và 6 mức**, gần hết cấu trúc, mà vẫn đơn
điệu và vẫn không mất gì đo được. Đây là kết quả đáng đem đi dùng.

**`real_estate_loans` chỉ đơn điệu được bằng cách rút về nhị phân** (nhóm 0, 1, 2 so với nhóm 3+).
Không mất gì, nhưng "đơn điệu 2 mức" không cùng một loại kết quả với "đơn điệu 8 mức".

**`open_credit_lines` thì câu hỏi gần như rỗng.** Bản đơn điệu của nó gộp bin 01 đến 09 thành
một mức, còn 2 trên 10, và `d` = −0,00006 gần trùng với `d` = +0,00004 của việc **xoá hẳn
biến**. Nó cũng là biến duy nhất mà chiều giảm **không** phải chiều nghiệp vụ: bin 01 có bad
rate 10,77%, cao nhất của biến, nên "rủi ro tăng theo số hạn mức" sai chiều với dữ liệu, và
đúng vì thế mà bản đơn điệu của nó phải gộp phẳng chín bin đầu.

### Phán quyết

**Dự đoán 2 sai.** Với `open_credit_lines`, ép đơn điệu chiều nào cũng không mất gì, và bản
thân biến cũng không mang gì. Với `real_estate_loans`, chữ U rút được về nhị phân mà không mất
gì, nên hình chữ U không phải thứ mang thông tin; cái mang thông tin là contrast "có từ ba
khoản bất động sản trở lên hay không".

Ở khối 2, `open_credit_lines` được đo là đáng **0,0176 Gini** khi so đơn biến, ép chiều tăng.
`real_estate_loans` thì bảng CV của khối 2 **không có dòng nào**; tôi chỉ đọc hình chữ U của nó
từ bảng bad rate, và đó chính là cách đọc mà khối này bác. (Dòng còn thiếu đó đã được bổ sung
khi rà soát lại, xem `results/iv_report.md` §2: +0,0205 ép chiều tăng, +0,1005 ép chiều giảm.)

Vì sao `open_credit_lines` không mang gì: nhánh trái của nó là nhóm bị hạn chế tín dụng,
utilization trung vị 0,439 so với 0,137 ở đáy chữ U, mà `revolving_util` là biến mạnh nhất
bảng. Model không cần biết một người chỉ có 2 hạn mức, nó đã biết người đó dùng gần hết hạn
mức đang có.

**Dự đoán 1 đúng một nửa, và nửa sai có lý do rất cụ thể.** Gộp bin 01 của `revolving_util` vào
bin 02 không làm mất gì (`d` = +0,00011), đúng như tôi đoán. Nhưng nó không làm biến đơn điệu:
nhóm gộp có bad rate 1,91%, vẫn cao hơn bin 03 (1,38%).

PAVA chiều giảm thì làm được, và nó cũng gộp bin, chỉ là **gộp 01+02+03** thành một mức có bad
rate 1,73%, thấp hơn bin 04 (1,83%) nên đơn điệu. Tôi đã gộp **thiếu đúng một bin**. Đó là phán
quyết chính xác cho dự đoán 1, và ai cũng kiểm được ngay trên bảng bad rate: 1,91% so với 1,38%
thì hỏng, 1,73% so với 1,83% thì xong.

**Dự đoán 3** (không ghi trong file nào, chỉ là kỳ vọng lúc mở khối 3): Gini test 0,60 đến
0,68. Sai vì bi quan, kết quả 0,6984.

### Hệ quả: scorecard này có thể làm đơn điệu mà không mất gì đo được

Đây là kết quả có giá trị thực tế nhất của mục này, và nó ngược với kết luận tôi ghi ở khối 2.
Đơn điệu là chuẩn mực của scorecard chính vì nó làm mã lý do nói được thành câu: "điểm của anh
thấp vì tỉ lệ sử dụng hạn mức cao", không kèm ngoại lệ "trừ khi anh dùng quá ít".

Cần nói rõ nó sửa được gì và không sửa được gì. Nó xoá cái móc ở bin 01 của `revolving_util`,
đúng chỗ tôi không giải thích được với khách hàng. Nó **không** sửa được việc 84,7% hồ sơ bị từ
chối nhận cùng một lý do ở mục 8: nguyên nhân của con số đó là `revolving_util` chi phối model,
và ép đơn điệu chỉ rút biên độ của biến đó xuống chút ít chứ không đổi bản chất.

Tôi **không** đổi cách chia bin trong khối này, vì bảng WOE và pipeline SQL của khối 2 đang là
nền cho mọi con số phía dưới, và đổi nền ở cuối khối là cách chắc chắn nhất để một chỗ nào đó
lệch mà không ai biết. Đây là việc đầu tiên của khối 4.

### Hai lỗi tôi đã mắc ở chính phép kiểm này

**Lỗi thứ nhất.** Bản đầu của `_pava` chọn bin để ép bằng `str(k).isdigit()`. `'3+'.isdigit()`
là `False`, nên với `real_estate_loans` (bin `0, 1, 2, 3+`) hàm chỉ ép trên `0, 1, 2` và bỏ
nguyên bin `3+` ra ngoài. Kết quả in ra là `d` = −0,00015, một con số nhỏ và hợp lý.

**Lỗi thứ hai.** Sau khi sửa, hàm tự chọn chiều bằng dấu hiệp phương sai có trọng số. Với
`real_estate_loans` nó chọn tăng, ra `d` = −0,00241, và tôi kết luận "chữ U sống sót". Chiều là
một lựa chọn có hậu quả lớn hơn cả bản thân phép ép, mà tôi để nó cho một heuristic không in ra
đâu cả. `force_monotone` bây giờ **bắt buộc** truyền chiều, không còn chế độ tự đoán.

Cùng họ với lỗi IV sai 10 lần ở khối 2: phép kiểm chạy trót lọt, in ra con số trông hợp lý,
không có gì báo rằng nó đang đo một thứ khác với thứ tôi nghĩ. Ba lần rồi, và cả ba lần đều lộ
ra nhờ đối chiếu chứ không nhờ chạy lại.

### Hai điều phải nói kèm về mặt thống kê

**Đa so sánh.** Khối này chạy 27 phép kiểm ghép cặp trong ba bảng (10 dòng đóng góp biên, 8
dòng hai chiều, 9 dòng gộp bin), và với mỗi biến tôi báo cả hai chiều rồi bàn
về chiều rẻ hơn. Nếu đó là dò số thì các t quanh 2 đến 4 mất giá trị (Benjamini & Hochberg 1995;
Berk và cộng sự 2013). Nhưng chiều ở đây **không phải kết quả của việc dò**: chiều rủi ro tăng
dần là chiều duy nhất triển khai được trong một scorecard, nên đó là chiều cần đo từ đầu, còn
chiều kia chỉ để định giá việc chọn sai. Với `open_credit_lines` tôi nói thẳng ở trên rằng chiều
đó không đúng nghiệp vụ, thay vì lấy con số rẻ hơn rồi im.

**t-test trên k-fold đánh giá thấp phương sai**, vì năm tập fit chồng nhau tới 75% nên các `d`
không độc lập (Dietterich 1998; Bengio & Grandvalet 2004). Hệ quả là |t| bị thổi lên. Chiều lệch
này **có lợi** cho các kết luận null ở đây: một phép kiểm vốn dễ bác mà vẫn không bác được thì
câu "không đo được" càng chắc. Ngược lại nó làm yếu đúng những khẳng định dương nhỏ, cụ thể là
`monthly_income` (t = −3,39) và "gộp 3+ vào 2" (t = −3,57). Tôi không lấy hai con số đó làm căn
cứ quyết định. Chỗ tôi **có** dựa vào một t cùng cỡ là dòng "gộp hết" của `real_estate_loans`
(t = −3,37) để nói biến này mang thông tin; ở đó tôi có một bằng chứng độc lập không dính CV,
là hệ số Wald của biến trong model cuối, z = 8,66 trên toàn bộ train.

### Quyết định

**`open_credit_lines` bị loại**, vì đóng góp biên bằng không (+0,00006) đo bằng CV trong train.
Hệ số âm ở mục 1 **không phải lý do thứ hai độc lập**: z = −0,79 nên nó không phân biệt được
với 0, và nó là cùng một sự việc nhìn từ góc khác. Nó là hệ quả không bảo vệ được về mặt nghiệp
vụ: giữ biến lại thì bảng điểm sẽ trừ điểm người có ít hạn mức hơn, và tôi không có câu trả lời
nào đúng cho câu hỏi đó.

`monthly_income` (−0,00016) và `dependents` (−0,00006) cũng gần bằng không nhưng **giữ lại**:
dấu đúng, đóng góp không âm, và là hai biến bên kinh doanh mong thấy trong một scorecard. Bỏ
chúng đổi lấy 0,0002 Gini là đổi một câu hỏi khó lấy một con số không đo được.

Khối 2 còn để lại một cảnh báo: chỗ chồng lấn thật nằm ở cặp `debt_ratio_valid` và
`monthly_income` (31.365 dòng dùng chung thông tin missing), và nếu có hệ số lạ thì nhìn ở đó
trước. Nửa đúng: `monthly_income` đúng là bị chiết khấu gần hết (hệ số 0,0901, z = 1,67) trong
khi `debt_ratio_valid` giữ nguyên sức mạnh (0,7969); nhưng chỗ **lật dấu** lại rơi vào một biến
không nằm trong cảnh báo đó.

### Đọc bảng gộp bin của `real_estate_loans`

Bốn dòng `real_estate` phải đọc cùng nhau, vì đọc lẻ thì mâu thuẫn:

| phép gộp | bảng bin còn lại | d |
|---|---|---|
| gộp 0 vào 1 | nhóm 0+1 −0,024 · bin 2 +0,188 · bin 3+ −0,256 | −0,00173 |
| gộp 3+ vào 2 | bin 0 −0,238 · bin 1 +0,260 · nhóm 2+3+ +0,065 | −0,00158 |
| gộp 0,1,2 | nhóm 0+1+2 +0,021 · bin 3+ −0,256 | +0,00004 |
| gộp hết | một bin | −0,00300 |

Dòng thứ ba nói toàn bộ đóng góp của biến nằm ở contrast `3+` so với phần còn lại. Nhưng dòng
thứ hai xoá đúng contrast đó mà chỉ mất một nửa, và lý do là nó **không xoá hẳn**: bin gộp
gộp 2 với 3+ có WOE +0,065, vẫn nằm dưới bin 1 (+0,260), nên một phiên bản pha loãng của cùng
contrast sống sót.

Dòng đầu mất 0,00173 dù nhánh trái được cho là không mang gì. Cơ chế: gộp bin 0 (WOE −0,238)
với bin 1 (+0,260) tạo ra một contrast giả giữa nhóm gộp (−0,024) và bin 2 (+0,188), và một hệ
số duy nhất buộc phải quy contrast giả đó thành điểm. Hai dòng đầu vì vậy không đo "giá trị của
một nhánh", chúng đo hậu quả của việc gộp hai bin ngược dấu; chỉ dòng ba và dòng bốn mới trả
lời được câu hỏi ban đầu.

Dòng "gộp 0,1,2" và dòng "ép giảm" ở bảng trên cho **đúng cùng một con số** `d` = +0,00004, và
đó **không phải tình cờ**: cả hai đều rút biến xuống đúng hai mức, mà một cột chỉ có hai mức thì
mọi cách gán giá trị đều sai khác nhau một phép biến đổi affine, nên logistic fit ra cùng một
model.

Bản đầu hai con số này lệch nhau 1e-05 và tôi giải thích rằng đó là do hai đường tính WOE khác
nhau (+0,021 so với +0,044). Giải thích đó **tự mâu thuẫn**: nếu hai cột affine với nhau thì giá
trị WOE khác nhau không thể đổi kết quả fit. Phần dư 1e-05 là nhiễu tối ưu hoá của lỗi hội tụ
nói ở mục 1, và sau khi sửa `fit_logit` thì nó biến mất hẳn. Một con số dư nhỏ mà có sẵn lời giải thích nghe hợp lý thì rất dễ trôi qua.

### Việc để lại từ khối 2: rủi ro leakage của `late_90`

`late_90` đo **đúng cùng một sự kiện** với target, chỉ khác cửa sổ thời gian, và không có cột
ngày để kiểm hai cửa sổ có tách bạch không. Không kiểm được thì đo mức phụ thuộc.

Gộp bin sentinel vào bin 0, tức giả vờ 193 dòng đó không có trễ hạn, làm mất 0,00235 Gini, năm
fold cùng dấu. 0,18% dữ liệu mà đáng 0,0024 Gini là nhiều, đúng như bad rate 55,96% của nhóm
đó báo trước.

Bỏ hẳn `late_90` mất 0,0269. Bỏ cả ba biến `late_*` làm Gini rơi từ 0,715 xuống 0,590, mất
0,1246, tức **17% sức mạnh của model nằm ở lịch sử trễ hạn**.

Con số 0,1246 là **cận trên** của thiệt hại nếu hoá ra cửa sổ đo feature chồng lấn cửa sổ
target, cận trên vì nó bỏ toàn bộ ba biến chứ không chỉ phần chồng lấn. Ngay ở kịch bản xấu
nhất đó scorecard còn Gini 0,590, vẫn dùng được. Rủi ro leakage ở đây là rủi ro **phóng đại
thành tích**, không phải rủi ro model rỗng.

---

## 4. Model cuối

Chín biến, chia bin giữ nguyên như khối 2, không gộp bin nào.

|                   |   hệ số |     SE |        z |
|:------------------|--------:|-------:|---------:|
| (intercept)       |  2.6052 | 0.0152 | 171.8416 |
| age               |  0.3777 | 0.0319 |  11.8466 |
| debt_ratio_valid  |  0.7969 | 0.0522 |  15.2637 |
| dependents        |  0.2626 | 0.0755 |   3.4791 |
| late_30_59        |  0.5219 | 0.0153 |  34.0035 |
| late_60_89        |   0.368 |  0.017 |   21.601 |
| late_90           |  0.5143 | 0.0137 |  37.4431 |
| monthly_income    |  0.0901 |  0.054 |   1.6683 |
| real_estate_loans |  0.5477 | 0.0631 |   8.6822 |
| revolving_util    |  0.6239 | 0.0146 |  42.6105 |

| | Gini | KS |
|---|---|---|
| train | 0.7165 | 0.5574 |
| test | 0.6984 | 0.5472 |

Mọi hệ số dương, Gini test bằng đúng model 10 biến. Đây là báo cáo chứ không phải căn cứ:
quyết định bỏ `open_credit_lines` đã ra đời ở mục 3 bằng CV trong train, và nếu test có nói
ngược thì tôi vẫn phải giữ quyết định đó rồi ghi lại mâu thuẫn, chứ không được đổi ý theo test.

`monthly_income` có z = 1,67, p ≈ 0,10, không có ý nghĩa ở mức 5%. Có một mâu thuẫn bề
ngoài đáng nói: CV bảo bỏ nó làm Gini giảm 0,00017 với t = −3,39 (có ý nghĩa), Wald bảo hệ số
không khác 0. Hai phép kiểm hỏi hai câu khác nhau. Wald hỏi hệ số có khác 0 trên một mẫu
train; CV ghép cặp hỏi việc bỏ biến có làm giảm Gini nhất quán qua các fold, và vì ghép cặp
nên phát hiện được cả chênh lệch cực nhỏ. Cả hai cùng nói một điều: biến này gần như không có
tác dụng.

---

## 5. Thang điểm

Điểm là hàm **tuyến tính của log-odds**, không phải của xác suất:

```
score = Offset + Factor · ln(odds)
```

Tuyến tính theo log-odds thì "thêm bao nhiêu điểm" mới có nghĩa cố định trên toàn thang: cộng
cùng một số điểm luôn nhân odds với cùng một hệ số, dù đang ở đầu nào.

Ba tham số quy ước (Siddiqi 2017): **PDO** = 20 điểm làm odds gấp đôi, **base odds** =
50:1, **base score** = 600. Giải từ hai ràng buộc:

```
600     = Offset + Factor·ln(50)
620     = Offset + Factor·ln(100)
-->  20 = Factor·ln 2  -->  Factor = PDO/ln 2 = 28.8539
                              Offset = 600 - Factor·ln(50) = 487.1229
```

Ba hằng số này **không đổi thứ tự xếp hạng**. Điểm là biến đổi tuyến tính đơn điệu của log-odds
nên Gini, KS, AUC không đổi dù chọn bộ nào; chúng chỉ quyết định con số hiện ra trông thế nào.
Sau bước làm tròn về số nguyên thì PDO có ảnh hưởng đến độ chính xác của PD đọc từ điểm, xem
mục 6.

Chia tổng điểm về từng bin:

```
score = Offset + Factor·(b0 + Σ bj·WOEj)
      = Σ [ (bj·WOEj + b0/n)·Factor + Offset/n ]
```

Việc chia `b0` và `Offset` đều cho n biến là **quy ước cho tiện**, không phải kết quả toán học:
nó dời điểm qua lại giữa các biến chứ không đổi tổng. Hệ quả khi đọc bảng điểm: so sánh điểm
tuyệt đối giữa hai biến khác nhau là vô nghĩa, chỉ chênh lệch **trong cùng một biến** mới có nghĩa.

### Biên độ điểm

| variable          |   min |   max |   biên độ |
|:------------------|------:|------:|----------:|
| revolving_util    |    36 |    93 |        57 |
| late_90           |    16 |    68 |        52 |
| late_30_59        |    19 |    71 |        52 |
| late_60_89        |    30 |    66 |        36 |
| debt_ratio_valid  |    49 |    70 |        21 |
| age               |    56 |    75 |        19 |
| real_estate_loans |    58 |    67 |         9 |
| dependents        |    60 |    66 |         6 |
| monthly_income    |    61 |    64 |         3 |

Biên độ là dạng đọc được nhất của hệ số: `revolving_util` chênh 57 điểm giữa bin tốt nhất và
xấu nhất, tức 2,85 lần PDO, nên riêng biến này đã làm odds chênh 2^2,85 ≈ 7 lần.
`monthly_income` chênh 3 điểm, gần như không tham gia quyết định.

Tổng điểm chạy từ 385 đến 640. Thang hẹp so với các thang thương mại (FICO 300 đến 850)
vì nó là **hệ quả** của model chứ không phải thiết kế: biên độ bằng Factor nhân biên độ
log-odds mà chín biến này tạo ra được.

### Bảng điểm đầy đủ

Cột "thiếu" là số điểm mất so với bin tốt nhất của chính biến đó, tức đầu vào của mã lý do.

#### `age`

|   bin |     n |   bad % |     WOE |   điểm |   thiếu |
|------:|------:|--------:|--------:|-------:|--------:|
|    01 | 12007 |   11.52 | -0.5974 |     56 |      19 |
|    02 | 10401 |    9.53 | -0.3855 |     58 |      17 |
|    03 | 11139 |    8.47 | -0.2556 |     60 |      15 |
|    04 | 10319 |    8.15 | -0.2142 |     60 |      15 |
|    05 | 10402 |    7.84 | -0.1713 |     61 |      14 |
|    06 |  9918 |    6.31 |  0.0613 |     63 |      12 |
|    07 |  9352 |    4.88 |  0.3346 |     66 |       9 |
|    08 | 11443 |    4.10 |  0.5164 |     68 |       7 |
|    09 | 10018 |    2.76 |  0.9275 |     73 |       2 |
|    10 | 10000 |    2.18 |  1.1675 |     75 |       0 |

#### `debt_ratio_valid`

| bin       |     n |   bad % |     WOE |   điểm |   thiếu |
|:----------|------:|--------:|--------:|-------:|--------:|
|    01 |  8303 |    4.91 |  0.3264 |     70 |       0 |
|    02 |  8303 |    6.95 | -0.0418 |     62 |       8 |
|    03 |  8302 |    6.49 |  0.0311 |     63 |       7 |
|    04 |  8303 |    6.02 |  0.1114 |     65 |       5 |
|    05 |  8301 |    5.01 |  0.3057 |     70 |       0 |
|    06 |  8303 |    5.43 |  0.2208 |     68 |       2 |
|    07 |  8301 |    6.30 |  0.0632 |     64 |       6 |
|    08 |  8302 |    7.58 | -0.1350 |     59 |      11 |
|    09 |  8302 |    9.41 | -0.3714 |     54 |      16 |
|    10 |  8302 |   11.62 | -0.6077 |     49 |      21 |
| X_INVALID | 21977 |    5.59 |  0.1900 |     67 |       3 |

#### `dependents`

| bin       |     n |   bad % |     WOE |   điểm |   thiếu |
|:----------|------:|--------:|--------:|-------:|--------:|
|     0 | 60920 |    5.84 |  0.1433 |     64 |       0 |
|     1 | 18377 |    7.50 | -0.1246 |     62 |       2 |
|     2 | 13625 |    8.07 | -0.2039 |     61 |       3 |
|    3+ |  9324 |    9.19 | -0.3458 |     60 |       4 |
| 9_MISSING |  2753 |    4.43 |  0.4348 |     66 |      -2 |

#### `late_30_59`

| bin        |     n |   bad % |     WOE |   điểm |   thiếu |
|:-----------|------:|--------:|--------:|-------:|--------:|
|     0 | 88270 |    3.98 |  0.5473 |     71 |       0 |
|     1 | 11189 |   15.28 | -0.9237 |     49 |      22 |
|     2 |  3173 |   26.32 | -1.6067 |     38 |      33 |
|   3-4 |  1771 |   37.72 | -2.1348 |     30 |      41 |
|    5+ |   403 |   45.91 | -2.4722 |     25 |      46 |
| 9_SENTINEL |   193 |   55.96 | -2.8758 |     20 |      48 |

#### `late_60_89`

| bin        |     n |   bad % |     WOE |   điểm |   thiếu |
|:-----------|------:|--------:|--------:|-------:|--------:|
|     0 | 99756 |    5.11 |  0.2851 |     66 |       0 |
|     1 |  3936 |   31.02 | -1.8372 |     43 |      23 |
|     2 |   778 |   50.00 | -2.6363 |     34 |      32 |
|    3+ |   336 |   60.12 | -3.0467 |     30 |      36 |
| 9_SENTINEL |   193 |   55.96 | -2.8758 |     20 |      48 |

#### `late_90`

| bin        |     n |   bad % |     WOE |   điểm |   thiếu |
|:-----------|------:|--------:|--------:|-------:|--------:|
|     0 | 99149 |    4.64 |  0.3872 |     68 |       0 |
|     1 |  3662 |   33.42 | -1.9472 |     34 |      34 |
|     2 |  1109 |   48.60 | -2.5804 |     24 |      44 |
|   3-4 |   667 |   61.77 | -3.1161 |     16 |      52 |
|    5+ |   219 |   62.56 | -3.1496 |     16 |      52 |
| 9_SENTINEL |   193 |   55.96 | -2.8758 |     20 |      48 |

#### `monthly_income`

| bin       |     n |   bad % |     WOE |   điểm |   thiếu |
|:----------|------:|--------:|--------:|-------:|--------:|
|    01 |  8527 |    9.08 | -0.3320 |     62 |       1 |
|    02 |  8103 |    9.74 | -0.4095 |     61 |       2 |
|    03 |  8389 |    8.62 | -0.2752 |     62 |       1 |
|    04 |  8218 |    8.35 | -0.2403 |     62 |       1 |
|    05 |  8309 |    7.00 | -0.0503 |     62 |       1 |
|    06 |  8404 |    6.66 |  0.0033 |     62 |       1 |
|    07 |  8166 |    5.98 |  0.1195 |     63 |       0 |
|    08 |  8306 |    5.00 |  0.3089 |     63 |       0 |
|    09 |  8308 |    4.63 |  0.3880 |     63 |       0 |
|    10 |  8292 |    4.67 |  0.3805 |     63 |       0 |
| X_MISSING | 20807 |    5.72 |  0.1661 |     63 |       0 |
| X_ZERO |  1170 |    3.33 |  0.7310 |     64 |      -1 |

#### `real_estate_loans`

| bin   |     n |   bad % |     WOE |   điểm |   thiếu |
|:------|------:|--------:|--------:|-------:|--------:|
|     0 | 39283 |    8.33 | -0.2382 |     59 |       8 |
|     1 | 36645 |    5.23 |  0.2605 |     67 |       0 |
|     2 | 22087 |    5.60 |  0.1884 |     65 |       2 |
|    3+ |  6984 |    8.46 | -0.2552 |     58 |       9 |

#### `revolving_util`

| bin           |     n |   bad % |     WOE |   điểm |   thiếu |
|:--------------|------:|--------:|--------:|-------:|--------:|
|    01 | 10483 |    2.55 |  1.0082 |     81 |      12 |
|    02 | 10483 |    1.27 |  1.7181 |     93 |       0 |
|    03 | 10482 |    1.38 |  1.6305 |     92 |       1 |
|    04 | 10482 |    1.83 |  1.3451 |     87 |       6 |
|    05 | 10482 |    2.42 |  1.0593 |     82 |      11 |
|    06 | 10482 |    3.42 |  0.7058 |     75 |      18 |
|    07 | 10482 |    5.27 |  0.2535 |     67 |      26 |
|    08 | 10482 |    8.64 | -0.2783 |     57 |      36 |
|    09 | 10482 |   16.47 | -1.0124 |     44 |      49 |
|    10 | 10482 |   23.57 | -1.4601 |     36 |      57 |
| X_IMPLAUSIBLE |   177 |    7.91 | -0.1816 |     59 |      34 |

Hai chi tiết sẽ quay lại ở phần mã lý do.

`revolving_util` bin 01 được 81 điểm, **thiếu 12** so với bin 02. Đó là cái móc từ khối 2, giờ
có giá cụ thể: 12 điểm là 0,6 PDO, tức odds thấp hơn khoảng 1,5 lần. Người có ít hạn mức
nhất bị trừ điểm, và bảng điểm nói thẳng ra thay vì giấu trong hệ số.

`monthly_income` bin `X_ZERO` được **điểm cao nhất** của biến, cao hơn cả nhóm thu nhập cao
nhất. Đây là hệ quả trực tiếp của phát hiện ở khối 1 rằng missing mang thông tin ngược trực
giác. Đúng về thống kê trên bộ này, và là câu hỏi khó đầu tiên bất kỳ ai nhìn bảng điểm cũng
hỏi. Biên độ cả biến chỉ 3 điểm nên nó không đổi được quyết định nào, nhưng vẫn phải giải
thích được.

---

## 6. Điểm có quy ngược về đúng model không

| phép kiểm | kết quả |
|---|---|
| điểm chưa làm tròn → log-odds | lệch lớn nhất 6.22e-15 |
| làm tròn về số nguyên → PD | lệch lớn nhất 2.84 điểm %, trung bình 0.172 điểm % |

Lệch 6e-15 là sai số dấu phẩy động, tức phép quy đổi đúng chính xác chứ không phải xấp
xỉ. Đáng kiểm vì rất dễ sai dấu hoặc quên chia `b0` cho n, và cái sai đó **không lộ ra ở bất
kỳ chỉ số xếp hạng nào**: Gini vẫn y nguyên vì thứ tự không đổi, chỉ PD suy ra từ điểm là sai.

Chỗ lệch lớn nhất **không** nằm ở đáy thang như tôi tưởng lúc đầu mà nằm ở giữa, tại 488 điểm,
nơi PD model là 52,1% so với 49,2% đọc từ bảng điểm. Lý do là số học: sai số làm tròn cộng dồn tối đa 9 × 0,5 = 4,5 điểm, tức 0,156
đơn vị log-odds, và `|ΔPD| ≈ p(1−p)·Δ log-odds`, mà `p(1−p)` cực đại ở p = 0,5. Ở hai đầu thang
cùng một sai số điểm lại cho sai số PD nhỏ: trung bình chỉ 0,05 điểm phần trăm ở nhóm trên 600
điểm.

Làm tròn đổi lấy một bảng điểm cộng được bằng tay. Muốn giảm sai số thì tăng PDO, vì PDO lớn
hơn nghĩa là mỗi điểm mang ít log-odds hơn.

Phân bố điểm trên train: min 406, p1 466, trung vị 593, p99 631, max 637.
Lệch trái mạnh, đúng hình dạng của một danh mục có bad rate 6,7%.

---

## 7. Điểm có nói đúng odds không

Thang điểm được **định nghĩa** sao cho 600 điểm là odds 50:1 và mỗi 20 điểm gấp đôi odds.
Đó là điều model nói. Câu hỏi khác hẳn là dữ liệu có nói vậy không. Gini 0,6984 nói model
**xếp hạng** tốt; nó không nói gì về việc con số PD suy ra từ điểm có đúng không.

### train

|   dec |     n |   điểm TB |   PD dự báo % |   PD thực % |   odds thực |   odds model |
|------:|------:|----------:|--------------:|------------:|------------:|-------------:|
|     0 | 10777 |    508.27 |         34.44 |       35.3  |        1.83 |         2.08 |
|     1 | 10562 |    550.75 |         10.25 |       11.32 |        7.83 |         9.07 |
|     2 | 10360 |    565.25 |          6.43 |        6.88 |       13.53 |        14.99 |
|     3 | 10702 |    577.83 |          4.26 |        4.66 |       20.45 |        23.19 |
|     4 | 11367 |    588.76 |          2.96 |        2.66 |       36.64 |        33.87 |
|     5 |  9765 |    597.11 |          2.24 |        1.78 |       55.12 |        45.24 |
|     6 | 10774 |    604.07 |          1.78 |        1.23 |       80.62 |        57.57 |
|     7 | 11418 |    611.05 |          1.41 |        0.85 |      116.71 |        73.34 |
|     8 |  8835 |    617.41 |          1.13 |        0.71 |      139.24 |        91.42 |
|     9 | 10439 |    625.83 |          0.85 |        0.36 |      273.71 |       122.4  |

Brier = 0,05031, PD dự báo trung bình 6,684%, bad rate thực 6,684%. Hai con số bằng nhau tới
chữ số thứ sáu, đúng như phương trình chuẩn tắc của hợp lý cực đại đòi hỏi; bản đầu ghi 6,678%
và gọi là "khớp gần như hoàn hảo", trong khi đó là dấu hiệu model chưa hội tụ.

### test

|   dec |    n |   điểm TB |   PD dự báo % |   PD thực % |   odds thực |   odds model |
|------:|-----:|----------:|--------------:|------------:|------------:|-------------:|
|     0 | 2262 |    508.92 |         34.02 |       35.19 |        1.84 |         2.13 |
|     1 | 2297 |    550.66 |         10.28 |       11.32 |        7.83 |         9.04 |
|     2 | 2276 |    565.43 |          6.39 |        6.55 |       14.28 |        15.09 |
|     3 | 2312 |    577.84 |          4.26 |        4.5  |       21.23 |        23.19 |
|     4 | 2119 |    588.16 |          3.02 |        2.55 |       38.24 |        33.18 |
|     5 | 2235 |    596.64 |          2.28 |        2.46 |       39.64 |        44.51 |
|     6 | 2370 |    604.03 |          1.78 |        1.22 |       80.72 |        57.49 |
|     7 | 2474 |    611.06 |          1.4  |        1.17 |       84.31 |        73.34 |
|     8 | 2140 |    617.86 |          1.11 |        0.56 |      177.33 |        92.86 |
|     9 | 2015 |    626.47 |          0.83 |        0.79 |      124.94 |       125.14 |

Brier = 0,05020, PD dự báo trung bình 6,592%, bad rate thực 6,684%.

Trên train PD dự báo trung bình khớp bad rate thực gần như hoàn hảo, nhưng đó **không phải
bằng chứng gì cả**: hồi quy logistic ước lượng bằng hợp lý cực đại luôn cho trung bình dự báo
bằng đúng trung bình quan sát trên chính tập fit. Đó là tính chất của phương trình chuẩn tắc.

Chỗ có thông tin là từng decile, và ở đó model sai **có hệ thống**: decile thấp nhất dự báo
34,44% trong khi thực tế 35,30%, decile cao nhất dự báo 0,85% trong khi thực tế 0,36%. Dự báo
bị **nén về giữa**, model chưa đủ tự tin ở cả hai đầu. Đọc trên thang điểm: ở 626 điểm model
bảo odds 122:1 nhưng thực tế 274:1.

Hai điều rút ra. Thang điểm PDO chỉ **đổi nhãn** cho log-odds của model chứ không tự làm nó
đúng; model lệch thì thang điểm lệch y hệt. Và đây là loại sai **sửa được sau**, bằng một hàm
đơn điệu áp lên điểm, nên không đụng gì đến Gini; ngược lại thì không, xếp hạng sai thì không
calibration nào cứu được. Để xử lý ở khối 5 cùng với PSI.

Trên test hình dạng lặp lại gần y hệt, nên đây không phải overfit mà là **dạng hàm**: logistic
tuyến tính theo WOE quá trơn so với quan hệ thật ở hai đuôi.

---

## 8. Mã lý do

ECOA Regulation B (12 CFR 1002.9) buộc bên cho vay từ chối hồ sơ phải nêu **các lý do chính**,
cụ thể, chứ không được nói "hệ thống chấm điểm từ chối". Tháng 5/2022 CFPB ra Circular 2022-03
nói rõ yêu cầu này không được miễn trừ vì model quá phức tạp để giải thích. Đây là lý do thực
sự để làm scorecard thay vì bắn thẳng XGBoost vào bài toán.

Với bảng điểm cộng thì việc này thành số học: mỗi biến lấy điểm cao nhất có thể cho, trừ đi
điểm hồ sơ thực nhận, xếp giảm dần, lấy ba cái đầu.

Chọn mốc "điểm cao nhất" thay vì "trung bình quần thể" là lựa chọn có thể tranh luận.
Regulation B không quy định công thức, chỉ đòi hỏi lý do nêu ra phải là lý do thật sự dẫn đến
quyết định. Mốc điểm cao nhất trả lời đúng câu khách hàng hỏi là tôi mất điểm ở đâu; nhược
điểm là nó luôn chỉ ra biến có biên độ lớn nhất, nên với hồ sơ gần ngưỡng thì lý do nêu ra
chưa chắc là thứ thực sự đẩy họ qua ngưỡng.

### Năm hồ sơ điểm thấp nhất của tập test

- **#104643**, 409 điểm, PD 93,6%, nhãn thực tế **xấu**: `revolving_util` bin 10 (−57 điểm); `late_90` bin 3-4 (−52 điểm); `late_30_59` bin 3-4 (−41 điểm)
- **#29326**, 410 điểm, PD 93,9%, nhãn thực tế **xấu**: `revolving_util` bin 10 (−57 điểm); `late_30_59` bin 9_SENTINEL (−52 điểm); `late_90` bin 9_SENTINEL (−48 điểm)
- **#138826**, 413 điểm, PD 92,4%, nhãn thực tế **xấu**: `revolving_util` bin 10 (−57 điểm); `late_30_59` bin 5+ (−46 điểm); `late_90` bin 2 (−44 điểm)
- **#42082**, 413 điểm, PD 92,6%, nhãn thực tế **xấu**: `revolving_util` bin 10 (−57 điểm); `late_90` bin 5+ (−52 điểm); `late_30_59` bin 3-4 (−41 điểm)
- **#37025**, 415 điểm, PD 92,5%, nhãn thực tế **xấu**: `revolving_util` bin 10 (−57 điểm); `late_30_59` bin 9_SENTINEL (−52 điểm); `late_90` bin 9_SENTINEL (−48 điểm)

### Lý do số 1 tập trung ở đâu

Ngưỡng 580, tập test: 8.291 hồ sơ bị từ chối (36,8%).

|                  |   % hồ sơ bị từ chối |
|:-----------------|---------------------:|
| revolving_util   |                 84.7 |
| late_30_59       |                  8.7 |
| late_90          |                  3.7 |
| late_60_89       |                  1.4 |
| debt_ratio_valid |                  1.4 |
| age              |                  0.1 |

84,7% hồ sơ bị từ chối nhận cùng một lý do số 1. Về kỹ thuật thì đúng,
`revolving_util` có biên độ 57 điểm nên hầu như luôn thắng. Về mục đích của Regulation B thì
đây là **vấn đề**: một thông báo từ chối mà 5 trên 6 người nhận nội dung giống hệt nhau thì
không giúp ai biết mình cần sửa gì.

Không phải lỗi cài đặt mà là hệ quả của việc một biến chi phối model. Siddiqi (2017) chương về
triển khai đưa ra hai hướng: ghép biến thành các nhóm lý do rồi buộc ba mã phải đến từ ba nhóm
khác nhau, hoặc đổi mốc so sánh từ điểm tối đa sang điểm của một hồ sơ tham chiếu ở giữa quần
thể. Tôi **chưa làm** cái nào, vì chọn giữa chúng là quyết định của bên nghiệp vụ chứ không
phải của model, và ghi lại đây làm giới hạn đã biết.

---

## 9. Chọn ngưỡng cắt

Model cho điểm; ngưỡng cắt là quyết định kinh doanh. Bảng tính trên tập test.

|   ngưỡng |   duyệt % |   bad trong nhóm duyệt % |   bad trong nhóm từ chối % |   bắt được bad % |
|---------:|----------:|-------------------------:|---------------------------:|-----------------:|
|      520 |     94.49 |                     4.44 |                      45.24 |            37.3  |
|      540 |     90.24 |                     3.53 |                      35.88 |            52.39 |
|      560 |     79.09 |                     2.45 |                      22.7  |            71.01 |
|      570 |     71.47 |                     1.98 |                      18.48 |            78.86 |
|      580 |     63.04 |                     1.62 |                      15.32 |            84.71 |
|      590 |     53.33 |                     1.34 |                      12.79 |            89.3  |
|      600 |     41.35 |                     0.97 |                      10.71 |            94.02 |
|      610 |     26.24 |                     0.76 |                       8.79 |            97.01 |

Ngưỡng 580 duyệt 63,0%, bad rate nhóm duyệt còn 1,62% so với 6,68% nếu duyệt hết, bắt được
84,7% số ca xấu. Ngưỡng 600 đẩy bad rate xuống 0,97% nhưng chỉ còn duyệt 41,3%.

Cột `bad trong nhóm từ chối %` hay bị bỏ quên. Ở ngưỡng 580, nhóm bị từ chối có bad rate 15,32%, nghĩa là **gần
85% số người bị từ chối lẽ ra vẫn trả nợ bình thường**. Đó là cái giá của việc cắt: mỗi ca xấu
chặn được đi kèm khoảng năm hồ sơ tốt bị đuổi. Không ngưỡng nào làm tỉ lệ đó nhỏ đi mà không
kéo bad rate của nhóm duyệt lên. Chọn ngưỡng là đặt giá cho hai loại sai này, và cái giá đó
đến từ biên lợi nhuận và tổn thất khi vỡ nợ chứ không từ dữ liệu.

Một cảnh báo về chính bảng này: nó tính trên **những người đã được duyệt trong quá khứ**, vì
bộ dữ liệu chỉ có nhóm đó. Áp lên dòng hồ sơ thật, nơi có cả người trước đây bị từ chối, tỉ lệ
duyệt và bad rate sẽ khác. Đây đúng là chỗ cần reject inference, nằm ngoài phạm vi vì dữ liệu
không có hồ sơ bị từ chối.

---

## Giới hạn đã biết

- Calibration lệch theo decile ở cả train và test, chưa sửa. Sửa ở khối 5.
- 84,7% hồ sơ bị từ chối nhận cùng một lý do số 1, thông báo từ chối gần như không mang thông
  tin riêng. Cần ghép chủ đề lý do ở mức nghiệp vụ.
- Bảng ngưỡng cắt tính trên quần thể đã được duyệt, không có reject inference.
- `revolving_util` bin 01 vẫn chưa có lý do nghiệp vụ nói được với khách hàng. Mục 3 đã tìm
  được cách xoá nó mà không mất Gini đo được (ép đơn điệu chiều giảm, gộp 01+02+03), nhưng bảng
  điểm nộp ở đây là bảng chưa đơn điệu, nên giới hạn này còn nguyên trong sản phẩm.
- Điểm cắt bin không được tính lại trong từng fold CV, chỉ WOE mới được, nên các chênh lệch
  Gini báo ở mục 2 và 3 lạc quan hơn thực tế một chút.
- Cận trên thiệt hại nếu `late_*` có leakage thời gian là 0,125 Gini.

## Việc để lại cho bước sau

- Khối 4, việc đầu tiên: chia bin lại đơn điệu theo chiều rủi ro tăng dần, rồi dựng lại bảng
  điểm và mã lý do. Cái giá đã đo ở mục 3.
- Khối 4, sau đó: XGBoost trên cùng bộ chia bin, so Gini với scorecard, SHAP để xem cây tìm
  được tương tác nào mà mô hình cộng tính bỏ sót. Không biến nào trong bộ này còn hình dạng
  **phi đơn điệu** có giá trị sau khi kiểm đa biến (bậc thang WOE vẫn là phi tuyến, và toàn bộ
  sức mạnh của scorecard nằm ở đó, nên đừng lẫn hai chữ này). Nếu XGBoost vượt scorecard đáng
  kể thì phần vượt phải đến từ một trong ba chỗ: **tương tác giữa các biến**, **chỗ cắt bin**
  mịn hơn hoặc khác đi (điểm cắt lấy nguyên `NTILE(10)` của khối 2, khối này chưa thử cắt khác),
  hoặc độ cong phi đơn điệu mà CV của tôi không phát hiện được. Ghi lại trước khi chạy, và ghi
  cả ba chứ không chỉ một, vì một dự đoán loại trừ thiếu thì kiểu gì cũng đúng.
- Khối 5: calibration (Platt / isotonic), PSI/CSI, và tập OOT.

---

## Nguồn

- Siddiqi, N. (2017), *Intelligent Credit Scoring*, 2nd ed., Wiley — scaling PDO/base-odds,
  chia điểm về bin, mã lý do.
- Barlow, R. E., Bartholomew, D. J., Bremner, J. M. & Brunk, H. D. (1972), *Statistical
  Inference Under Order Restrictions*, Wiley — pool-adjacent-violators.
- Dietterich, T. G. (1998), "Approximate statistical tests for comparing supervised
  classification learning algorithms", *Neural Computation*, 10(7), 1895–1923.
- Bengio, Y. & Grandvalet, Y. (2004), "No unbiased estimator of the variance of k-fold
  cross-validation", *JMLR*, 5, 1089–1105.
- Benjamini, Y. & Hochberg, Y. (1995), "Controlling the false discovery rate", *JRSS-B*, 57(1),
  289–300.
- Berk, R., Brown, L., Buja, A., Zhang, K. & Zhao, L. (2013), "Valid post-selection inference",
  *Annals of Statistics*, 41(2), 802–837.
- Regulation B, 12 CFR Part 1002 § 1002.9 và Appendix C — yêu cầu nêu lý do cụ thể khi từ chối.
- CFPB Circular 2022-03 (5/2022), "Adverse action notification requirements in connection with
  credit decisions based on complex algorithms".
- Brier, G. W. (1950), "Verification of forecasts expressed in terms of probability",
  *Monthly Weather Review*, 78(1), 1–3.
