# Khảo sát dữ liệu — Give Me Some Credit

Ghi chép quá trình mở `data/cs-training.csv` ra xem có gì, và các quyết định làm sạch
rút ra từ đó. Mọi con số đều chạy trực tiếp trên file, không lấy từ mô tả của Kaggle.

Lưu ý về phạm vi: phần lớn số liệu ở đây tính trên **cả 150.000 dòng, trước khi split**.
Đây là bước thăm dò để biết cần tìm gì. Riêng các bảng WOE/IV ở mục 10 và 12 vì tính trên
toàn bộ dữ liệu nên có target leakage, chỉ dùng để minh hoạ; bảng thật phải tính lại chỉ
trên `split = 'train'`. Mục 13 ghi kết quả sau khi
đã split và đối chiếu ngược lại.

---

## 1. Tổng quan

| | |
|---|---|
| File | `data/cs-training.csv`, 7,56 MB |
| Kích thước | 150.000 dòng × 12 cột |
| Đơn vị mỗi dòng | một khách hàng tại một thời điểm quan sát |
| Target | `SeriousDlqin2yrs`, bằng 1 nếu trễ 90+ ngày hoặc tệ hơn trong 2 năm |
| Bad rate | 6,684% (10.026 bad / 139.974 good) |
| Cột thời gian | không có |
| Dòng trùng feature | 1.000 dòng trong 354 nhóm |

Kaggle còn có `cs-test.csv` nhưng file đó không có nhãn nên không dùng được. Toàn bộ
train/test/OOT của project cắt ra từ 150.000 dòng này.

Về loại scorecard: các biến ở đây là hành vi tín dụng của khách đã có quan hệ tín dụng
(số lần trễ hạn 2 năm qua, tỉ lệ sử dụng hạn mức, số hạn mức đang mở), tức thứ chỉ quan sát
được sau khi khoản vay đã chạy một thời gian. Đây là behavioral scorecard, không phải
application scorecard.

Không có cột ngày kéo theo ba hệ quả: không dựng được vintage analysis, không split được
theo thời gian, và mọi thứ gọi là out-of-time trong project này chỉ là mô phỏng. Cấu trúc
thời gian mà mô tả Kaggle ngụ ý (feature nhìn lùi 24 tháng, nhãn nhìn tới 24 tháng) là một
giả định tôi chấp nhận mà không kiểm chứng được.

## 2. Mười hai cột

| # | Cột | Ý nghĩa |
|---|---|---|
| 1 | `Unnamed: 0` | Số thứ tự dòng, chạy 1..150000. Kiểm tương quan với target rồi bỏ (mục 9). |
| 2 | `SeriousDlqin2yrs` | Target. 1 = bad. |
| 3 | `RevolvingUtilizationOfUnsecuredLines` | Dư nợ chia hạn mức của tín dụng quay vòng không tài sản đảm bảo (thẻ tín dụng, thấu chi). 0,3 nghĩa là dùng 30% hạn mức. |
| 4 | `age` | Tuổi. |
| 5 | `NumberOfTime30-59DaysPastDueNotWorse` | Số lần trễ 30–59 ngày trong 2 năm qua. Cụm NotWorse để ba cột đếm không trùng nhau. |
| 6 | `DebtRatio` | Nghĩa vụ trả hàng tháng chia thu nhập gộp hàng tháng. |
| 7 | `MonthlyIncome` | Thu nhập hàng tháng, USD. |
| 8 | `NumberOfOpenCreditLinesAndLoans` | Số hạn mức và khoản vay đang mở. |
| 9 | `NumberOfTimes90DaysLate` | Số lần trễ 90+ ngày trong 2 năm qua. |
| 10 | `NumberRealEstateLoansOrLines` | Trong số trên, bao nhiêu là vay hoặc hạn mức thế chấp bất động sản. |
| 11 | `NumberOfTime60-89DaysPastDueNotWorse` | Số lần trễ 60–89 ngày trong 2 năm qua. |
| 12 | `NumberOfDependents` | Số người phụ thuộc, không tính bản thân. |

Cả mười biến dự báo đều là số, không có biến hạng mục, không có văn bản.

## 3. Thống kê từng biến

| Cột | missing | n_unique | 1% | 25% | 50% | 75% | 99% | max |
|---|---|---|---|---|---|---|---|---|
| `RevolvingUtilization` | 0 | 125.728 | 0 | 0,030 | 0,154 | 0,559 | 1,093 | 50.708 |
| `age` | 0 | 86 | 24 | 41 | 52 | 63 | 87 | 109 (min 0) |
| `DebtRatio` | 0 | 114.194 | 0 | 0,175 | 0,367 | 0,868 | 4.979 | 329.664 |
| `MonthlyIncome` | 29.731 (19,82%) | 13.594 | 0 | 3.400 | 5.400 | 8.249 | 25.000 | 3.008.750 |
| `NumberOfOpenCreditLines` | 0 | 58 | 0 | 5 | 8 | 11 | 24 | 58 |
| `NumberRealEstateLoans` | 0 | 28 | 0 | 0 | 1 | 2 | 4 | 54 |
| `NumberOfDependents` | 3.924 (2,62%) | 13 | — | 0 | 0 | 1 | 4 | 20 |

Tỉ lệ giá trị 0 ở ba cột đếm trễ hạn: 30–59 là 84,0%, 60–89 là 94,9%, 90+ là 94,4%. Mức
lệch này là nguyên nhân của vấn đề binning ở mục 10.

Không có giá trị âm ở bất kỳ cột số nào. `age` có 13 dòng trên 100 tuổi, giữ lại vì tuổi
đó vẫn có thật.

## 4. Mã 96 và 98 nằm ở mức bản ghi

Mỗi cột trong ba cột đếm trễ hạn có đúng 5 giá trị 96 và 264 giá trị 98. Điều đáng chú ý
là cùng 269 dòng đó bị đánh dấu đồng thời ở cả ba cột, 269 trên 269. Vậy đây không phải
giá trị đếm bất thường của từng biến mà là mã trạng thái của cả bản ghi, kiểu hồ sơ đang
tranh chấp hoặc bureau từ chối cung cấp.

| Nhóm | n | bad rate |
|---|---|---|
| Có 96/98 | 269 | 54,65% |
| Còn lại | 149.731 | 6,60% |

Xử lý: giữ, đánh cờ, cho thành một bin riêng khi binning. Drop hoặc impute là thay tín
hiệu mạnh nhất tập bằng giá trị của nhóm trung bình.

Không thổi phồng: 269 trên 149.999 là 0,18% dữ liệu nên đóng góp vào Gini sẽ nhỏ.

### Nghi vấn leakage chưa giải quyết được

Không xác định được mã 96/98 (và cả `NumberOfTimes90DaysLate` nói chung) được ghi nhận
trước hay sau observation point. Muốn kết luận thì cần cột thời gian hoặc tài liệu định
nghĩa feature, dataset không có cả hai.

Một bằng chứng gián tiếp nghiêng về phía hợp lệ, và chỉ là gợi ý: nếu là cờ hậu nghiệm của
default thì bad rate phải tiệm cận 100%, trong khi thực tế là 54,65%, gần một nửa nhóm không
default.

Bản đầu của mục này ghi hai bằng chứng, thêm rằng WOE của nhóm sentinel (−2,82) nằm gọn
giữa nhóm "2 lần" (−2,63) và "3–4 lần" (−3,06). Đó không phải bằng chứng thứ hai: WOE là hàm
đơn điệu của bad rate của chính bin đó, mà bad rate ba nhóm là 49,90% → 54,65% → 60,54%, nên
việc −2,82 nằm giữa hai số kia là **hệ quả số học** của dữ kiện thứ nhất viết lại bằng đơn vị
khác. Một cờ rò rỉ có bad rate 54,65% cũng sẽ rơi đúng vào chỗ đó.

Cách xử lý ở bước đánh giá: chạy model có và không có biến này rồi báo cáo chênh lệch
Gini, để chuyển một câu hỏi không trả lời được thành một giới hạn đã định lượng.

## 5. `debt_ratio` và `monthly_income` là cùng một vấn đề

`DebtRatio` là tỉ lệ nên đáng lẽ nằm quanh 0–1. Nhưng phân vị 90 là 1.267 và có 16.892
dòng lớn hơn 1.000. Tách theo tình trạng thu nhập:

| Nhóm | n | DebtRatio median | % DebtRatio > 10 |
|---|---|---|---|
| `MonthlyIncome > 0` | 118.635 | 0,29 | 0,56% |
| `MonthlyIncome` missing | 29.731 | 1.159 | 90,04% |
| `MonthlyIncome = 0` | 1.634 | 930 | 88,56% |

Giả thuyết: mẫu số hỏng nên cột chuyển thành số tiền tuyệt đối. Bằng chứng là so độ lớn:
median 1.159 ở nhóm thiếu thu nhập cùng bậc với khoản trả nợ hàng tháng thật, mà nhân ngược
`DebtRatio × MonthlyIncome` trên nhóm có thu nhập cho median 1.649 USD và phân vị 90 là
4.396.

Phải nói đúng mức phép này chứng minh được gì. Nó được thực hiện trên nhóm mà giả thuyết
không nói tới: ở nhóm có thu nhập, `debt_ratio` theo định nghĩa đã là payment/income, nên
`debt_ratio × income = payment` là một đồng nhất thức và ra một median hợp lý gần như hiển
nhiên. Nó cho biết thang tham chiếu, và không kiểm được gì về 31.365 dòng thiếu hoặc bằng
0 thu nhập. Bản đầu viết "giả thuyết đứng vững"; câu đúng là giả thuyết **tương thích với dữ
liệu và chưa có cách bác**, và cách xử lý (tách nhóm invalid ra một bin riêng) không phụ thuộc
vào việc giả thuyết đúng hay sai.

Hệ quả: 20,91% số dòng đang mang đơn vị đô la, phần còn lại mang đơn vị tỉ lệ. Nếu
binning thẳng trên cột gốc thì các bin cao sẽ toàn nhóm thiếu thu nhập, tức biến đo
"khách có khai thu nhập không" thay vì đo mức nợ. Phải tách nhóm invalid ra trước.

## 6. Missing mang thông tin

| Nhóm | bad rate |
|---|---|
| `MonthlyIncome` missing | 5,61% |
| `MonthlyIncome` có giá trị | 6,95% |
| `MonthlyIncome = 0` | 4,04% |
| `NumberOfDependents` missing | 4,56% |
| Toàn tập | 6,684% |

Người không khai thu nhập lại an toàn hơn người có khai, ngược với trực giác thông thường.

Giả thuyết nhóm hưu trí có chứng cứ ủng hộ: nhóm thiếu `monthly_income` có tuổi trung vị
57 so với 51 ở nhóm có khai, và tỉ lệ từ 65 tuổi trở lên là 30,3% so với 18,5%. Không
chứng minh dứt khoát nhưng đủ để nêu như một giả thuyết có căn cứ.

Cơ chế missing ở đây không phải MCAR, nên impute median là làm nhoè mất chênh lệch. Cách
xử lý là để missing thành một bin riêng và để WOE do dữ liệu quyết định. Tách `= 0` khỏi
`missing` vì hai nhóm có bad rate khác nhau rõ (4,04% và 5,61%).

## 7. `revolving_util` và ngưỡng rác

Ngưỡng chọn theo nghĩa của biến, không dò theo target: giá trị 10 nghĩa là dùng 1000%
hạn mức. Dò ngưỡng theo target là để nhãn dẫn dắt một quyết định tiền xử lý. Chọn xong mới
nhìn dữ liệu:

| Dải | n | bad rate |
|---|---|---|
| 1,0 – 1,2 | 2.283 | 38,20% |
| 1,2 – 1,5 | 438 | 47,72% |
| 1,5 – 2,0 | 229 | 44,54% |
| 2 – 5 | 117 | 29,06% |
| 5 – 100 | 31 | 29,03% |
| > 100 | 223 | 4,93% |

Nhóm trên 100 có bad rate thấp hơn base rate 6,684%, tức cư xử như một mẫu ngẫu nhiên và
không mang thông tin rủi ro. Đây là lỗi dữ liệu. Ngược lại nhóm 1,0–2,0 có bad rate 38–48%,
là rủi ro thật của người vượt hạn mức nên phải giữ.

**Ngưỡng thực thi rộng hơn bằng chứng ở trên, và đây là một chỗ chưa chặt.** `config.UTIL_IMPLAUSIBLE`
đặt ở 10, không phải 100. Khoảng (10; 100] có 18 dòng với bad rate 33,33%, tức đúng loại
rủi ro thật mà đoạn trên nói phải giữ, nhưng chúng vẫn bị gắn cờ. Cả nhóm bị cờ (n = 241) vì
thế có bad rate 7,05%, gần đúng base rate, do trộn 223 dòng nhiễu (4,93%) với 18 dòng rủi ro
cao. Bảng ở trên cũng gộp dải "5 – 100" nên che mất chính khoảng này.

Câu hỏi thật là phân biệt lỗi dữ liệu với rủi ro thật mà không dùng nhãn, vì dùng nhãn để
chọn ngưỡng tiền xử lý là đúng thứ bình luận trong `config.py` cấm. Có một dấu hiệu làm được việc
đó: utilization thật là thương của hai số tiền nên có nhiều chữ số thập phân, còn một giá trị
nguyên lớn thì không phải kết quả của phép chia đó.

| dải | n | % giá trị nguyên | bad rate nếu nguyên | bad rate nếu có phần thập phân |
|---|---|---|---|---|
| (1; 2] | 2.950 | 0,0% | | 40,10% |
| (2; 10] | 130 | 1,5% | 0,00% | 28,91% |
| (10; 100] | 18 | 61,1% | 0,00% | 85,71% |
| >100 | 223 | 99,6% | 4,50% | |

Chữ ký chọn mà không nhìn nhãn, rồi nhãn xác nhận: nhóm 18 dòng tách thành 11 dòng nguyên với
bad rate 0% và 7 dòng thập phân với bad rate 86%. Cơ chế khớp thêm một đường nữa: 54,55% số dòng
nguyên trong nhóm đó có thu nhập thiếu hoặc bằng 0, so với 0% ở nhóm thập phân và 20,91% ở toàn
tập. Chúng chính là những bản ghi hỏng đã bắt được ở `DebtRatio` tại mục 5, cùng một cơ chế mẫu
số hỏng, chỉ khác cột.

Nên ngưỡng 10 giữ nguyên: nâng lên 100 sẽ thả 11 dòng lỗi vào model, còn cái giá của việc
giữ là gắn cờ nhầm 7 dòng thật, tức 0,005% dữ liệu. Tiêu chí đúng hơn một ngưỡng độ lớn là chính
dấu hiệu này, nhưng đổi cách đánh cờ thì phải dựng lại toàn bộ feature và mọi con số từ khối 2 trở
đi, nên tôi ghi lại làm việc còn treo thay vì đổi giữa chừng.

Bad rate theo dải rộng hơn, dùng khi binning:

| Dải | n | bad rate |
|---|---|---|
| 0 – 0,1 | 64.404 | 1,81% |
| 0,1 – 0,3 | 28.478 | 3,14% |
| 0,3 – 0,5 | 15.830 | 5,85% |
| 0,5 – 0,7 | 11.340 | 9,57% |
| 0,7 – 0,9 | 9.879 | 14,86% |
| 0,9 – 1,0 | 16.748 | 19,40% |
| 1,0 – 1,5 | 2.721 | 39,73% |

Đơn điệu tăng rất sạch từ 1,81% lên 39,73%. Đây là biến mạnh nhất tập.

## 8. Dòng trùng lặp: giữ

1.000 dòng trong 354 nhóm (bỏ cột index, giữ cả hai bản). Trước khi drop theo phản xạ, soi
xem chúng là gì:

| | trong nhóm trùng | toàn tập |
|---|---|---|
| `RevolvingUtilization = 0` | 49,80% | 7,25% |
| `DebtRatio = 0` | 98,10% | 2,74% |
| `MonthlyIncome` missing | 85,00% | 19,82% |
| `NumberOfOpenCreditLines = 0` | 30,10% | 1,26% |

Đây là hồ sơ thin file: **27,00% dưới 25 tuổi** so với 2,02% toàn tập, chưa có thu nhập ghi
nhận, không nợ, một hoặc không có hạn mức nào. Tuổi trung vị của nhóm trùng là 52, đúng bằng
toàn tập, nên nhìn trung vị thì không thấy gì; phân bố của nó lưỡng đỉnh (p25 = 25, p50 = 52,
p75 = 70) và đỉnh trẻ mới là phần đáng nói. Hai người như vậy trùng khớp mười biến vì gần như
không có gì để phân biệt họ. Là va chạm ngẫu nhiên do dữ liệu thưa, không phải lỗi nhân bản.

Bằng chứng trực tiếp, và phải nói đúng mức nó chứng minh được bao nhiêu: 37 nhóm có nhãn mâu
thuẫn, cùng hệt mười biến nhưng một người default còn người kia không. Bản sao của cùng một hồ
sơ thì nhãn phải giống nhau, nên **37 nhóm đó chắc chắn là người khác nhau**.

Nhưng 37 nhóm chỉ chứa 145 trên 1.000 dòng trùng, nên 317 nhóm còn lại không được chứng minh
gì. Và nếu tính kỹ hơn thì con số 37 còn nghiêng về hướng ngược: dưới giả thuyết "tất cả là
người độc lập" với bad rate nhóm trùng 6,00%, số nhóm mâu thuẫn kỳ vọng là 55,1 (sd 6,7),
nên quan sát 37 nằm ở z = −2,69, tức ÍT hơn kỳ vọng một cách có ý nghĩa. Điều đó tương
thích với việc một phần các nhóm đúng là bản sao thật. Bản đầu gọi đây là "bằng chứng dứt
điểm"; câu đúng là nó dứt điểm cho 37 nhóm và không nói gì về phần còn lại.

Xoá đi là mất 646 người thật, và mất lệch hẳn về phía nhóm thin file, tức làm model mù
trước đúng nhóm nó ít dữ liệu nhất. Bad rate nhóm trùng là 6,00% so với 6,69% phần còn
lại nên giữ hay xoá cũng không đổi kết quả bao nhiêu, nhưng lý do giữ là về nguyên tắc.

**Cái giá của quyết định này, không nêu ở bản đầu:** vì split cắt ngẫu nhiên theo dòng, các
dòng có vector feature giống hệt nhau bị chia sang hai tập khác nhau. **204 trên 354 nhóm
trùng nằm ở hơn một split, và 261 dòng test/oot (0,58%)** có một bản sao feature y hệt
nằm trong train. Đó là contamination train/holdout thật, do chính quyết định giữ dòng trùng
tạo ra. Quy mô nhỏ nên tác động lên Gini gần như không đo được, nhưng phần lập luận về rò rỉ
ở mục 13 chỉ xét rò rỉ theo `id` và theo thứ tự dòng, nên đã bỏ sót đúng kênh này. Cách chặn
là split theo nhóm feature thay vì theo dòng.

## 9. Cột `id` không leak

`corr(id, target) = 0,0028`. Với n = 150.000 thì sai số chuẩn của hệ số tương quan vào
khoảng 0,0026, nên con số này bằng **1,08 sai số chuẩn** (bản đầu viết "nằm trong khoảng một
sai số chuẩn", tức hơi rộng tay). Phép kiểm chắc hơn là chi-square trên 10 khối dưới đây:
chi2 = 14,93 với 9 bậc tự do, p = 0,093, không bác bỏ được tính phẳng.

Bad rate qua 10 khối index liên tiếp: 6,47 · 6,47 · 7,13 · 6,64 · 6,61 · 6,45 · 6,95 ·
6,51 · 6,55 · 7,07. Phẳng. Bỏ cột được.

## 10. IV phụ thuộc hoàn toàn vào cách chia bin

Đây là chỗ tôi suýt mắc lỗi. Tính IV cho cả 10 biến bằng cách máy móc nhất, `pd.qcut` chia
10 phần bằng nhau, missing thành bin riêng:

```
1,1134  RevolvingUtilizationOfUnsecuredLines
0,4718  NumberOfTime30-59DaysPastDueNotWorse
0,2592  age
0,0737  DebtRatio
0,0734  MonthlyIncome
0,0669  NumberOfOpenCreditLinesAndLoans
0,0272  NumberOfDependents
0,0121  NumberRealEstateLoansOrLines
0,0000  NumberOfTimes90DaysLate
0,0000  NumberOfTime60-89DaysPastDueNotWorse
```

Hai biến ra IV bằng 0, tức "vô dụng, loại bỏ" theo bảng ngưỡng thông thường. Nguyên nhân:
`NumberOfTimes90DaysLate` có 94,4% giá trị bằng 0, nên mọi ranh giới phân vị đều rơi vào
giá trị 0; `qcut` gộp các ranh giới trùng lại và trả về đúng một bin chứa toàn bộ dữ liệu.
Một bin thì g = b = 1, WOE = ln 1 = 0, IV = 0. Con số 0 nói về phương pháp chia bin,
không nói gì về biến.

`NTILE` trong SQLite hỏng theo kiểu khác và nguy hiểm hơn. Nó không gộp ranh giới trùng
mà chia đều số dòng bất kể giá trị bằng nhau:

```
 bin     n  v_min  v_max   bad   good  bad_rate     WOE  IV_part
   1 15000      0      0   664  14336      4.43  0.4360  0.01578
   2 15000      0      0   687  14313      4.58  0.4003  0.01350
   3 15000      0      0   720  14280      4.80  0.3511  0.01060
   4 15000      0      0   681  14319      4.54  0.4095  0.01408
   5 15000      0      0   681  14319      4.54  0.4095  0.01408
   6 15000      0      0   693  14307      4.62  0.3912  0.01295
   7 15000      0      0   723  14277      4.82  0.3467  0.01036
   8 15000      0      0   679  14321      4.53  0.4126  0.01427
   9 15000      0      0   689  14311      4.59  0.3973  0.01332
  10 15000      0     98  3809  11191     25.39 -1.5585  0.46750
                                                  IV = 0.5864
```

Chín bin đầu có `v_min = v_max = 0`, cùng một giá trị duy nhất bị cắt bừa thành chín nhóm.
Chênh lệch WOE giữa chúng (0,3467 đến 0,4360) là nhiễu lấy mẫu thuần tuý. Bin 10 gom cả
phần zeros còn dư lẫn toàn bộ giá trị khác 0 nên pha loãng tín hiệu mạnh nhất của biến.

Kết quả là IV = 0,5864, một con số trông rất hợp lý và không ai nghĩ đến chuyện đi kiểm.
`qcut` hỏng ồn ào nên an toàn; `NTILE` hỏng im lặng.

Chia bin bằng `CASE` theo nghĩa nghiệp vụ:

```
       bin      n  bad   good  bad_rate     WOE  IV_part
         0 141662 6554 135108      4.63  0.3897  0.12141
         1   5243 1765   3478     33.66 -1.9580  0.29603
         2   1555  776    779     49.90 -2.6324  0.18910
       3-4    958  580    378     60.54 -3.0644  0.16900
        5+    313  204    109     65.18 -3.2630  0.06385
9_SENTINEL    269  147    122     54.65 -2.8227  0.03893
                                          IV = 0.8783
```

Ba biến, ba cách:

| Biến | `qcut(10)` | `NTILE(10)` | `CASE` |
|---|---|---|---|
| `NumberOfTimes90DaysLate` | 0,0000 | 0,5864 | 0,8783 |
| `NumberOfTime60-89DaysPastDue` | 0,0000 | — | 0,6003 |
| `NumberOfTime30-59DaysPastDue` | 0,4718 | — | 0,7593 |

IV là thuộc tính của cặp (biến, cách chia bin), không phải của riêng biến. Luôn nhìn
bảng bin trước khi tin con số IV.

Bảng `CASE` cũng trả lời luôn câu hỏi đặt nhóm sentinel ở đâu: WOE −2,82 nằm giữa nhóm
"2 lần" và nhóm "3–4 lần", tức nhóm mã đặc biệt hành xử như người đã trễ 90 ngày khoảng
hai đến ba lần. Không cần đoán.

## 11. Quy tắc chọn kỹ thuật binning

Kiểm trước khi bin: nếu số giá trị phân biệt dưới khoảng 20, hoặc một giá trị đơn lẻ chiếm
trên khoảng 10% số dòng, thì dùng `CASE`, không dùng `NTILE`.

| `NTILE` | `CASE` |
|---|---|
| `revolving_util` | `late_30_59` |
| `age` | `late_60_89` |
| `debt_ratio_valid` (*) | `late_90` |
| `monthly_income` (*) | `dependents` (57,9% bằng 0) |
| `open_credit_lines` | `real_estate_loans` |

(*) Binning hai tầng: `CASE` tách missing, zero và mã đặc biệt ra trước, rồi `NTILE` trên
phần đuôi liên tục còn lại.

## 12. Bảng WOE minh hoạ cho `age`

| age | n | bad | good | bad rate | %good | %bad | WOE | IV phần |
|---|---|---|---|---|---|---|---|---|
| ≤25 | 3.028 | 338 | 2.690 | 11,16% | 0,01922 | 0,03371 | −0,5618 | 0,00814 |
| 26–35 | 18.458 | 2.053 | 16.405 | 11,12% | 0,11720 | 0,20477 | −0,5580 | 0,04886 |
| 36–45 | 29.819 | 2.628 | 27.191 | 8,81% | 0,19426 | 0,26212 | −0,2996 | 0,02033 |
| 46–55 | 36.690 | 2.786 | 33.904 | 7,59% | 0,24222 | 0,27788 | −0,1373 | 0,00490 |
| 56–65 | 33.406 | 1.531 | 31.875 | 4,58% | 0,22772 | 0,15270 | +0,3996 | 0,02998 |
| 66–75 | 18.470 | 483 | 17.987 | 2,62% | 0,12850 | 0,04817 | +0,9812 | 0,07882 |
| >75 | 10.129 | 207 | 9.922 | 2,04% | 0,07088 | 0,02065 | +1,2333 | 0,06195 |
| | | | | | | | | IV = 0,2530 |

Đơn điệu sạch. WOE đổi dấu giữa bin 46–55 và 56–65, đúng chỗ bad rate cắt qua mức nền
6,684%.

Bin 46–55 đông dân nhất bảng (24,5% dữ liệu) nhưng đóng góp IV nhỏ nhất. Kích thước bin
quyết định trần của đóng góp, còn giá trị thực đến từ độ lệch khỏi mức nền. Ở biến `age`
hai thứ này tình cờ gặp nhau ở giữa dải giá trị; ở `revolving_util` thì ngược lại, bin
0–0,1 vừa đông nhất (42,9%) vừa đóng góp IV lớn nhất.

## 13. Sau khi split

| split | n | bad | bad rate | sentinel | income missing | util>10 | debt_ratio invalid |
|---|---|---|---|---|---|---|---|
| train | 104.999 | 7.018 | 6,6839% | 193 | 20.807 | 177 | 21.977 |
| test | 22.500 | 1.504 | 6,6844% | 32 | 4.434 | 32 | 4.664 |
| oot | 22.500 | 1.504 | 6,6844% | 44 | 4.490 | 32 | 4.724 |

Seed 42, phân tầng theo nhãn, chạy lại `build()` cho đúng cùng một split. Tổng đối chiếu
ngược với các mục trên đều khớp: 149.999 dòng sau khi bỏ 1 dòng `age = 0`, 10.026 bad,
269 sentinel, 29.731 thiếu thu nhập, 1.634 thu nhập bằng 0, 3.924 thiếu dependents, 241
util quá ngưỡng, 31.365 debt_ratio invalid.

OOT có 1.504 ca dương. Theo công thức Hanley–McNeil với AUC quanh 0,78, sai số chuẩn của
Gini ở cỡ mẫu này là 0,0144, tức khoảng tin cậy 95% rộng ±0,028. Mọi chênh lệch dưới
khoảng 0,03 Gini trên OOT phải coi là chưa kết luận được.

Một điều phải nhớ khi đọc kết quả về sau, và phải quy đúng nguyên nhân cho từng thứ. Vì đã
**phân tầng theo nhãn**, bad rate ba tập bằng nhau theo thiết kế và calibration trên OOT sẽ
đẹp. PSI thì khác: nó tính trên phân bố feature hoặc điểm, không tính trên nhãn, nên PSI
gần 0 không đến từ phân tầng mà đến từ việc cả ba tập được rút ngẫu nhiên từ cùng một tổng
thể; bỏ phân tầng đi thì PSI vẫn gần 0. Kết luận chung không đổi: cả hai là hệ quả của cách
cắt tập, không phải bằng chứng về model.

## 14. Việc còn để mở

| Việc | Ở bước |
|---|---|
| Tính lại toàn bộ WOE/IV chỉ trên `split = 'train'` | binning |
| Bin hai tầng cho `debt_ratio_valid` và `monthly_income` | binning |
| Gộp bin đầu của `revolving_util` để giữ đơn điệu, kèm lý do | binning |
| Chạy model có và không có cờ sentinel, báo cáo chênh lệch Gini | đánh giá |
| Dựng một tập OOT dịch chuyển nhân tạo để kiểm PSI phản ứng đúng ngưỡng | đánh giá |
