# Ghi chú nền tảng credit scoring

Tôi vào project này mà chưa từng làm tín dụng, nên phải dựng lại các khái niệm từ đầu
trước khi viết dòng code nào. Đây là ghi chú của quá trình đó. Các con số minh hoạ đều
đo trực tiếp trên `data/cs-training.csv` chứ không lấy từ sách.

Mục lục: bài toán PD · leakage · WOE và IV · AUC, Gini, KS · calibration · PSI ·
class imbalance · scorecard và mô hình cây · vòng vận hành · giả thuyết ghi trước.

---

## 1. Bài toán PD

### 1.1 Cấu trúc kinh tế của bài toán

Cho vay 100, thu về 100 cộng lãi khoảng 15. Khách vỡ nợ thì mất phần lớn gốc, khoảng 70
sau khi thu hồi. Tỉ lệ ăn thua lệch hẳn: cần chừng năm khách trả tốt mới bù nổi một khách
vỡ nợ.

Cái lệch đó giải thích gần như mọi thứ phía sau. Nó là lý do ngành ám ảnh với việc xếp
hạng rủi ro cho chính xác, lý do tỉ lệ xấu vài phần trăm vẫn là chuyện sống còn, và lý do
accuracy không dùng được ở đây.

### 1.2 DPD, bucket, default

Thực tế không có ranh giới rõ giữa trả và không trả. Trễ ba ngày rồi trả là chuyện ai cũng
gặp; trễ 200 ngày thì coi như mất. Ngành lượng hoá cái phổ đó bằng **DPD (days past due)**,
số ngày quá hạn tính từ ngày đến hạn thanh toán.

Kỳ đến hạn 05/03, khách chưa trả. Đến 20/03 là 15 DPD, đến 10/04 là 36 DPD. Nếu ngày 12/04
khách trả bù thì DPD về 0, gọi là **cure**.

Vì DPD là số ngày liên tục nên người ta gom thành **delinquency bucket** cho dễ báo cáo:
0 DPD (hiện hành), 1–29, 30–59, 60–89, 90+. Mỗi tháng một tài khoản hoặc đứng yên, hoặc
rơi xuống bucket nặng hơn, hoặc nhảy về 0. Đây là bộ khung ngôn ngữ chung của ngành, và
cũng là cách đọc tên cột `NumberOfTime30-59DaysPastDueNotWorse` trong dataset.

**Default** là lúc ngân hàng chính thức tuyên bố khoản này hỏng. Chuẩn Basel định nghĩa là
90+ DPD, hoặc "unlikely to pay" kể cả chưa đủ 90 ngày (phá sản, khách chết, nợ bị bán tháo).
Con số 90 không thiêng liêng, nó dựa trên bằng chứng thực nghiệm ở mục 1.5.

### 1.3 good, bad, và PD

Quy ước của ngành là **bad = 1**, tức lớp thiểu số và lớp "xấu" được gán nhãn dương. Ngược
trực giác thông thường, và nó khiến mọi phát biểu về positive rate, recall hay
`scale_pos_weight` sau này đều nói về khách xấu. Dataset theo đúng quy ước này.

**PD (probability of default)** là xác suất khách thành bad. Model xuất ra xác suất chứ
không xuất nhãn, vì ba lý do.

Ngưỡng duyệt là quyết định kinh doanh chứ không phải quyết định kỹ thuật. Cùng một model,
phòng kinh doanh muốn tăng trưởng thì hạ ngưỡng, kinh tế xấu đi thì nâng. Trả về xác suất
thì chỉ cần xoay một con số; trả về nhãn cứng thì mỗi lần đổi khẩu vị lại phải train lại.

PD cộng dồn được thành tiền. Với PD 3%, lãi 15 và lỗ 70, lợi nhuận kỳ vọng là
0,97 × 15 − 0,03 × 70 = +12,45. Với PD 20% thì 0,8 × 15 − 0,2 × 70 = −2,0. Ngưỡng hoà vốn
rơi vào PD khoảng 17,6%. Một nhãn 0/1 không cho phép phép tính này.

PD dùng để định giá. Khách rủi ro hơn chịu lãi suất cao hơn, đúng bằng phần bù rủi ro tính
từ PD.

Hệ quả: vì PD được dùng như một con số thật để tính tiền, model phải **calibrated**. Một
model có thể xếp hạng rất tốt mà vẫn sai lệch về mức độ. Xếp hạng tốt và calibrate tốt là
hai tính chất tách rời nhau, đo bằng hai loại công cụ khác nhau (mục 4 và 5).

### 1.4 Ba tham số phải chốt trước khi chạm dữ liệu

"Khách này rủi ro 8%" là câu chưa hoàn chỉnh. Đầy đủ phải là: 8% khả năng trễ 90+ ngày,
trong 24 tháng, kể từ hôm nay.

**Observation point** là thời điểm chấm điểm, mốc 0 trên trục thời gian. Luật sắt của ngành:
mọi feature chỉ được dùng thông tin tồn tại tại hoặc trước mốc này. Lý do rất vật lý, vì
khi model chạy thật thì thông tin xảy ra sau đó chưa tồn tại trên đời.

**Performance window** là khoảng theo dõi sau mốc 0 để dán nhãn. Nếu trong khoảng đó khách
có bất kỳ lúc nào chạm 90+ DPD thì dán bad.

**Bad definition** là trạng thái nào bị tính là bad.

Trục thời gian của một quan sát:

```
[ observation window: 12-24 thang ] -> MOC 0 -> [ performance window: 24 thang ]
        feature nhin ve qua khu               nhan nhin ve tuong lai
                     hai ben khong duoc cham nhau
```

Gần như mọi lỗi leakage trong tín dụng là hình này bị vi phạm.

Độ dài cửa sổ là một đánh đổi thật. Ngắn quá thì đa số khách chưa kịp xấu, và ta chỉ bắt
được nhóm vỡ nợ cực sớm, thường là gian lận hoặc sốc tài chính đột ngột chứ không phải rủi
ro tín dụng thông thường. Dài quá thì nhãn về chậm, dữ liệu build đã cũ lúc model lên
production, và cửa sổ trộn nhiều chu kỳ kinh tế vào chung một nhãn.

Có một cái bẫy ở đây đáng ghi riêng. Rút ngắn cửa sổ thường làm **Gini tăng**, vì nhãn gần
feature hơn nên ít nhiễu ngoại sinh chen vào. Càng xa mốc 0 càng nhiều biến cố mà không
feature nào tại mốc 0 dự báo được: mất việc ở tháng 14, ly hôn ở tháng 18, kinh tế xấu đi ở
tháng 20. Nhưng đó là tăng giả. Nhóm bad ở cửa sổ ngắn chủ yếu là người đã đang xấu sẵn tại
mốc 0, mà nhóm đó nhân viên tín dụng nào cũng loại được bằng một quy tắc đơn giản. Giá trị
thật của scorecard nằm ở chỗ phân biệt trong nhóm trông đều tốt như nhau. Metric tốt lên
trong khi model xấu đi.

### 1.5 Chọn ba tham số đó bằng gì

Tôi không phải làm hai phân tích này (dataset không có cột thời gian) nhưng cần biết chúng
tồn tại, vì trả lời "vì Kaggle cho sẵn" là hỏng.

**Vintage analysis** dùng để chọn độ dài performance window. Vintage nghĩa là lứa: nhóm
khách theo tháng mở tài khoản. Với mỗi lứa, đo tỉ lệ bad tích luỹ theo tuổi tài khoản, gọi
là **MOB (months on book)**, chứ không theo ngày dương lịch. Các đường cong có hình rất đặc
trưng: gần phẳng ở MOB 1–3, dốc lên ở MOB 6–15, rồi cong dần và đi ngang từ khoảng MOB
18–24. Chỗ phẳng là lúc rủi ro đã chín, kéo dài thêm cũng gần như không bắt thêm được khách
xấu nào. Với vay tiêu dùng không tài sản đảm bảo, điểm này thường rơi vào 12–24 tháng, và
đó là lý do thật đằng sau con số 2 năm của dataset.

Vintage analysis còn một công dụng thứ hai: đặt các lứa cạnh nhau ở cùng MOB, lứa nào cao
hẳn thì có gì đó đã đổi ở tháng đó, kiểu nới policy hoặc một kênh bán mới kém chất lượng.

**Roll-rate analysis** dùng để chọn ngưỡng bad. Lấy toàn bộ tài khoản đang ở một bucket
tháng này, đếm tháng sau chúng đi đâu, được một bảng chuyển trạng thái. Cái đọc từ bảng đó
là **cure rate** theo từng bucket, và điểm mấu chốt là nó giảm rất đột ngột ở một chỗ. Ở 30
DPD còn khá nhiều khách chữa được, ở 60 DPD ít hơn hẳn, đến 90 DPD thì tỉ lệ chữa được rơi
xuống rất thấp. Đó là điểm không hoàn lại, và là bằng chứng thực nghiệm cho ngưỡng 90+ DPD.
Basel chọn 90 cũng vì lý do này chứ không phải quy ước tuỳ tiện.

### 1.6 Indeterminate

Nếu bad là 90+ DPD thì khách đang ở 45 DPD là gì. Gọi là good thì oan vì rõ ràng rủi ro
hơn người chưa từng trễ; gọi là bad thì sai định nghĩa.

Ngành gọi nhóm này là **indeterminate** và thường loại khỏi tập train để hai lớp tách bạch
hơn. Nhưng khi triển khai thì vẫn phải chấm điểm họ, nghĩa là ta cố tình train trên một
quần thể hẹp hơn quần thể sẽ chấm, và phải chấp nhận rủi ro đó một cách có ý thức.

Dataset không cho lựa chọn này vì target đã nhị phân sẵn: mọi khách chưa chạm 90 DPD đều bị
gộp vào good, kể cả người đang ở 60–89 DPD. Đây là một giới hạn phải nêu.

### 1.7 Application và behavioral scorecard

Khác nhau ở chỗ đặt observation point.

**Application scorecard** đặt mốc 0 lúc khách nộp hồ sơ. Ngân hàng chưa có quan hệ gì với
người này nên feature chỉ gồm thông tin trên đơn và thông tin mua từ credit bureau (ở Việt
Nam là CIC, nơi lưu lịch sử vay và trả nợ ở tất cả tổ chức tín dụng khác). Model trả lời
"có nên duyệt không".

**Behavioral scorecard** đặt mốc 0 ở một thời điểm giữa đời khoản vay, thường là cuối mỗi
tháng, và chấm lại định kỳ. Feature là hành vi trả nợ thật của chính khách. Model trả lời
"có nên nâng hạn mức, có nên chào bán thêm, có cần gọi nhắc sớm". Behavioral thường mạnh
hơn application vì hành vi thật là tín hiệu tốt hơn lời khai.

Dataset này nghiêng hẳn về behavioral. Các cột như số lần trễ 30–59 ngày, tỉ lệ sử dụng hạn
mức quay vòng, số hạn mức đang mở đều là dữ liệu hành vi của người đã có quan hệ tín dụng.

### 1.8 Ánh xạ về dataset

`SeriousDlqin2yrs = 1` nghĩa là bad definition 90+ DPD, performance window 24 tháng,
observation point là thời điểm dữ liệu được chụp. Các cột đếm trễ hạn là số lần rơi vào
bucket đó trong 2 năm trước, và cụm NotWorse để ba cột không đếm trùng nhau.

Thứ không có là cột ngày. Không biết mốc 0 của từng khách rơi vào lúc nào, không nhóm được
thành vintage, không tách được tập theo thời gian. Mọi thứ gọi là out-of-time ở đây là mô
phỏng, và phải viết đúng như vậy.

---

## 2. Leakage

Leakage là hiện tượng thông tin không được phép có mặt lúc dự đoán lại lọt vào lúc train,
làm model trông giỏi hơn thực lực. Trong tín dụng nó có ba dạng khác nhau về bản chất chứ
không chỉ về mức độ.

### 2.1 Leakage thời gian

Feature chứa thông tin phát sinh sau observation point. Ví dụ kinh điển: số lần bị nhân
viên thu hồi nợ gọi điện, khoản vay đã được tái cơ cấu, có hồ sơ thu hồi nợ. Cả ba chỉ xảy
ra khi khách đã xấu, tức chúng gần như là nhãn được viết lại dưới dạng feature. AUC sẽ đẹp
kinh ngạc lúc backtest rồi chết ngay khi lên production, vì lúc chấm thật những cột đó rỗng
cho mọi khách.

Đây là dạng nguy hiểm nhất vì nó không lộ ra trong bất kỳ metric nào. Cả train lẫn test đều
đẹp, chỉ production là sai.

**Point-in-time correctness.** Sửa định nghĩa feature trên giấy là chưa đủ, còn phải hỏi kho
dữ liệu có lưu được giá trị tại thời điểm quá khứ hay không. Nhiều data warehouse chỉ giữ
giá trị hiện tại của một chỉ tiêu và ghi đè mỗi đêm. Khi dựng tập train hồi tố cho các mốc 0
của hai năm trước, query trả về con số tính đến hôm nay, tức đã bao trùm cả performance
window. Định nghĩa đúng, câu SQL trông vô hại, mà leakage vẫn xảy ra qua đường hạ tầng. Cách
phòng là hạ tầng phải có bảng snapshot theo tháng hoặc bảng lịch sử có `valid_from`/`valid_to`.

Một câu hỏi kèm theo: feature có tồn tại lúc chấm điểm thật không. "Số lần gọi hotline" dùng
được với behavioral vì khách đã ở trong hệ thống hai năm, nhưng vô dụng với application vì
người nộp đơn lần đầu luôn bằng 0, và feature hằng số thì không mang thông tin.

### 2.2 Leakage thống kê giữa train và test

Bất cứ thứ gì tính ra từ dữ liệu đều là tham số học được: ranh giới bin, giá trị WOE, giá
trị impute, scaler. Tính chúng trên toàn bộ dữ liệu rồi mới split thì tỉ lệ bad của test đã
tham gia vào việc quyết định các con số đó, và điểm trên test không còn là ước lượng trung
thực nữa.

So sánh mức nghiêm trọng giữa hai trường hợp hay bị nhầm là ngang nhau:

`StandardScaler` chỉ rò rỉ **phân phối biên của X**, không đụng tới y. Với hồi quy logistic
không regularization, phép chuẩn hoá affine từng biến bị hệ số và intercept hấp thụ hoàn
toàn, nên xác suất dự báo không đổi và AUC trên test **giống hệt** chứ không phải gần bằng.
Có L2 thì nghiệm đổi chút ít, nhưng đó là một nhiễu loạn chứ không phải một thiên lệch có
hướng: nó chẳng có lý do gì để làm metric đẹp lên thay vì xấu đi.

WOE thì rò rỉ chính **nhãn**. Nhãn của các dòng test đã tham gia vào việc quyết định con số
mà chính các dòng đó nhận làm giá trị feature. Thiên lệch này luôn theo hướng lạc quan. Độ
lớn phụ thuộc bin mịn đến đâu: với 10 bin trên 150k dòng, mỗi bin khoảng 1.000 ca dương nên
một dòng đóng góp cỡ một phần nghìn và thổi phồng ở mức vài phần nghìn AUC; nhưng với 50
bin, hoặc một level hiếm chỉ 30 dòng và 3 bad, thì giá trị WOE của bin đó gần như ghi nhớ
trực tiếp nhãn của mấy dòng ấy.

Trong chính quy trình WOE cũng nên phân biệt hai bước. Chia bin bằng NTILE là bước
unsupervised, cùng loại với scaler, rò rỉ nhẹ. Gán giá trị WOE cho bin là bước supervised,
và đây mới là chỗ nguy hiểm.

Đây là lý do quy trình bắt tính bảng WOE chỉ trên train rồi JOIN áp sang test và OOT.

### 2.3 Sai lệch quần thể và reject inference

Model được build trên một quần thể nhưng đem chấm cho một quần thể khác. Không hẳn là
leakage theo nghĩa hẹp nhưng hậu quả giống nhau, và là lý do PSI tồn tại (mục 6).

Dạng đặc thù của nó trong tín dụng có tên riêng. Dữ liệu lịch sử chỉ chứa khách **đã được
duyệt**; người bị từ chối không bao giờ được giải ngân nên không bao giờ có nhãn. Và họ
không bị loại ngẫu nhiên mà bị loại bởi chính policy hiện hành, tức bởi những tiêu chí ta
cũng đang định dùng để dự báo.

Hệ quả nghe hơi trái khoáy nhưng rất thật: model mới học quy luật trong vùng policy cũ cho
qua, rồi khi triển khai phải chấm cả những hồ sơ mà policy cũ từng loại, tức vùng nó chưa
từng thấy dữ liệu, và ở đó nó có thể tự tin sai. Tệ hơn, quy trình tự củng cố: policy cũ
loại nhóm X nên không có dữ liệu nhóm X nên model mới cũng không học được gì về X nên tiếp
tục loại. Ngân hàng vĩnh viễn không biết mình đang bỏ lỡ khách tốt nào.

Tập tất cả người đến nộp hồ sơ gọi là **through-the-door population**, và chỉ một phần trong
đó có nhãn. **Reject inference** là nhóm kỹ thuật gán nhãn suy đoán cho phần bị từ chối, có
ba hướng chính. *Parcelling* chấm điểm hồ sơ bị từ chối bằng model hiện có, chia thành các
dải điểm, rồi gán nhãn bad theo tỉ lệ của dải tương ứng ở nhóm được duyệt, thường nhân thêm
một hệ số phạt. *Fuzzy augmentation* nhân đôi mỗi hồ sơ bị từ chối thành hai dòng good và
bad với trọng số bằng xác suất ước lượng. Cách sạch nhất nhưng tốn tiền là mua outcome từ
bureau, tức xem người mình từ chối sau đó vay ở nơi khác và trả ra sao.

Dataset không kèm dữ liệu hồ sơ bị từ chối nên tôi ghi vào future work chứ không làm nửa
vời. Nói được vấn đề là gì và biết ba hướng xử lý mạnh hơn một implementation hình thức
trên dữ liệu không phù hợp.

---

## 3. WOE và IV

### 3.1 Vì sao cần

Hồi quy logistic giả định log-odds là hàm tuyến tính của biến đầu vào. Quan hệ thật trong
tín dụng hiếm khi tuyến tính: rủi ro theo tuổi có hình chữ U ngược, người rất trẻ rủi ro
cao, trung niên thấp, người rất già nhích lên lại vì thu nhập giảm. Nhét thẳng `age` vào
logistic là ép một đường thẳng qua một đường cong.

WOE xử lý bằng cách bỏ hẳn giá trị gốc, thay bằng một con số do chính dữ liệu chỉ ra là mức
rủi ro của nhóm đó.

### 3.2 Công thức

Trong mỗi bin, đếm số good và số bad, rồi tính hai tỉ lệ:

```
g_i = (so good trong bin i) / (TONG SO GOOD TOAN TAP)
b_i = (so bad  trong bin i) / (TONG SO BAD  TOAN TAP)

WOE_i = ln( g_i / b_i )
```

Mẫu số là tổng toàn tập chứ không phải tổng dòng trong bin. Nghĩa là `g_i` trả lời "bin này
chứa bao nhiêu phần trăm của toàn bộ người tốt", và cả hai dãy đều cộng lại bằng 1. Chúng là
hai phân phối trên cùng tập bin.

WOE dương nghĩa là bin gom nhiều người tốt hơn so với phần người xấu nó gom, tức bin an toàn
hơn mức nền. WOE âm là rủi ro hơn. WOE bằng 0 khi và chỉ khi bad rate của bin đúng bằng bad
rate quần thể.

Ví dụ với 10.000 khách, 9.300 good và 700 bad, bad rate chung 7%:

| Bin | n | good | bad | bad rate | g_i | b_i | WOE |
|---|---|---|---|---|---|---|---|
| age 21–30 | 2.000 | 1.780 | 220 | 11,0% | 0,1914 | 0,3143 | −0,4959 |
| age 51–60 | 2.500 | 2.420 | 80 | 3,2% | 0,2602 | 0,1143 | +0,8228 |

### 3.3 Đẳng thức đáng nhớ

Viết lại bằng số đếm thô, với `G_i, B_i` là số good và bad trong bin và `G, B` là tổng:

```
WOE_i = ln(G_i/G) - ln(B_i/B) = ln(G_i/B_i) - ln(G/B)
```

Vế phải là log-odds của bin trừ log-odds của toàn quần thể. Kiểm với bin 21–30:
ln(1780/220) = 2,0908, ln(9300/700) = 2,5867, hiệu bằng −0,4959, khớp.

**WOE chính là độ lệch log-odds của bin so với mức nền.** Đây là lý do sâu xa WOE hợp với
logistic: logistic mô hình hoá log-odds theo tổ hợp tuyến tính của feature, mà feature WOE
vốn đã ở đơn vị log-odds. Ta không ép mô hình tuyến tính lên một quan hệ cong nữa mà đã tự
tay duỗi thẳng quan hệ đó trước.

Kèm theo là một tính chất rất được việc khi kiểm tra: hệ số hồi quy trở nên dễ đọc. Lý tưởng
thì mọi hệ số đều gần 1 và cùng dấu; một hệ số lệch xa hoặc trái dấu là dấu hiệu bất thường
cần điều tra, thường là đa cộng tuyến giữa các biến.

### 3.4 Ba lợi ích của binning

Duỗi thẳng quan hệ phi tuyến, là lý do chính.

Nuốt outlier mà không xoá dữ liệu. `revolving_util` có giá trị lớn nhất 50.708; đưa thẳng
vào logistic thì nó kéo lệch toàn bộ ước lượng, nhưng nếu bin thì nó chỉ rơi vào bin cuối và
nhận một WOE hữu hạn như mọi bin khác.

Xử lý missing như một hạng mục có nghĩa. Trong pipeline ML thông thường phải chọn giữa xoá
dòng và impute, cả hai đều là bịa. Trong khung WOE có lựa chọn thứ ba tự nhiên hơn: missing
là một bin riêng và nhận giá trị WOE tính từ chính tỉ lệ good/bad của nhóm missing. Không
giả định gì. Với dataset này, khi `monthly_income` thiếu 19,82% và nhóm đó có bad rate thấp
hơn nhóm có khai, đây không phải tiện lợi mà là bắt buộc.

Giá phải trả là mất thông tin trong nội bộ mỗi bin. Hai người utilization 0,31 và 0,49 rơi
cùng một bin thì với model họ giống hệt nhau. Đây là đánh đổi có ý thức: hi sinh độ mịn để
lấy tính vững, tính diễn giải và khả năng kiểm toán.

### 3.5 Ba cách chia bin, bốn tiêu chí bin tốt

Chia đều theo giá trị (equal-width) gần như luôn tệ với dữ liệu tín dụng vì dữ liệu lệch
nặng: chia `monthly_income` từ 0 đến 3.008.750 thành 10 đoạn thì 99,99% dòng rơi vào đoạn
đầu. Chia đều theo số lượng (`NTILE` trong SQL, `qcut` trong pandas) ổn định hơn hẳn và là
điểm khởi đầu cho biến liên tục. Chia có nhìn nhãn (monotonic binning) chọn ranh giới sao
cho bad rate biến thiên đơn điệu, thường bằng cách gộp dần các bin kề vi phạm thứ tự.

Bốn tiêu chí:

**Đủ lớn.** Quy ước là mỗi bin chứa ít nhất 5% tổng số dòng. Bin nhỏ thì WOE ước lượng từ ít
quan sát, phương sai lớn, và rủi ro target leakage nặng hơn.

**Không bin nào rỗng một lớp.** Bin có 0 bad thì WOE bằng vô cực và model nổ. Cách chữa là
gộp với bin kề, ưu tiên hơn Laplace smoothing vì gộp giữ được tính diễn giải.

**Đơn điệu theo nghiệp vụ.** Một cái gai ở giữa bảng gần như luôn là nhiễu lấy mẫu và sẽ
không lặp lại trên test. Ở đây scorecard cố tình ép cấu trúc lên dữ liệu, không phải vì đơn
điệu đúng hơn mà vì nó vững hơn theo thời gian và giải thích được với người không làm kỹ
thuật.

**Missing và mã đặc biệt tách riêng.** Không trộn `monthly_income` thiếu vào một bin thu
nhập, không để 96/98 nằm chung với "trễ 96 lần".

### 3.6 IV

Bảng WOE cho mỗi bin một giá trị, nhưng khi phải chọn trong nhiều biến thì cần một con số
cho cả biến:

```
IV = SUM_i (g_i - b_i) * WOE_i = SUM_i (g_i - b_i) * ln(g_i/b_i)
```

IV và PSI là **đúng cùng một công thức**, đều là Jeffreys divergence tức KL đối xứng hoá. IV
đo khoảng cách giữa phân phối good và phân phối bad trên các bin; PSI đo khoảng cách giữa
phân phối lúc build và lúc chạy. Cùng một cái thước, hai cặp phân phối khác nhau. Nhớ như
vậy thì không lẫn công thức.

Ngưỡng quy ước: dưới 0,02 vô dụng, 0,02–0,1 yếu, 0,1–0,3 trung bình, 0,3–0,5 mạnh, trên 0,5
thì nghi ngờ leakage và phải điều tra. Cả năm ngưỡng là kinh nghiệm ngành chứ không có nền
tảng lý thuyết.

### 3.7 IV là thuộc tính của cặp (biến, cách chia bin)

Chi tiết và số liệu nằm ở mục 10 của `results/data_profile.md`. Tóm tắt: cùng biến
`NumberOfTimes90DaysLate`, `qcut(10)` cho IV = 0,0000, `NTILE(10)` cho 0,5864, và `CASE`
theo nghĩa nghiệp vụ cho 0,8783.

`qcut` hỏng ồn ào nên an toàn, ai nhìn thấy 0,0000 cũng đi kiểm tra. `NTILE` hỏng im lặng:
nó không gộp ranh giới trùng mà chia đều số dòng bất kể giá trị bằng nhau, tạo ra chín bin
đều chứa đúng giá trị 0, và trả về một con số IV trông rất hợp lý.

Kết luận thực hành: "biến này có IV 0,88" là câu chưa hoàn chỉnh, y hệt "khách này rủi ro
8%". Và luôn nhìn bảng bin trước khi tin con số IV.

### 3.8 Ba giới hạn của IV

**Đơn biến, mù trước tương quan.** Ba cột delinquency đều có IV cao nhưng đo cùng một thứ và
chồng chéo nặng. IV 0,88 cộng 0,76 cộng 0,60 không cho model mạnh gấp ba một biến IV 0,88.
IV dùng để sàng lọc thô chứ không để chọn bộ feature cuối.

**Ngưỡng 0,5 phải đọc theo bối cảnh.** Ngưỡng đó sinh ra cho application scorecard, nơi
feature chủ yếu là nhân khẩu học và bureau tổng hợp. Dataset này là behavioral, nơi hành vi
quá khứ dự báo hành vi tương lai rất mạnh, nên IV trên 0,5 là bình thường chứ không phải lỗi.
Riêng `NumberOfTimes90DaysLate` vẫn đáng soi kỹ nhất vì nó đo đúng cùng một sự kiện với
target, chỉ khác cửa sổ thời gian.

**IV không đo tính vững.** Một biến IV 0,4 ổn định qua thời gian giá trị hơn một biến IV 0,7
trôi liên tục. IV chụp một khoảnh khắc, PSI mới theo dõi chuyển động.

### 3.9 Ba cái bẫy công thức

**Lấy tổng dòng trong bin làm mẫu số.** `n_i` triệt tiêu và ra `ln(G_i/B_i)`, mất số hạng
nền. Điều bất ngờ là về mặt toán học nó gần như vô hại: mọi WOE bị dịch cùng một hằng số
`c = ln(G/B)`, và trong logistic thì intercept nuốt trọn phần đó. Tôi có chạy kiểm: hệ số gần
như trùng khít, xác suất dự báo lệch nhau 6e-05, và IV cũng bất biến vì `SUM(g_i - b_i) = 0`.

Cái mất là **diễn giải**. Với công thức đúng, WOE = 0 mang nghĩa xác định là bin trung tính,
và dấu cho biết bin an toàn hơn hay rủi ro hơn mức nền. Bỏ số hạng nền thì số 0 tụt xuống
nghĩa "bin có 50% bad", một mốc vô nghĩa trong quần thể 6,7% bad, và toàn bộ ngữ pháp để đọc
bảng WOE sụp đổ dù model vẫn chạy y hệt. Bài học rộng hơn: một phép biến đổi affine thường
vô hại với model tuyến tính nhưng có thể phá huỷ hoàn toàn khả năng diễn giải, mà trong một
project lấy tính diễn giải làm lý do tồn tại thì đó là phá huỷ sản phẩm.

**Lộn dấu**, tức định nghĩa `ln(b_i/g_i)`. Không phải dịch hằng số mà đảo dấu toàn bộ. Model
vẫn fit vì hệ số tự đổi dấu theo, nhưng mọi diễn giải lộn ngược và điểm số quy đổi chạy sai
chiều. Project này dùng `ln(g_i/b_i)`, WOE dương là an toàn hơn.

**Trộn mẫu số**, kiểu `ln((G_i/n_i)/(B_i/B))`. Không triệt tiêu được gì, bóp méo phi tuyến
theo kích thước từng bin, phá cả model lẫn IV, và im lặng.

---

## 4. AUC, Gini, KS

### 4.1 Vì sao accuracy vô dụng

Xây một model ngu nhất có thể trên chính dataset này: dự đoán mọi khách đều tốt, không nhìn
feature nào. Kết quả là **accuracy 93,316%** và bắt được 0 trên 10.026 khách xấu.

Accuracy cộng gộp hai loại sai lầm rất khác nhau vào một con số rồi cân chúng theo tần suất.
Khi 93,3% quần thể là good thì việc đoán đúng nhóm good chi phối hoàn toàn, còn nhóm 6,7%
duy nhất ta quan tâm gần như không có tiếng nói.

### 4.2 Confusion matrix và hai loại sai lầm

Với bad = 1 là lớp dương:

| | Dự đoán good | Dự đoán bad |
|---|---|---|
| Thực tế good | TN, duyệt đúng người tốt | FP, từ chối nhầm người tốt, mất khoảng 15 |
| Thực tế bad | FN, duyệt nhầm người xấu, mất khoảng 70 | TP, chặn đúng người xấu |

FN đắt hơn FP khoảng năm lần. Đây là lý do ngưỡng cắt không bao giờ là 0,5: 0,5 chỉ tối ưu
khi hai loại sai lầm đắt ngang nhau, điều gần như không bao giờ đúng trong tín dụng.

```
TPR = TP/(TP+FN)   trong toan bo khach XAU, bat duoc bao nhieu %
FPR = FP/(FP+TN)   trong toan bo khach TOT, tu choi nham bao nhieu %
```

Điểm mấu chốt là mỗi tỉ lệ có mẫu số riêng của lớp mình. TPR chỉ nhìn trong nhóm bad, FPR
chỉ nhìn trong nhóm good, nên **tỉ lệ mất cân bằng 6,7/93,3 không xuất hiện trong công thức**.
Đó chính là điều accuracy không làm được, và là nền của mọi metric bên dưới.

Precision thì khác: mẫu số TP + FP trộn hai lớp nên nó có bị ảnh hưởng bởi tỉ lệ mất cân
bằng.

### 4.3 ROC

Cho ngưỡng trượt từ 1 xuống 0, chấm điểm (FPR, TPR). Ở ngưỡng 1 không ai bị gắn cờ nên ta ở
(0,0); ở ngưỡng 0 mọi người bị gắn cờ nên ta ở (1,1). Mỗi bậc lên là một khách bad được xử
lý đúng thứ tự, mỗi bậc ngang là một khách good bị xếp nhầm lên cao. Model hoàn hảo ôm góc
trên trái, model ngẫu nhiên là đường chéo.

Điều quan trọng nhất về ROC, và tôi sẽ dùng lại nhiều lần: hình dạng của nó chỉ phụ thuộc
**thứ tự** điểm số, không phụ thuộc **giá trị** điểm số.

### 4.4 AUC

Hình học thì AUC là diện tích dưới ROC, 0,5 là ngẫu nhiên và 1,0 là hoàn hảo. Dưới 0,5 nghĩa
là xếp hạng ngược, đảo dấu điểm là được model tốt.

Cách hiểu đáng nhớ hơn là cách xác suất: **AUC là xác suất một khách bad lấy ngẫu nhiên có
điểm rủi ro cao hơn một khách good lấy ngẫu nhiên** (chặt chẽ thì cộng thêm nửa xác suất hai
người bằng điểm). AUC 0,78 nghĩa là bốc ngẫu nhiên một cặp gồm một người sau này vỡ nợ và
một người không, thì 78% số lần model chấm người vỡ nợ điểm cao hơn.

Cách hiểu này cho luôn công thức tính không cần vẽ đường nào, chính là thống kê Mann–Whitney U:

```
AUC = ( SUM_{i thuoc bad} rank_i  -  n1(n1+1)/2 ) / (n1 * n0)
```

Nhìn công thức là thấy chỉ có hạng xuất hiện, không có giá trị điểm số.

### 4.5 Gini

`Gini = 2 * AUC - 1`, chỉ là đổi thang tuyến tính để ngẫu nhiên thành 0 và hoàn hảo thành 1.
Đọc là model đi được bao nhiêu phần đường từ ngẫu nhiên đến hoàn hảo.

Ba lưu ý về tên, vì nó gây nhầm nhiều. Trong tín dụng Gini thường được dựng từ đường **CAP
(Cumulative Accuracy Profile)** chứ không phải ROC, và tỉ số diện tích gọi là **Accuracy
Ratio (AR)**. CAP vẽ tỉ lệ bad bắt được theo tỉ lệ dân số bị từ chối, gần ngôn ngữ vận hành
hơn. Có một kết quả đẹp là AR = 2·AUC − 1, tức bằng đúng Gini tính từ ROC. Hai đường khác
nhau, cùng một con số. Thứ hai, đây không phải Gini bất bình đẳng thu nhập của kinh tế học.
Thứ ba, tín dụng quen dùng Gini còn ML quen dùng AUC, nên trong README ghi cả hai.

Mức tham chiếu thì tôi nói dè dặt vì nó đổi mạnh theo thị trường và sản phẩm: application
scorecard thường rơi vào Gini 0,35–0,55, behavioral cao hơn. Gini 0,95 gần như chắc chắn là
leakage chứ không phải model giỏi.

Mốc sàn cho project này: một "model" một biến duy nhất (chia `revolving_util` thành 20 nhóm
rồi lấy bad rate của nhóm làm điểm) cho **AUC 0,7829, Gini 0,5658, KS 0,4575**. Model đầy đủ
không vượt rõ con số này thì có gì đó sai.

### 4.6 KS

Tên lấy từ kiểm định Kolmogorov–Smirnov, vốn để hỏi hai mẫu có cùng phân phối không. Ở đây
dựng hai CDF trên cùng trục điểm số: `CDF_good(t)` là tỉ lệ người tốt có điểm ≤ t, và
`CDF_bad(t)` tương tự cho người xấu. Vì người xấu có điểm cao hơn nên đường CDF của họ dâng
chậm hơn.

```
KS = max_t | CDF_good(t) - CDF_bad(t) |
```

Tức khoảng cách dọc lớn nhất giữa hai đường. Model càng tách bạch thì hai đường càng rời xa
nhau.

Liên hệ với ROC làm KS dễ nhớ hẳn. Với một ngưỡng t, tỉ lệ bad có điểm trên t chính là
TPR(t), nên `CDF_bad(t) = 1 - TPR(t)`, tương tự `CDF_good(t) = 1 - FPR(t)`. Trừ nhau được
`TPR(t) - FPR(t)`, mà đó đúng là khoảng cách dọc từ đường ROC xuống đường chéo. Vậy **KS là
khoảng cách dọc lớn nhất từ ROC tới đường chéo**. AUC đo diện tích giữa hai đường đó, KS đo
khoảng cách lớn nhất. Cùng một bức tranh, hai cách tóm tắt. (Đại lượng TPR − FPR còn có tên
riêng là Youden's J, và KS là cực đại của nó.)

Dân tín dụng thích KS hơn dân ML vì KS đi kèm một địa chỉ: ngưỡng nơi đạt cực đại chính là
ngưỡng tách hai nhóm tốt nhất, dùng ngay được cho vận hành. Trong một cuộc họp với phòng
kinh doanh thì "cắt ở điểm 620 thì chặn được 68% nợ xấu mà chỉ từ chối nhầm 22% khách tốt"
là câu nói được, còn "AUC của mình là 0,78" thì không.

Điểm yếu là nó chỉ là một điểm trên toàn đường cong và bỏ qua tất cả phần còn lại. Hai model
cùng KS 0,45 có thể hành xử rất khác ở những ngưỡng khác, và nếu ngưỡng vận hành thật không
trùng cực đại thì KS đang mô tả một chỗ không ai đứng. Là cực đại nên nó cũng nhiễu hơn AUC
trên mẫu nhỏ. Vì vậy báo cáo cả AUC/Gini lẫn KS chứ không thay thế nhau.

Mức tham chiếu thô: KS 0,30–0,50 là ổn, trên 0,60 nên đi kiểm tra leakage.

### 4.7 PR curve

Precision là TP/(TP+FP), tức trong số hồ sơ gắn cờ xấu thì bao nhiêu thật sự xấu; recall
chính là TPR. Đường PR vẽ precision theo recall và diện tích dưới nó gọi là average precision.

Khác biệt then chốt so với ROC là precision có mẫu số trộn hai lớp nên nhạy với tỉ lệ mất
cân bằng, trong khi TPR và FPR thì không. Hệ quả: khi lớp dương rất hiếm, ROC có thể trông
rất đẹp trong khi precision thảm hại, vì một FPR nhỏ trên 139.974 người tốt vẫn sinh ra rất
nhiều FP so với 10.026 người xấu.

Mốc so sánh của PR-AUC không phải 0,5 mà là tỉ lệ dương, tức 0,0668 ở đây.

Dùng PR thay ROC khi bài toán là lọc một danh sách hữu hạn để con người xử lý, kiểu đội điều
tra gian lận chỉ soi được 200 hồ sơ mỗi ngày. Với credit scoring cổ điển, nơi mọi hồ sơ đều
được chấm và ngưỡng đặt theo bài toán kinh tế, ROC/AUC/Gini/KS vẫn là bộ chính và PR chỉ là
bổ sung.

### 4.8 Cả ba chỉ nhìn thứ tự

Lấy model một biến ở trên rồi bóp méo điểm bằng bốn phép biến đổi đơn điệu tăng, tức không
đảo thứ tự của bất kỳ cặp nào:

```
bad rate that su                                             = 0,0668

score goc              AUC=0,7829  Gini=0,5658  KS=0,4575   PD_tb = 0,0668
score chia 10          AUC=0,7829  Gini=0,5658  KS=0,4575   PD_tb = 0,0067
can bac hai cua score  AUC=0,7829  Gini=0,5658  KS=0,4575   PD_tb = 0,2278
log cua score          AUC=0,7829  Gini=0,5658  KS=0,4575   PD_tb = -3,2045
```

Ba metric giống hệt tới bốn chữ số thập phân trong cả bốn trường hợp. Cột cuối thì loạn hoàn
toàn: PD trung bình dự báo đi từ 0,0668 xuống 0,0067, lên 0,2278, rồi thành −3,2045, thậm
chí không còn là xác suất.

Không có gì bí ẩn khi nhìn lại công thức Mann–Whitney: chỉ có hạng xuất hiện, mà biến đổi
đơn điệu theo định nghĩa không đổi hạng.

**AUC, Gini và KS đo khả năng xếp hạng và hoàn toàn mù trước việc con số PD có đúng hay
không.** Một model Gini 0,60 rất đẹp vẫn có thể nói "khách này 0,5% rủi ro" trong khi sự
thật là 5%. Với bài toán chọn ai để từ chối thì vẫn dùng được; với định giá, expected loss
hay trích lập dự phòng thì sai gấp mười lần mà không metric nào ở trên báo động.

Một chi tiết về ngưỡng đáng ghi riêng vì nó hại người trong production. **AUC bất biến vô
điều kiện**, còn **confusion matrix chỉ bất biến khi ngưỡng được dời theo đúng phép biến đổi**.
Nhân PD với 3 mà quên dời ngưỡng thì confusion matrix đổi kịch liệt trong khi AUC vẫn 0,7829,
và không chỉ số xếp hạng nào báo động. Recalibrate model là một phép đơn điệu hoàn toàn chính
đáng, nhưng nếu ngưỡng vận hành đang neo vào thang điểm cũ thì chính sách tín dụng đã đổi
trong im lặng. Ngưỡng và thang điểm là một cặp.

(Nhân xác suất với 3 thật ra không phải phép biến đổi hợp lệ vì 3 × 0,4 vượt quá 1. Đó là lý
do mọi hiệu chỉnh trong thực tế được làm trong không gian log-odds, nơi cộng trừ hằng số luôn
cho ra một xác suất hợp lệ.)

### 4.9 Sai số chuẩn, bảng phải nhớ

Công thức Hanley–McNeil, AUC quanh 0,78:

| Tập | n_bad | n_good | SE(Gini) | KTC 95% Gini |
|---|---|---|---|---|
| Toàn tập 150k | 10.026 | 139.974 | 0,0056 | ±0,011 |
| Train 70% | 7.018 | 97.982 | 0,0066 | ±0,013 |
| OOT-proxy 15% | 1.504 | 20.996 | 0,0144 | ±0,028 |

Trên OOT-proxy, mọi chênh lệch dưới khoảng 0,03 Gini phải coi là chưa kết luận được. So sánh
hai model trên cùng một tập thì dùng test cặp (DeLong) chứ không so bằng mắt.

---

## 5. Calibration

### 5.1 Định nghĩa và cách đo

Một model **calibrated** nếu trong nhóm khách được chấm PD = p thì tỉ lệ vỡ nợ thực tế đúng
bằng p. Chấm 5% cho một nghìn người thì khoảng năm mươi người trong đó phải vỡ nợ.

Đây là phát biểu về nhóm chứ không phải cá nhân. Không ai kiểm được PD của một người cụ thể
là đúng hay sai, vì người đó hoặc vỡ nợ hoặc không, 0 hoặc 1.

Cách đo là **calibration curve**, còn gọi reliability diagram: chia dự báo thành mười nhóm
theo phân vị, mỗi nhóm tính PD trung bình dự báo và bad rate thực tế quan sát, rồi vẽ cái
thứ hai theo cái thứ nhất. Calibrate hoàn hảo thì mọi điểm nằm trên đường 45°.

Chạy trên model một biến ở mục 4:

```
-- score goc --                    -- score chia 10 --
dec  PD_du_bao  bad_rate_thuc      dec  PD_du_bao  bad_rate_thuc
  0     0,0128        0,0128         0     0,0013        0,0128
  5     0,0347        0,0347         5     0,0035        0,0347
  9     0,2317        0,2317         9     0,0232        0,2317
Brier = 0,056712                    Brier = 0,064916
```

Cột `bad_rate_thực` y hệt nhau ở hai bảng vì thứ tự không đổi nên các decile chứa đúng cùng
những người. Cột `PD_dự_báo` thì thấp hơn mười lần ở mọi dòng. Model bên phải xếp hạng giỏi
ngang model bên trái và nói dối về mức độ ở mọi điểm.

**Brier score** là sai số bình phương trung bình trên xác suất, `mean((p - y)^2)`, càng nhỏ
càng tốt. Nó bắt được sự xuống cấp mà AUC/Gini/KS không thấy.

Nhưng Brier trộn hai thứ vào một số nên đừng dùng một mình. Phân rã Murphy tách ra:

```
Brier = Reliability - Resolution + Uncertainty
```

Reliability là sai lệch calibration (nhỏ thì tốt), Resolution là khả năng phân biệt (lớn thì
tốt), Uncertainty là `p_ngang * (1 - p_ngang)` và không giảm được. Với dataset này Uncertainty
bằng 0,0668 × 0,9332 = 0,0623, tức mức Brier của một model chỉ biết đoán bad rate chung cho
mọi người. Score gốc đạt 0,0567, thấp hơn, vì Resolution bù được.

Dùng Brier để theo dõi, dùng calibration curve để chẩn đoán. Brier tệ đi thì phải nhìn đường
cong mới biết hỏng ở calibration hay ở discrimination.

### 5.2 Bốn nguyên nhân mất calibration

**Train trên dữ liệu đã cân bằng lại.** `scale_pos_weight` nhân trọng số lớp dương lên w lần,
tương đương nhân bản mỗi khách bad thành w khách. Model học trong một thế giới nơi tỉ lệ bad
cao hơn thật nên xuất ra PD cao hơn thật một cách có hệ thống. SMOTE và undersampling gây
đúng vấn đề đó.

Hiệu chỉnh được, và công thức đơn giản vì độ lệch nằm gọn ở log-odds:

```
scale_pos_weight = w      ->  logit_that = logit_model - ln(w)
undersample good, giu r   ->  odds_that  = odds_model * r
```

Và phép hiệu chỉnh này là một biến đổi đơn điệu nên **không đổi AUC, Gini, KS một chữ số nào**.
Nó chỉ sửa mức.

**Regularization.** L1/L2 co hệ số về 0, kéo dự báo về phía trung bình. Xếp hạng gần như giữ
nguyên, mức bị nén.

**Bản chất thuật toán.** Logistic tối ưu log-loss nên vốn khá calibrated. Random forest có xu
hướng tránh xa 0 và 1 vì lấy trung bình nhiều cây. SVM xuất ra khoảng cách tới siêu phẳng chứ
không phải xác suất.

**Dịch chuyển quần thể.** Suy thoái nâng rủi ro của mọi người gần như đều nhau nên thứ tự giữ
nguyên và Gini vẫn đẹp, chỉ có mức là sai. Đây là loại hỏng mà PSI có thể báo trước.

### 5.3 Cách chữa và một bất đối xứng quan trọng

Platt scaling fit một logistic một chiều từ điểm model sang nhãn trên một tập giữ riêng: đơn
giản, ít tham số, khó overfit, chỉ giả định độ lệch có dạng sigmoid. Isotonic regression khớp
một hàm đơn điệu bất kỳ, linh hoạt hơn nhưng cần nhiều dữ liệu và dễ overfit. Hiệu chỉnh giải
tích như công thức `−ln(w)` dùng được khi biết chính xác đã lấy mẫu lại thế nào, sạch nhất và
không tốn dữ liệu.

Cả ba đều đơn điệu nên không bao giờ đổi Gini/AUC/KS. Từ đó ra một bất đối xứng chi phối mọi
quyết định chọn model:

**Calibration là thứ sửa được sau, xếp hạng thì không.**

Nếu model A có Gini 0,58 nhưng calibrate tệ và model B có Gini 0,52 nhưng calibrate tốt, thì
lựa chọn thật không phải A với B mà là **A-đã-hiệu-chỉnh** với B, và A-đã-hiệu-chỉnh trội hơn
ở cả hai mặt. Xếp hạng là thông tin; nếu model không có thì không hậu xử lý nào tạo ra được.

B chỉ thắng trong vài tình huống. Khi sai lệch của A **phụ thuộc phân khúc**, ví dụ calibrate
tốt cho khách trẻ và lệch cho khách già, thì không hàm đơn điệu một chiều nào chữa được, phải
recalibrate riêng từng phân khúc và làm thế là xây thêm một model nữa. Khi không còn tập giữ
riêng sạch để fit lớp hiệu chỉnh. Hoặc khi chi phí quản trị mô hình cho lớp thêm vào lớn hơn
phần Gini kiếm được, vì mỗi lớp trong pipeline phải được tài liệu hoá, thẩm định độc lập và
giám sát riêng.

Và trước hết phải hỏi chênh lệch 0,06 có vượt nhiễu không. Với cỡ OOT ở đây thì khoảng tin
cậy 95% là ±0,028, nên 0,06 có lẽ là thật nhưng phải kiểm bằng test cặp chứ không bằng mắt.

### 5.4 Dùng metric nào khi nào

| Câu hỏi | Metric |
|---|---|
| Model xếp hạng rủi ro tốt tới đâu | AUC / Gini, ghi cả hai |
| Nên cắt ngưỡng ở đâu | KS, cho địa chỉ nhưng chỉ là một điểm |
| PD có phải con số thật không | Calibration curve + Brier |
| Lọc danh sách cho người xử lý tay | PR-AUC, mốc là 0,0668 |
| Model còn dùng được không khi chưa có nhãn | PSI |
| — | Accuracy thì không bao giờ |

---

## 6. PSI và giám sát

### 6.1 Công thức và ngưỡng

```
PSI = SUM_i (a_i - e_i) * ln(a_i / e_i)
   e_i = ti trong bin i o mau THAM CHIEU (tap train luc build)
   a_i = ti trong bin i o mau MOI
```

Từng số hạng luôn không âm, nên PSI ≥ 0 và bằng 0 khi hai phân phối trùng khít. Ngưỡng quy
ước: dưới 0,1 ổn định, 0,1–0,25 theo dõi, trên 0,25 điều tra. Đây là ngưỡng kinh nghiệm,
không có nền tảng lý thuyết.

Cùng công thức với IV, tức Jeffreys divergence.

Có một xấp xỉ rất được việc. Với `a` gần `e` thì `ln(a/e) ≈ (a-e)/e`, nên
`PSI ≈ SUM (a-e)²/e`, đúng dạng thống kê chi-bình phương chia cho cỡ mẫu. Từ đó ước lượng
được PSI kỳ vọng khi hai mẫu thật sự cùng phân phối: xấp xỉ `(k-1)(1/n1 + 1/n2)`. Với k = 10
và hai mẫu 30.000 dòng thì khoảng **0,0006**.

### 6.2 Bin tham chiếu phải đóng băng

Đây là lỗi cài đặt phổ biến nhất và nó làm PSI vô dụng mà không báo lỗi.

Ranh giới bin phải tính một lần trên tập train lúc build, rồi lưu lại và dùng y nguyên mãi
mãi. Nếu mỗi kỳ giám sát lại chia decile mới trên dữ liệu mới thì `a_i` cũng luôn bằng 0,10
và PSI luôn bằng 0 dù quần thể có dịch chuyển tới đâu. Ta đang đo phân phối của dữ liệu mới
so với chính nó. Cái bẫy này im lặng tuyệt đối, dashboard xanh mượt trong khi model đang chết.

Ranh giới bin là tham số của hệ thống giám sát, và tham số thì đóng băng cùng model.

### 6.3 Hai tầng

**PSI trên điểm số** là chuông báo cháy, một con số mỗi kỳ.

**CSI (Characteristic Stability Index)** là PSI trên từng biến, dùng khi chuông đã kêu để tìm
thủ phạm. Trong scorecard có một thuận tiện đáng kể: vì feature đã là WOE, tức hàm bậc thang
trên các bin, nên CSI của một biến chính là PSI trên các bin của biến đó. Không cần công cụ
riêng, cùng một truy vấn `GROUP BY` đếm tỉ trọng từng bin.

### 6.4 Quy trình khi PSI vượt ngưỡng

Đừng train lại ngay, đó là phản xạ đắt tiền và thường sai.

Trước hết nghi ngờ đường ống dữ liệu. Một feed hỏng, một trường đổi đơn vị, một job ETL chạy
thiếu, một nguồn bureau đổi định dạng, tất cả đều tạo ra PSI vọt lên trông hệt như dịch
chuyển quần thể thật. Đây là nguyên nhân phổ biến nhất trong thực tế và cũng rẻ nhất để sửa.

Sau đó xác định dịch ở đâu bằng CSI từng biến. Một biến nhảy vọt trong khi phần còn lại yên
gần như chắc chắn là vấn đề nguồn dữ liệu của biến đó; nhiều biến cùng dịch nhẹ theo một
hướng có nghĩa thì quần thể thật sự đổi.

Rồi hỏi hiệu năng có thực sự giảm không. PSI đo đầu vào chứ không đo hiệu năng. Quần thể dịch
mà Gini giữ nguyên là chuyện thường và không nhất thiết phải làm gì.

Cuối cùng chọn mức can thiệp nhẹ nhất đủ dùng: recalibrate nếu chỉ mức PD lệch còn xếp hạng
giữ, tính lại ranh giới bin và bảng WOE nếu vài bin bị rỗng hoặc phình, build lại nếu bản
thân quan hệ giữa feature và nhãn đã đổi.

### 6.5 PSI mù trước cái gì

**PSI nhìn P(X) và không bao giờ nhìn P(y|X).** Trong toàn bộ công thức không có y ở đâu cả,
và đó chính là lý do nó dùng được khi chưa có nhãn.

Ngành đặt tên cho hai hiện tượng. **Data drift** là P(X) đổi, PSI bắt được. **Concept drift**
là P(y|X) đổi, PSI mù hoàn toàn. Ví dụ concept drift: một chương trình giãn nợ của chính phủ
làm nhóm khách có nhiều lần trễ hạn không rơi vào default nữa nên biến đếm delinquency mất
phần lớn sức phân biệt; hoặc một trường từ bureau đổi ngữ nghĩa mà vẫn giữ nguyên thang giá
trị nên phân phối giống hệt trong khi thông tin bên trong đã thành rác; hoặc Goodhart, khi
môi giới học được scorecard thích gì và huấn luyện khách trình bày đúng dáng hồ sơ đó.

Đó là lý do "PSI ổn" không bao giờ đồng nghĩa với "model còn tốt". PSI là chỉ báo sớm phía
đầu vào, đo được ngay hôm nay khi chưa có nhãn nào; Gini là chỉ báo hiệu năng, chỉ đo được
sau khi nhãn về, tức trễ đúng bằng độ dài performance window. Toàn bộ lý do PSI tồn tại là vì
phải chờ 24 tháng mới biết model còn tốt hay không, mà 24 tháng là quá dài để ngồi im. Nhưng
chính vì nó rẻ và sớm nên nó cũng nông.

### 6.6 PSI cao mà CSI phẳng: bốn cơ chế

Tôi mô phỏng để kiểm hai cơ chế đầu chứ không tin lập luận suông.

**Cộng dồn.** Ngưỡng CSI đặt cho từng biến còn điểm số cộng gộp cả bộ. Mười biến mỗi cái dịch
nhẹ và dưới ngưỡng nhưng cùng chiều thì điểm số nhận trọn tổng của chúng:

```
CSI tung bien : 0,0209  0,0210  0,0207  0,0224  0,0223  0,0224  0,0224  0,0205  0,0242  0,0217
CSI lon nhat  : 0,0242    (thap hon nguong canh bao 0,1 bon lan)
PSI diem so   : 0,2158    (gap chin lan CSI lon nhat, da vao vung canh bao)
```

**Đổi tương quan trong khi giữ nguyên marginal.** CSI đo phân phối biên của từng biến, còn
điểm số là hàm của phân phối đồng thời. Dựng hai biến chuẩn hoá, mỗi biến luôn là N(0,1) bất
kể tương quan, rồi đổi tương quan từ 0 lên 0,85:

```
CSI bien 1 : 0,00007
CSI bien 2 : 0,00010     (ca hai gan bang 0)
PSI diem so: 0,1181      (vuot nguong canh bao)
```

Điểm số là tổng hai biến nên phương sai của nó là 2(1+rho), giãn ra khi tương quan tăng. CSI
không có cách nào thấy vì nó không bao giờ nhìn hai biến cùng lúc.

**Lỗi cài đặt.** Bin của điểm số bị tính lại mỗi kỳ trong khi bin của biến thì đóng băng
(hoặc ngược lại); điểm số được chấm bằng bảng WOE phiên bản mới nhưng mốc tham chiếu giám
sát vẫn là phiên bản cũ; hoặc ai đó đổi thang điểm mà không cập nhật mốc tham chiếu. Cả ba
tạo ra PSI vọt với CSI phẳng, và cả ba không phải drift mà chỉ là bug.

**Thành phần bên trong bin đổi.** Marginal giữ nguyên nhưng bin "utilization 0,3–0,5" vẫn
chứa 12% dân số mà nay toàn khách trẻ thay vì khách trung niên. CSI chỉ đếm tỉ trọng rơi vào
mỗi bin nên bằng 0, trong khi rủi ro thật đã đổi.

Rút ra: đừng đọc CSI như một danh sách rồi kết luận không biến nào có vấn đề. Nhiều biến cùng
nhích theo một hướng có nghĩa nghiệp vụ là tín hiệu thật dù không con số nào chạm ngưỡng.

### 6.7 PSI trong project này

OOT-proxy là lát cắt ngẫu nhiên từ cùng một quần thể nên PSI sẽ ra khoảng 0,0006 theo thiết
kế. Nó không đo được drift nào cả.

Cái nó thật sự đo được có ba thứ, và cả ba đều đáng giá. Thứ nhất là tính đúng đắn của
pipeline: ranh giới bin lấy từ train có được áp nhất quán sang test và OOT không, bảng tra
WOE có JOIN đúng không. PSI ra lớn trên dữ liệu đáng lẽ đồng nhất là bug chứ không phải drift.
Thứ hai là một mức nền để có mốc so sánh, vì không biết mức nền thì ngưỡng 0,1 chỉ là con số
học thuộc. Thứ ba là bằng chứng cài đặt đúng công thức.

Kế hoạch bổ sung: dựng một tập OOT dịch chuyển có chủ ý bằng cách lấy mẫu lại thiên về nhóm
`revolving_util` cao, mô phỏng một chiến dịch mang về khách rủi ro hơn. Rồi báo cáo ba con số
cạnh nhau: PSI trên OOT ngẫu nhiên (mức nền), PSI trên OOT đã dịch (kỳ vọng vượt 0,25), và
Gini trên cả hai. Rẻ, và biến một giới hạn của dataset thành một minh chứng.

---

## 7. Class imbalance

### 7.1 Cái gì hỏng, cái gì không

Không hỏng: **AUC, Gini, KS**, vì chúng xây trên TPR và FPR, mỗi cái có mẫu số riêng của lớp
mình nên tỉ lệ lớp không xuất hiện trong công thức. Nói "AUC bị ảnh hưởng bởi mất cân bằng"
là sai.

Không hỏng: **khả năng xếp hạng của model**. Logistic tối ưu log-loss trên phân phối tự nhiên
cho hệ số dốc nhất quán. Có một kết quả cổ điển đóng đinh chuyện này: **Prentice và Pyke
(1979)** chứng minh rằng dưới case-control sampling, tức lấy mẫu lệch theo nhãn, các hệ số dốc
của logistic vẫn được ước lượng nhất quán và chỉ intercept bị lệch đúng một lượng đã biết
bằng log của tỉ lệ lấy mẫu. Nói cách khác, lấy mẫu lệch theo nhãn làm hỏng **mức** chứ không
làm hỏng **hình dạng**. Công thức `logit − ln(w)` ở mục 5.2 là hệ quả trực tiếp của định lý
này chứ không phải một mẹo vặt.

Hỏng: **accuracy**, đã nói ở mục 4.1. Hỏng: **ngưỡng mặc định 0,5**, nhưng đó là lỗi để ngưỡng
ở mặc định chứ không phải lỗi của mất cân bằng. Hỏng: **precision và F1**, vì mẫu số trộn lớp
nên không so được giữa các tập có tỉ lệ khác nhau. Hỏng: **calibration**, nếu resample.

Bị ảnh hưởng thật là **phương sai ước lượng**, và cái quyết định không phải tỉ lệ mà là **số
ca dương tuyệt đối**.

### 7.2 Events per variable

Quy tắc từ thống kê y sinh: cần khoảng 10–20 ca dương cho mỗi tham số ước lượng.

```
n_bad = 10.026, 10 bien -> EPV = 1.003   du thua 50 lan   <- project nay
n_bad =    150, 10 bien -> EPV =    15   vua sat mep
n_bad =    150, 30 bien -> EPV =     5   he so khong tin duoc
```

Nên nói cho chính xác: **6,684% với 10.026 ca dương là mất cân bằng vừa phải và xử lý được
bằng cách không làm gì đặc biệt.** Mất cân bằng nghiêm trọng là 0,1% với 150 ca dương.

### 7.3 Thực đơn xử lý

| Cách | Khi nào đúng |
|---|---|
| Không làm gì, chọn ngưỡng từ ma trận chi phí | Lựa chọn đúng cho project này và cho phần lớn bài toán tín dụng |
| Trọng số lớp / `scale_pos_weight` | Chỉ khi có bằng chứng thực nghiệm rằng nó giúp. Phá calibration, sửa bằng `−ln(w)`, hiệu ứng lên xếp hạng nhỏ và không rõ chiều |
| Undersample nhóm đa số | Khi bị chặn bởi bộ nhớ hoặc thời gian, hoặc khi sự kiện cực hiếm (mục 7.5) |
| Oversample bằng nhân bản | Tương đương trọng số lớp nhưng tốn bộ nhớ hơn |
| SMOTE | Không dùng trong tín dụng (mục 7.4) |

`scale_pos_weight` giải quyết một vấn đề project này không có. Nó sinh ra cho người dùng
ngưỡng mặc định 0,5 và metric nhãn cứng như accuracy hay F1: với dữ liệu 6,7% dương thì ngưỡng
0,5 gắn cờ gần như không ai, và trọng số lớp kéo ngưỡng ngầm về chỗ dùng được. Nhưng ở đây
ngưỡng chọn từ bài toán chi phí và metric là AUC/KS, tức vấn đề đó đã được giải quyết bằng
cách khác.

Cách xử lý đúng khi có tranh luận là **đo chứ không cãi**: train hai model, so Gini và KS trên
cùng tập validation. Ngang nhau thì lấy bản không trọng số vì nó calibrate đúng miễn phí và
không cần thêm một lớp hiệu chỉnh để giải thích với người kiểm toán. Bản có trọng số tốt hơn
rõ thì giữ và áp `−ln(w)`. Tệ hơn thì bỏ hẳn.

### 7.4 SMOTE

**Synthetic Minority Over-sampling TEchnique**, Chawla và cộng sự 2002. Thay vì nhân bản y hệt
các quan sát thuộc lớp thiểu số, nó sinh ra quan sát mới nằm giữa các quan sát thật:

1. Chọn một điểm x thuộc lớp thiểu số
2. Tìm k láng giềng gần nhất trong chính lớp thiểu số, mặc định k = 5
3. Bốc ngẫu nhiên một láng giềng x'
4. Sinh `x_new = x + lambda*(x' - x)` với lambda lấy ngẫu nhiên đều trên [0,1]

Mỗi điểm tổng hợp nằm trên đoạn thẳng nối hai điểm thiểu số thật. Ý đồ là bôi lớp thiểu số ra
một vùng thay vì chồng chất bản sao lên đúng vị trí cũ, mong ranh giới quyết định được vẽ
rộng rãi hơn. Vài biến thể: Borderline-SMOTE chỉ sinh gần ranh giới, ADASYN sinh nhiều hơn ở
chỗ khó phân loại, SMOTE-NC vá víu để xử lý biến hạng mục.

Một bẫy cài đặt phải biết: SMOTE chỉ được áp lên fold train sau khi đã split. Áp trước khi
split thì các điểm tổng hợp sinh từ những khách nằm trong test sẽ lọt vào train, tức leakage
thuần tuý và metric trên test đẹp giả ngoạn mục. Cùng họ với lỗi tính WOE trên toàn bộ dữ liệu.

Năm lý do không dùng trong tín dụng:

**Nó tạo ra người không tồn tại được.** Lấy hai khách thật từ đúng cấu trúc cột của dataset:

```
Khach A:  age 34  dependents 2  90DaysLate 0  utilization 0,85  openLines 7
Khach B:  age 61  dependents 0  90DaysLate 3  utilization 0,42  openLines 14
Tong hop (lambda = 0,4):
          age 44,8  dependents 1,2  90DaysLate 1,2  utilization 0,678  openLines 9,8
```

1,2 người phụ thuộc và 1,2 lần trễ 90 ngày. Sáu trên mười biến là số đếm nguyên, nên nội suy
tuyến tính vi phạm ràng buộc cấu trúc ở mọi điểm sinh ra.

**Nó bịa ra tự tin ở đúng chỗ ít thông tin nhất.** SMOTE lấp đầy những vùng thưa của lớp thiểu
số, mà thưa chính là vì ta quan sát được ít ở đó. Ta không thêm thông tin, ta thêm một giả
định về tính trơn cục bộ rồi để model học giả định đó như thể là dữ liệu.

**Nó phá calibration**, cùng cơ chế với oversampling.

**Nó không giải thích được với người kiểm toán.** "Ranh giới quyết định ở vùng này được định
hình bởi những khách hàng tổng hợp mà chúng tôi tự tạo ra" là câu không nói được trong một hồ
sơ model risk management.

**Trên dữ liệu bảng nó thường không cải thiện AUC.** Có một dòng nghiên cứu vài năm gần đây
trong y sinh và tài chính phản biện khá mạnh các kỹ thuật chữa mất cân bằng, luận điểm chính
là chúng làm hỏng calibration mà không cải thiện khả năng phân biệt.

SMOTE có chỗ dùng khi lớp thiểu số nằm trên một đa tạp trơn trong không gian đặc trưng, kiểu
ảnh, tín hiệu cảm biến hay embedding, vì ở đó nội suy giữa hai điểm thật cho ra một điểm hợp
lệ. Dữ liệu bảng tín dụng với số đếm và tỉ lệ có ràng buộc là trường hợp ngược lại.

### 7.5 Sự kiện hiếm (0,1%) là vấn đề khác chất

Giả sử vẫn 150.000 dòng nhưng chỉ 150 ca dương. Đây không phải cùng vấn đề nặng hơn.

**Không còn đo được gì.** Hanley–McNeil với cùng 150.000 dòng, chỉ đổi số ca dương:

| Tỉ lệ dương | n_bad | SE(Gini) | KTC 95% Gini |
|---|---|---|---|
| 6,68% | 10.026 | 0,0056 | ±0,011 |
| 1,0% | 1.500 | 0,0142 | ±0,028 |
| 0,1% | 150 | 0,0448 | ±0,088 |
| 0,033% | 50 | 0,0776 | ±0,152 |

Ở 0,1% thì Gini đo được 0,50 có thể thật sự là 0,41, cũng có thể là 0,59. Nghĩa là không phân
biệt được model tốt với model tầm thường, và mọi quyết định chọn model, thêm biến hay tuning
đều dựa trên chênh lệch nhỏ hơn nhiễu. Vấn đề đầu tiên của sự kiện hiếm không phải model học
kém mà là mất khả năng biết nó học tốt hay kém.

**Không đủ ca dương cho số tham số.** 150 ca cho tối đa khoảng 10–15 biến, bất kể kho có bao
nhiêu. Đây là ràng buộc cứng chứ không phải chuyện chỉnh siêu tham số.

**Binning sụp.** Quy tắc 5% nói về số dòng, nhưng cái quyết định là số ca dương trong bin. 150
ca chia 10 bin cho trung bình 15 ca dương mỗi bin, và bin ở đuôi có thể chỉ có ba hoặc không
có ca nào (WOE bằng vô cực). Mỗi khách bad đóng góp khoảng 1/15 vào WOE của bin mình, tức
model học thuộc lòng chứ không học quy luật.

**Logistic lệch có hệ thống ở mẫu nhỏ.** MLE của hệ số logistic nhất quán tiệm cận nhưng lệch
ở mẫu hữu hạn, và độ lệch bậc nhất tỉ lệ nghịch với số ca dương chứ không phải số dòng, chiều
lệch là ra xa 0 tức tự tin quá mức. Kèm theo là **separation**, khi một tổ hợp dự báo hoàn hảo
nhãn trên train thì hàm hợp lý không có cực đại hữu hạn và hệ số chạy ra vô cực. Cách chữa
chuẩn là **hồi quy logistic phạt Firth** (Firth 1993, Heinze–Schemper 2002), thêm một số hạng
phạt Jeffreys để khử độ lệch bậc nhất và đảm bảo nghiệm luôn hữu hạn. Đây là công cụ đúng cho
sự kiện hiếm và gần như không bao giờ được nhắc trong các bài viết về class imbalance.

**Thẩm định thành cờ bạc.** CV 5 fold với 150 ca cho 30 ca mỗi fold, sai số giữa các fold lớn
hơn hiệu ứng đang đo.

SMOTE **không** phải câu trả lời ở 0,1%, vì nó không tạo ra thông tin. Mỗi điểm tổng hợp là
tổ hợp lồi của những điểm thật, nên sau khi sinh mười nghìn điểm thì lượng bằng chứng độc lập
vẫn đúng bằng 150 quan sát. Khoảng tin cậy Gini ±0,088 không hẹp lại, EPV vẫn là 15, WOE mỗi
bin vẫn ước lượng từ vài ca dương thật, độ lệch mẫu nhỏ vẫn nguyên. Tệ hơn, nó che mất vấn đề
vì sau SMOTE mọi bin đều đầy ắp.

Năm việc thật sự giúp:

**Kiếm thêm ca dương** là câu trả lời đúng nhất và bị bỏ qua nhiều nhất. Không phải bằng cách
bịa mà bằng cách xem lại định nghĩa bài toán: kéo dài performance window, gộp nhiều năm dữ
liệu, nới bad definition (60+ thay vì 90+ DPD, được nhiều ca dương hơn hẳn đổi lấy nhãn kém
sạch hơn), hoặc dùng một target proxy. Ba cách đầu đều là quay lại chỉnh observation point,
performance window hoặc bad definition. Khi mô hình bí thì thường phải quay về định nghĩa bài
toán chứ không phải thêm thuật toán.

**Thu nhỏ model cho khớp số ca dương**: ít biến hơn, bin thô hơn, regularization mạnh hơn. Độ
phức tạp phải cân với số ca dương chứ không phải số dòng.

**Firth penalized logistic.**

**Undersample nhóm đa số**, ở đây thì chính đáng nhưng vì lý do khác với thông thường: 149.850
quan sát âm dư thừa thông tin khủng khiếp, quan sát âm thứ 100.000 gần như không thêm gì so
với thứ 10.000. Bỏ 90% nhóm âm mất rất ít thông tin mà làm bài toán tối ưu điều kiện tốt hơn.
Rồi hiệu chỉnh intercept. Đây là kết quả của **King và Zeng (2001), "Logistic Regression in
Rare Events Data"**. Không đúng ở bài toán 6,68% của project này.

**Báo cáo bằng khoảng chứ không bằng điểm.**

---

## 8. Scorecard và mô hình cây

### 8.1 Scorecard là gì

Ba lớp: binning và WOE, hồi quy logistic trên feature WOE, rồi quy đổi hệ số ra thang điểm
theo PDO và base-odds. Kết quả là một bảng cộng điểm:

```
Diem co so                              520
age trong [56, 65]                      +18
RevolvingUtilization trong [0,95 - 1,0] -47
NumberOfTimes90DaysLate = 0             +22
MonthlyIncome = missing                  +6
----------------------------------------------
Tong                                    634
```

Tính chất quyết định mọi thứ nằm ngay trong hình dạng đó: cấu trúc cộng tính, và mỗi số hạng
gắn với một khoảng giá trị cụ thể của một biến cụ thể.

### 8.2 Tính diễn giải ở đây không phải feature importance

Chỗ này hay bị nói sai. Tín dụng không cần biết "biến nào quan trọng nhất trong model", đó là
phát biểu về model. Cái cần là **"vì sao hồ sơ này bị từ chối"**, một phát biểu về một quyết
định cụ thể với một con người cụ thể.

Scorecard trả lời bằng phép trừ: điểm tối đa khách có thể đạt ở mỗi biến trừ đi điểm thực
nhận, xếp giảm dần, ra ngay danh sách lý do theo mức đóng góp. Không cần thư viện, không xấp
xỉ, không tranh cãi, vì con số đúng bằng con số đã dùng để ra quyết định.

Điều này có ràng buộc pháp lý thật. Ở Mỹ, **ECOA và Regulation B** buộc tổ chức cho vay phải
nêu lý do chính cụ thể khi từ chối tín dụng, gọi là **adverse action notice**, và lý do phải
là điều người nộp đơn hiểu và hành động được chứ không phải "mô hình chấm bạn 0,73". Nhiều
nước có quy định tương đương.

Với mô hình cây thì làm được điều tương tự bằng **SHAP**, nhưng khác về hạng: SHAP là xấp xỉ,
phụ thuộc lựa chọn baseline, tốn tính toán cho mỗi lần chấm, và cần thêm một bước ánh xạ từ
giá trị SHAP sang mã lý do đọc được. Nó làm được, nhưng là một hệ thống nữa phải xây, thẩm
định, giám sát và bảo vệ trước người kiểm toán.

### 8.3 Bốn lợi thế còn lại

**Thẩm định độc lập rẻ.** Khung quản trị rủi ro mô hình (ở Mỹ là hướng dẫn **SR 11-7**) đòi
model phải hợp lý về mặt khái niệm và thẩm định được bởi một bên độc lập. Scorecard được thẩm
định bằng cách một người ngồi đọc bảng điểm và hỏi vì sao bin này lại được cộng điểm; một
ensemble thì phải thẩm định bằng thí nghiệm.

**Vững hơn trước dịch chuyển**, vì ít tham số, bin thô, cấu trúc cộng tính.

**Giám sát rẻ và đọc được**, vì CSI theo bin rơi ra sẵn từ cấu trúc.

**Sửa cục bộ được.** Một bin có vấn đề thì gộp lại hoặc chỉnh điểm bin đó, phần còn lại giữ
nguyên. Với ensemble thì không có "phần" nào để sửa riêng.

### 8.4 Mô hình cây thắng ở đâu

Tương tác tự động, vì scorecard cộng tính không mô hình hoá được "utilization cao và thu nhập
thấp còn tệ hơn tổng của hai cái riêng lẻ" trừ khi con người tự nghĩ ra và tạo biến tương tác
bằng tay. Nhiều feature, vì với hai trăm biến thì binning thủ công là hàng tuần công. Phi
tuyến trong nội bộ bin. Và thường thắng vài điểm Gini, nhưng trên dữ liệu bảng thì chênh lệch
điển hình khiêm tốn chứ không cách biệt như hình dung phổ biến.

### 8.5 Chọn cái nào

| Bối cảnh | Chọn |
|---|---|
| Quyết định duyệt/từ chối trong tín dụng có quản lý | Scorecard, mô hình cây làm challenger |
| Ưu tiên danh sách thu hồi nợ, marketing, upsell | Mô hình cây, không phải quyết định tín dụng |
| Phát hiện gian lận | Mô hình cây, vì tương tác quan trọng và quyết định là "điều tra thêm" |
| Hàng trăm feature, không ràng buộc giải thích | Mô hình cây |

Một mẫu hình lai được dùng nhiều trong thực tế: **dùng ML để khám phá, dùng scorecard để triển
khai**. Chạy GBM, đọc SHAP để tìm biến nào và tương tác nào quan trọng, rồi mã hoá phát hiện
đó thành bin và biến mới trong scorecard. Lấy được phần lớn tín hiệu mà vẫn giữ cấu trúc
triển khai được.

---

## 9. Vòng vận hành đầy đủ

Phần lớn tài liệu chỉ mô tả nửa chấm điểm (bước 10–12). Nửa build và chỗ khép vòng mới là chỗ
phân biệt hiểu credit scoring như một hệ thống vận hành với hiểu nó như một bài toán phân loại.

| # | Chặng | Nội dung |
|---|---|---|
| 1 | Xác định bài toán | Chốt observation point, performance window (vintage analysis), bad definition (roll-rate) |
| 2 | Dựng tập build | Feature chỉ từ thông tin có tại hoặc trước mốc 0; nhãn từ những gì xảy ra trong performance window sau mốc 0; loại indeterminate |
| 3 | Tách tập | train / test / OOT trước khi tính bất cứ tham số nào, cố định seed |
| 4 | Chia bin trên train | `NTILE` cho liên tục, `CASE` cho biến đếm và mã đặc biệt, missing thành bin riêng, gộp cho đơn điệu, lưu ranh giới ra bảng |
| 5 | Tính WOE/IV trên train | `WOE = ln(g/b)`, IV để sàng lọc thô, JOIN áp bảng sang test/OOT |
| 6 | Fit logistic trên feature WOE | Kiểm hệ số cùng dấu và độ lớn quanh 1; loại biến chồng chéo |
| 7 | Quy đổi ra thang điểm | PDO / base-odds, ra bảng cộng điểm theo bin. Đây là bước biến một model thành một scorecard |
| 8 | Thẩm định | Gini/KS/AUC trên test và OOT kèm khoảng tin cậy, calibration curve, PSI mức nền, đối chiếu train để bắt overfit |
| 9 | Chọn ngưỡng vận hành | Từ ma trận chi phí, không phải 0,5 và không phải KS một cách máy móc |
| 10 | Triển khai | Đóng băng bảng bin, bảng WOE, bảng điểm và ngưỡng thành tạo tác có phiên bản |
| 11 | Chấm điểm | Hồ sơ mới tại observation point, tính feature, tra bin, tra điểm, cộng |
| 12 | Ra quyết định | Duyệt/từ chối, và cả lãi suất, hạn mức, tài sản đảm bảo. Nếu từ chối thì sinh reason code |
| 13 | Giám sát | Hằng tháng khi chưa có nhãn: PSI điểm số và CSI từng biến, đọc cùng nhau. Sau 24 tháng khi nhãn chín: Gini, KS, calibration |
| 14 | Khép vòng | Khoản vay diễn biến, 24 tháng sau sinh ra một nhãn, nhãn đó vào tập build lần sau |

Bước 14 là chỗ đáng nói nhất. **Chỉ những hồ sơ được duyệt mới sinh ra nhãn**; người bị từ
chối biến mất khỏi dữ liệu vĩnh viễn. Đó chính là selection bias ở mục 2.3, và là lý do vòng
lặp này tự củng cố: model cũ quyết định ai được vào dữ liệu của model mới.

Một chỗ dễ đặt sai: lúc **chấm điểm**, performance window chưa tồn tại, nó nằm ở tương lai.
Performance window được áp lúc **build**, lên dữ liệu lịch sử, để dán nhãn. Nó quay lại lúc
chấm điểm không phải như một khoảng thời gian mà như **ý nghĩa của con số**: vì lúc build dán
nhãn bằng cửa sổ 24 tháng nên PD hôm nay có nghĩa là xác suất trễ 90+ ngày trong 24 tháng tới.
Đổi cửa sổ lúc build là đổi ý nghĩa của mọi điểm số về sau.

---

## 10. Giả thuyết ghi trước

Ghi lại trước khi chạy, để các bước sau kiểm chứ không phải giải thích hồi tố.

| # | Giả thuyết | Kiểm ở |
|---|---|---|
| 1 | Bật và tắt `scale_pos_weight` chênh nhau dưới 0,01 Gini | Bước XGBoost |
| 2 | XGBoost vượt scorecard 0,02–0,06 Gini. Vượt trên 0,10 thì nghi scorecard làm ẩu; thua scorecard thì nghi XGBoost overfit hoặc cài sai | Bước đánh giá |
| 3 | PSI trên OOT-proxy ngẫu nhiên khoảng 0,0006; trên OOT dịch nhân tạo vượt 0,25 | Bước đánh giá |
| 4 | Mọi chênh lệch dưới 0,03 Gini trên OOT là chưa kết luận được, vì KTC 95% là ±0,028 | Bước đánh giá |
