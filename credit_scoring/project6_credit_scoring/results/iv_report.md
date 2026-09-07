# Báo cáo WOE / IV

Kết quả bước feature engineering bằng SQL. Toàn bộ số liệu tính trên `split = 'train'`
(104.999 dòng, 7.018 bad, bad rate 6,6839%), sinh ra từ `sql/features.sql`.

Chạy lại: `python src/build_features.py`

---

## 1. IV theo biến

| biến              |   số bin |     IV | mức                     |
|:------------------|---------:|-------:|:------------------------|
| revolving_util    |       11 | 1.1354 | rat manh - kiem leakage |
| late_90           |        6 | 0.8697 | rat manh - kiem leakage |
| late_30_59        |        6 | 0.7687 | rat manh - kiem leakage |
| late_60_89        |        5 | 0.598  | rat manh - kiem leakage |
| age               |       10 | 0.2545 | trung binh              |
| debt_ratio_valid  |       11 | 0.0781 | yeu                     |
| monthly_income    |       12 | 0.0748 | yeu                     |
| open_credit_lines |       10 | 0.0676 | yeu                     |
| real_estate_loans |        4 | 0.0564 | yeu                     |
| dependents        |        5 | 0.0364 | yeu                     |

Bốn biến vượt ngưỡng 0,5. Ngưỡng đó sinh ra cho application scorecard, nơi feature chủ yếu
là nhân khẩu học và bureau tổng hợp. Bộ dữ liệu này là behavioral, feature là hành vi trả nợ
thật của chính khách, nên hành vi quá khứ dự báo hành vi tương lai rất mạnh là chuyện bình
thường chứ không phải dấu hiệu lỗi.

Riêng `late_90` vẫn đáng soi kỹ nhất vì nó đo **đúng cùng một sự kiện** với target, chỉ khác
cửa sổ thời gian: feature đếm số lần 90+ DPD trong 2 năm trước, target đánh dấu 90+ DPD
trong 2 năm sau. Nếu hai cửa sổ tách bạch thì hợp lệ; nếu người dựng dataset ghép chồng lấn
thì đây là leakage thời gian. Không kiểm chứng được vì thiếu cột ngày, nên sẽ đo mức ảnh
hưởng ở bước đánh giá thay vì giả định.

## 2. Cách chia bin

Biến liên tục lấy điểm cắt từ `NTILE(10)` **trên train**, rồi khử trùng: dùng cận trên của
từng nhóm NTILE làm điểm cắt và `DISTINCT` chúng, để mọi dòng có cùng giá trị luôn rơi vào
cùng một bin. Không làm bước này thì `NTILE` cắt ngang giữa các giá trị bằng nhau, đúng lỗi
đã ghi ở mục 10 của `data_profile.md`.

Biến rời rạc và mọi giá trị đặc biệt (missing, zero, sentinel, ngoài ngưỡng) dùng `CASE`.

Điểm cắt thu được:

- `age` (9 điểm cắt → 10 bin): 33, 39, 44, 48, 52, 56, 60, 65, 72
- `debt_ratio_valid` (9 điểm cắt → 10 bin): 0.0211457, 0.106278, 0.174064, 0.234353, 0.292338, 0.354823, 0.428545, 0.529567, 0.724425
- `monthly_income` (9 điểm cắt → 10 bin): 2200, 3050, 3900, 4600, 5450, 6400, 7553, 9165, 11666
- `open_credit_lines` (9 điểm cắt → 10 bin): 3, 4, 5, 6, 8, 9, 10, 12, 15
- `revolving_util` (9 điểm cắt → 10 bin): 0.00285879, 0.019071, 0.0433193, 0.0828247, 0.15361, 0.269911, 0.441917, 0.692708, 0.97729

### Có ép đơn điệu không

Không. Quyết định dựa trên 5-fold cross-validation **bên trong train**, model một biến, so sánh
theo cặp trên cùng fold, `d = Gini(giữ nguyên hình) − Gini(ép đơn điệu)`:

| biến | chiều ép | d trung bình | SE | t | KTC 95% của d | 5 fold cùng dấu |
|---|---|---|---|---|---|---|
| `revolving_util` | giảm | +0,0063 | 0,0018 | 3,43 | [+0,0012; +0,0114] | có |
| `debt_ratio_valid` | giảm | +0,0146 | 0,0051 | 2,85 | [+0,0004; +0,0289] | có |
| `open_credit_lines` | tăng | +0,0176 | 0,0040 | 4,38 | [+0,0065; +0,0288] | có |
| `open_credit_lines` | giảm | +0,1257 | 0,0062 | 20,35 | [+0,1086; +0,1429] | có |
| `real_estate_loans` | tăng | +0,0205 | 0,0027 | 7,53 | [+0,0130; +0,0281] | có |
| `real_estate_loans` | giảm | +0,1005 | 0,0047 | 21,49 | [+0,0875; +0,1135] | có |
| `monthly_income` | tăng | +0,0003 | 0,0004 | 0,72 | [−0,0008; +0,0014] | không |
| `age` | tăng | 0 | — | — | — | PAVA không gộp gì |

**Bảng này đã được chạy lại.** Bản đầu ghi +0,0062 / +0,0172 / +0,0214 với t từ 4,13 đến 5,28,
và những con số đó **không tái lập được** bằng code hiện tại ở bất kỳ seed nào tôi thử. Chúng
được sinh ra bởi một phiên bản `cv_check.py` trước hai lần sửa lỗi ở khối 3, và tôi không khôi
phục được phiên bản đó để nói chính xác lỗi nào gây ra chênh lệch. Bài học ghi lại vì nó tốn
của tôi nhiều nhất trong cả dự án: **một bảng số trong báo cáo phải sinh ra từ code đang nằm
trong repo, không phải từ một lần chạy trong quá khứ.** Từ khối 3 trở đi mọi bảng đều được đối
chiếu lại với output notebook trước khi commit; bảng này thì không, và nó lọt.

Hai biến hình chữ U có **hai** dòng vì chiều ép không hiển nhiên, và đó là điểm chính: ép sai
chiều tốn gấp 6 đến 7 lần ép đúng chiều. Với `open_credit_lines` ép chiều giảm còn đưa Gini
đơn biến xuống **−0,0032**, tức phá huỷ hẳn biến. Bản đầu của bảng này ghi một con số cho
`open_credit_lines` mà không nói chiều nào, và thiếu hẳn `real_estate_loans` là biến chữ U mạnh
nhất, trong khi kết luận ngay dưới lại phủ lên cả bốn biến.

Bốn biến có quan hệ không đơn điệu với target, nhưng **không cùng một hình dạng**, và tôi
đã mô tả sai chỗ này ở bản đầu nên ghi lại cho đúng. Bad rate theo bin, đọc từ `woe_lookup`:

| biến | bad rate theo bin | hình |
|---|---|---|
| `open_credit_lines` | 10,77 · 6,49 · 6,26 · 5,55 · **5,27** · 5,88 · 6,12 · 5,96 · 6,33 · 6,70 | **chữ U thật**, đáy ở bin 5, nhánh trái sâu |
| `real_estate_loans` | 8,33 · **5,23** · 5,60 · 8,46 | **chữ U thật**, hai đầu cao gần bằng nhau |
| `revolving_util` | 2,55 · **1,27** · 1,38 · 1,83 · 2,42 · 3,42 · 5,27 · 8,64 · 16,47 · 23,57 | **đơn điệu tăng, có một cái móc ở bin đầu** |
| `debt_ratio_valid` | **4,91** · 6,95 · 6,49 · 6,02 · 5,01 · 5,43 · 6,30 · 7,58 · 9,41 · 11,62 | **tăng dần, nửa dưới dao động**; bin thấp nhất là bin an toàn nhất |

Chỉ hai biến đầu là chữ U. `revolving_util` tăng đơn điệu từ bin 2 đến bin 10, bad rate đi từ
1,27% lên 23,57%, và chỉ có một cái móc ở bin 1 (2,55% so với 1,27%). `debt_ratio_valid` thì
không phải chữ U chút nào: bin 1 có bad rate **thấp nhất bảng**, tức đầu thấp của biến là đầu
an toàn chứ không phải một đầu rủi ro; cái không đơn điệu ở đây là một dao động trong nửa dưới,
không phải hai nhánh.

Để tách gai thật khỏi nhiễu lấy mẫu, tôi so từng cặp bin kề nhau bằng kiểm định z hai tỉ lệ,
`z = (p2 − p1) / sqrt(p1(1−p1)/n1 + p2(1−p2)/n2)`. Cái móc ở bin 1 của `revolving_util` cho
z = −6,77, cách xa ngưỡng nhiễu. Nhưng ba điều phải nói kèm, nếu không thì mấy con số này bị
dùng quá tay. Thứ nhất, tôi chạy 45 phép so sánh mà không hiệu chỉnh đa so sánh, nên vài kết
quả quanh |z| ≈ 2 có thể là may rủi (Benjamini & Hochberg 1995). Thứ hai, mọi phép so mà một
đầu là "bin có bad rate thấp nhất" đều thiên lệch, vì cực tiểu của mười ước lượng nhiễu nằm
thấp hơn cực tiểu thật một cách có hệ thống (Berk và cộng sự 2013); tôi đã bỏ các con số dạng
đó khỏi báo cáo. Thứ ba, cái **không** phải vấn đề: ranh giới bin lấy từ phân vị của biến, không
nhìn vào nhãn, nên bản thân các bin không bị chọn theo target.

Vì vậy z ở đây chỉ là sàng lọc mô tả. Căn cứ để quyết định là bảng CV bên dưới, vì nó đo trực
tiếp thứ cần đo là khả năng tổng quát hoá, thay vì hỏi một câu gián tiếp về từng cặp bin.

### Cơ chế đằng sau từng hình dạng

Ban đầu tôi gán cho cả bốn cùng một câu chuyện, rằng hồ sơ mỏng và hồ sơ quá tải đều rủi ro
hơn hồ sơ trung bình. Đi kiểm thì câu chuyện đó chỉ đúng ở một biến. Chân dung các nhóm, đo
trên train:

| nhóm | n | bad rate | % từng trễ hạn | số hạn mức (median) | thu nhập (median) | utilization (median) |
|---|---|---|---|---|---|---|
| `open_credit_lines` 0–3 | 15.417 | 10,77% | 22,8% | 2 | 3.333 | 0,439 |
| `open_credit_lines` 8–9 (đáy) | 16.821 | 5,40% | 19,2% | 8 | 5.750 | 0,137 |
| `open_credit_lines` ≥15 | 12.226 | 6,85% | 23,0% | 17 | 7.166 | 0,166 |
| `real_estate_loans` = 0 | 39.283 | 8,33% | 21,3% | 5 | 3.967 | 0,169 |
| `real_estate_loans` ≥3 | 6.984 | 8,46% | 22,0% | 12 | 9.032 | 0,171 |
| `revolving_util` bin 01 | 10.483 | 2,55% | 12,4% | 6 | 5.053 | ~0 |
| `revolving_util` bin 02 | 10.483 | 1,27% | 8,6% | 8 | 5.318 | 0,010 |
| `debt_ratio_valid` thấp nhất 10% | 8.303 | 4,91% | 11,7% | 4 | 4.327 | 0,029 |

Đọc ra bốn kết luận khác nhau.

`open_credit_lines` là chỗ duy nhất câu chuyện thin file đứng vững, và còn mạnh hơn tôi tưởng:
nhánh trái có ít hạn mức, thu nhập thấp hơn 40%, 30,6% không khai thu nhập, không bất động sản,
**và utilization median 0,439 so với 0,137 ở đáy**. Không phải chỉ mỏng hồ sơ mà còn đang dùng
sát hạn mức ít ỏi họ có. Nhánh phải thì ngược lại hoàn toàn: thu nhập cao nhất nhóm, hai bất
động sản, không hề bị hạn chế tín dụng. Cái chung của hai nhánh là **tỉ lệ từng trễ hạn**,
22,8% và 23,0% so với 19,2% ở đáy.

`real_estate_loans` cũng hai cơ chế khác nhau: nhóm không có bất động sản có thu nhập trung vị
3.967 và 22,8% không khai thu nhập; nhóm từ ba khoản trở lên có thu nhập 9.032, cao nhất bảng.
Một bên là ít tài sản, bên kia là nhà đầu tư dùng đòn bẩy.

`revolving_util` bin 01 **không phải thin file**. Trung vị 6 hạn mức đang mở và không một ai có
0 hạn mức, tức đây là những người có lịch sử tín dụng đầy đủ. Khác biệt thật so với bin 02 nằm
ở tiền sử trễ hạn, 12,4% so với 8,6%. Cách đọc hợp lý hơn là utilization gần 0 gộp hai nhóm rất
khác nhau: người trả hết nợ mỗi tháng, và người đã bị cắt hoặc đóng hạn mức. Nhưng đây vẫn là
suy đoán, dữ liệu chỉ cho thấy tương quan với tiền sử trễ hạn chứ không cho thấy nguyên nhân.

`debt_ratio_valid` bin thấp nhất là nhóm **an toàn nhất** chứ không phải một đầu rủi ro: chỉ
11,7% từng trễ hạn so với 19,6% ở giữa, tuổi trung vị 60, utilization 0,029. Không có gì cần
giải thích ở đầu này.

### Vì sao vẫn giữ nguyên hình dạng

Không phải vì kể được câu chuyện. Với `revolving_util` tôi không kể được, và với
`debt_ratio_valid` thì hình dạng còn khác cả cái tôi tưởng.

Lý do là kết quả CV ở bảng trên: giữ nguyên hình cho Gini cao hơn ở mọi biến chữ U, cả năm fold
cùng dấu. Đọc `d` chứ không đọc `t`: `t` của phép so cặp 5 fold chỉ có 4 bậc tự do, và đổi seed
thì nó chạy từ 2,6 đến 6,3 trong khi `d` gần như đứng yên. Ở đây `d` nằm trong khoảng 0,006 đến
0,021 tuỳ biến, và cả năm fold cùng dấu ở mọi biến trừ `monthly_income`. Đó là bằng chứng thực nghiệm, không phụ thuộc vào việc có diễn giải được hay
không. Trong tài liệu và khi trình bày phải nói đúng như vậy, chứ không mượn một câu chuyện
nghiệp vụ chưa kiểm chứng để biện minh cho một quyết định đã đo được.

Chỗ này có một cái giá thật ở bước triển khai. Với biến đơn điệu, mỗi biến ứng với một chiều
lý do khi từ chối hồ sơ. Với chữ U, cùng một biến cần hai mã lý do khác nhau tuỳ khách nằm ở
nhánh nào, nên mã lý do phải gắn vào **bin** thay vì vào **biến**. Làm được, nhưng là thêm một
tầng phải tài liệu hoá và thẩm định. Riêng cái móc ở bin 01 của `revolving_util` thì hiện chưa
có lý do nào nói được với khách hàng.

> **Cập nhật sau khối 3: kết luận của mục này đã bị bác.** Bảng CV ở đây đo bằng model **một
> biến**. Khối 3 đo lại trong model đa biến, và ép đơn điệu *đúng chiều* lên cả bốn biến đều
> tốn 0,000 Gini, tức không hình dạng nào trong số này đáng giữ. Ngoài ra chiều của phép ép
> hoá ra quan trọng hơn bản thân nó: ép sai chiều `revolving_util` mất 0,0513 Gini.
> `open_credit_lines` bị loại khỏi model cuối vì đóng góp biên bằng không. Kết quả của hai giả
> thuyết ngay dưới đây, cùng toàn bộ số liệu, ở `results/scorecard.md` §3.

### Hai giả thuyết ghi trước cho bước dựng scorecard

1. Trong model đa biến, gộp bin 01 của `revolving_util` vào bin 02 sẽ làm Gini giảm **dưới
   0,005**, vì nguyên nhân của cái móc đó (tiền sử trễ hạn) đã nằm sẵn trong ba biến `late_*`.
   Nếu đúng thì gộp, và vấn đề mã lý do biến mất mà không mất gì.
2. Chữ U của `open_credit_lines` và `real_estate_loans` **sống sót** trong model đa biến, vì
   cơ chế của chúng (thu nhập, mức độ bị hạn chế tín dụng, đòn bẩy bất động sản) không nằm sẵn
   trong biến nào khác.

Với `monthly_income` thì gộp hay không cho kết quả như nhau (t = 0,01) nên giữ 10 bin cho
đồng nhất.

## 3. Đối chiếu SQL với pandas

WOE và IV được cài đặt lại độc lập bằng pandas rồi so từng bin:

```
so bin           : 80 (SQL) vs 80 (pandas), khop het
lech so dong     : 0
lech so bad      : 0
lech WOE lon nhat: 4,9e-07   (do SQL lam tron 6 chu so)
lech IV lon nhat : 0,0000
```

### Một lỗi phép kiểm tự bắt được

Bản pandas đầu tiên tính tổng good/bad từ bảng **long** thay vì bảng gốc. Bảng long có 10 dòng
cho mỗi dòng gốc, một dòng mỗi biến, nên tổng bị gấp 10 lần và `pct_good`, `pct_bad` đều bị
chia 10.

Vì `WOE = ln(pct_good / pct_bad)` là một **tỉ số**, sai số ở mẫu số chung triệt tiêu hoàn toàn:
WOE vẫn khớp tới 4,9e-07. Nhưng `IV = Σ(pct_good − pct_bad)·WOE` dùng **hiệu**, nên IV sai
đúng 10 lần ở cả mười biến.

Điều đáng ghi lại không phải bản thân lỗi mà là chuyện WOE khớp hoàn hảo **không** chứng minh
được IV đúng. Một phép kiểm không thể fail ở đại lượng mình cần kiểm thì không phải phép kiểm,
và ở đây chỉ có việc so IV riêng ra mới lộ.

## 4. Kiểm tra tính toàn vẹn

```
features_woe          : 149.999 dong  (= so dong bang applications)
row_bins              : 1.499.990 dong (= 149.999 x 10 bien)
dong co WOE = NULL    : 0
```

Con số cuối là phép kiểm quan trọng nhất. Bước 6 của `features.sql` dùng `LEFT JOIN` chứ
không `INNER JOIN` cố ý: nếu một bin xuất hiện ở test hoặc OOT mà train chưa từng thấy thì
WOE sẽ là NULL và lộ ra ngay, thay vì dòng đó biến mất khỏi kết quả trong im lặng.

## 5. Đa cộng tuyến

VIF cao nhất 1,30, tương quan cặp lớn nhất 0,35 (giữa `late_60_89` và `late_90`). Không có
vấn đề.

Điều này ngược với dự đoán ban đầu của tôi. Ba biến `late_*` dùng chung bin sentinel nên tôi
tưởng thông tin đó bị đếm ba lần, nhưng sentinel chỉ có 193 dòng train (0,18%) và các bin còn
lại chứa những nhóm người khác nhau: người trễ 30–59 ngày không phải người trễ 90+ ngày.

Chỗ chồng lấn thật nằm ở `debt_ratio_valid` và `monthly_income`: bin `X_INVALID` của biến đầu
đúng bằng hợp của `X_MISSING` và `X_ZERO` của biến sau, tức 31.365 dòng (21% dữ liệu) mang
thông tin giống nhau ở hai biến. Tương quan giữa chúng là 0,26, đủ nhỏ để giữ cả hai, nhưng
là chỗ cần nhìn lại nếu hệ số hồi quy có dấu lạ ở bước sau.

## 6. Bảng WOE đầy đủ

### `age`  —  IV = 0.2545

|   bin |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|------:|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
|    01 | 12007 |  10624 |  1383 |   11.435 |       11.518 | 0.108429 | 0.197065 | -0.597435 |  0.052954 |
|    02 | 10401 |   9410 |   991 |    9.906 |        9.528 | 0.096039 | 0.141208 | -0.385482 |  0.017412 |
|    03 | 11139 |  10196 |   943 |   10.609 |        8.466 | 0.104061 | 0.134369 | -0.255611 |  0.007747 |
|    04 | 10319 |   9478 |   841 |    9.828 |        8.15  | 0.096733 | 0.119835 | -0.214158 |  0.004947 |
|    05 | 10402 |   9587 |   815 |    9.907 |        7.835 | 0.097846 | 0.11613  | -0.17132  |  0.003132 |
|    06 |  9918 |   9292 |   626 |    9.446 |        6.312 | 0.094835 | 0.089199 |  0.061263 |  0.000345 |
|    07 |  9352 |   8896 |   456 |    8.907 |        4.876 | 0.090793 | 0.064976 |  0.334569 |  0.008638 |
|    08 | 11443 |  10974 |   469 |   10.898 |        4.099 | 0.112001 | 0.066828 |  0.516386 |  0.023327 |
|    09 | 10018 |   9742 |   276 |    9.541 |        2.755 | 0.099427 | 0.039327 |  0.927506 |  0.055743 |
|    10 | 10000 |   9782 |   218 |    9.524 |        2.18  | 0.099836 | 0.031063 |  1.16751  |  0.080293 |

### `debt_ratio_valid`  —  IV = 0.0781

| bin       |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:----------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 01        |  8303 |   7895 |   408 |    7.908 |        4.914 | 0.080577 | 0.058136 |  0.326422 |  0.007325 |
| 02        |  8303 |   7726 |   577 |    7.908 |        6.949 | 0.078852 | 0.082217 | -0.041791 |  0.000141 |
| 03        |  8302 |   7763 |   539 |    7.907 |        6.492 | 0.07923  | 0.076803 |  0.031113 |  7.6e-05  |
| 04        |  8303 |   7803 |   500 |    7.908 |        6.022 | 0.079638 | 0.071245 |  0.11136  |  0.000935 |
| 05        |  8301 |   7885 |   416 |    7.906 |        5.011 | 0.080475 | 0.059276 |  0.305737 |  0.006481 |
| 06        |  8303 |   7852 |   451 |    7.908 |        5.432 | 0.080138 | 0.064263 |  0.220761 |  0.003505 |
| 07        |  8301 |   7778 |   523 |    7.906 |        6.3   | 0.079383 | 0.074523 |  0.063178 |  0.000307 |
| 08        |  8302 |   7673 |   629 |    7.907 |        7.576 | 0.078311 | 0.089627 | -0.134964 |  0.001527 |
| 09        |  8302 |   7521 |   781 |    7.907 |        9.407 | 0.07676  | 0.111285 | -0.371416 |  0.012823 |
| 10        |  8302 |   7337 |   965 |    7.907 |       11.624 | 0.074882 | 0.137504 | -0.607738 |  0.038058 |
| X_INVALID | 21977 |  20748 |  1229 |   20.931 |        5.592 | 0.211755 | 0.175121 |  0.189954 |  0.006959 |

### `dependents`  —  IV = 0.0364

| bin       |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:----------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 0         | 60920 |  57360 |  3560 |   58.02  |        5.844 | 0.58542  | 0.507267 |  0.143291 |  0.011199 |
| 1         | 18377 |  16998 |  1379 |   17.502 |        7.504 | 0.173483 | 0.196495 | -0.124558 |  0.002866 |
| 2         | 13625 |  12525 |  1100 |   12.976 |        8.073 | 0.127831 | 0.15674  | -0.203879 |  0.005894 |
| 3+        |  9324 |   8467 |   857 |    8.88  |        9.191 | 0.086415 | 0.122115 | -0.345802 |  0.012345 |
| 9_MISSING |  2753 |   2631 |   122 |    2.622 |        4.432 | 0.026852 | 0.017384 |  0.434803 |  0.004117 |

### `late_30_59`  —  IV = 0.7687

| bin        |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:-----------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 0          | 88270 |  84758 |  3512 |   84.067 |        3.979 | 0.865045 | 0.500427 |  0.547319 |  0.199562 |
| 1          | 11189 |   9479 |  1710 |   10.656 |       15.283 | 0.096743 | 0.243659 | -0.92371  |  0.135708 |
| 2          |  3173 |   2338 |   835 |    3.022 |       26.316 | 0.023862 | 0.11898  | -1.60668  |  0.152824 |
| 3-4        |  1771 |   1103 |   668 |    1.687 |       37.719 | 0.011257 | 0.095184 | -2.13479  |  0.179166 |
| 5+         |   403 |    218 |   185 |    0.384 |       45.906 | 0.002225 | 0.026361 | -2.47216  |  0.059668 |
| 9_SENTINEL |   193 |     85 |   108 |    0.184 |       55.959 | 0.000868 | 0.015389 | -2.87577  |  0.041761 |

### `late_60_89`  —  IV = 0.5980

| bin        |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:-----------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 0          | 99756 |  94658 |  5098 |   95.007 |        5.11  | 0.966085 | 0.726418 |  0.285127 |  0.068336 |
| 1          |  3936 |   2715 |  1221 |    3.749 |       31.021 | 0.027709 | 0.173981 | -1.83717  |  0.268727 |
| 2          |   778 |    389 |   389 |    0.741 |       50     | 0.00397  | 0.055429 | -2.6363   |  0.13566  |
| 3+         |   336 |    134 |   202 |    0.32  |       60.119 | 0.001368 | 0.028783 | -3.04672  |  0.083527 |
| 9_SENTINEL |   193 |     85 |   108 |    0.184 |       55.959 | 0.000868 | 0.015389 | -2.87577  |  0.041761 |

### `late_90`  —  IV = 0.8697

| bin        |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:-----------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 0          | 99149 |  94551 |  4598 |   94.429 |        4.637 | 0.964993 | 0.655172 |  0.387223 |  0.11997  |
| 1          |  3662 |   2438 |  1224 |    3.488 |       33.424 | 0.024882 | 0.174409 | -1.94724  |  0.291164 |
| 2          |  1109 |    570 |   539 |    1.056 |       48.602 | 0.005817 | 0.076803 | -2.58038  |  0.183168 |
| 3-4        |   667 |    255 |   412 |    0.635 |       61.769 | 0.002603 | 0.058706 | -3.11605  |  0.174822 |
| 5+         |   219 |     82 |   137 |    0.209 |       62.557 | 0.000837 | 0.019521 | -3.14956  |  0.058847 |
| 9_SENTINEL |   193 |     85 |   108 |    0.184 |       55.959 | 0.000868 | 0.015389 | -2.87577  |  0.041761 |

### `monthly_income`  —  IV = 0.0748

| bin       |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:----------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 01        |  8527 |   7753 |   774 |    8.121 |        9.077 | 0.079128 | 0.110288 | -0.332032 |  0.010346 |
| 02        |  8103 |   7314 |   789 |    7.717 |        9.737 | 0.074647 | 0.112425 | -0.409516 |  0.015471 |
| 03        |  8389 |   7666 |   723 |    7.99  |        8.618 | 0.07824  | 0.103021 | -0.275154 |  0.006819 |
| 04        |  8218 |   7532 |   686 |    7.827 |        8.348 | 0.076872 | 0.097749 | -0.240257 |  0.005016 |
| 05        |  8309 |   7727 |   582 |    7.913 |        7.004 | 0.078862 | 0.08293  | -0.05029  |  0.000205 |
| 06        |  8404 |   7844 |   560 |    8.004 |        6.663 | 0.080056 | 0.079795 |  0.003272 |  1e-06    |
| 07        |  8166 |   7678 |   488 |    7.777 |        5.976 | 0.078362 | 0.069535 |  0.119504 |  0.001055 |
| 08        |  8306 |   7891 |   415 |    7.911 |        4.996 | 0.080536 | 0.059134 |  0.308904 |  0.006611 |
| 09        |  8308 |   7923 |   385 |    7.912 |        4.634 | 0.080863 | 0.054859 |  0.387987 |  0.010089 |
| 10        |  8292 |   7905 |   387 |    7.897 |        4.667 | 0.080679 | 0.055144 |  0.380531 |  0.009717 |
| X_MISSING | 20807 |  19617 |  1190 |   19.816 |        5.719 | 0.200212 | 0.169564 |  0.166148 |  0.005092 |
| X_ZERO    |  1170 |   1131 |    39 |    1.114 |        3.333 | 0.011543 | 0.005557 |  0.731001 |  0.004376 |

### `open_credit_lines`  —  IV = 0.0676

|   bin |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|------:|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
|    01 | 15417 |  13756 |  1661 |   14.683 |       10.774 | 0.140395 | 0.236677 | -0.52224  |  0.050283 |
|    02 |  8131 |   7603 |   528 |    7.744 |        6.494 | 0.077597 | 0.075235 |  0.030907 |  7.3e-05  |
|    03 |  9042 |   8476 |   566 |    8.612 |        6.26  | 0.086507 | 0.08065  |  0.070105 |  0.000411 |
|    04 |  9490 |   8963 |   527 |    9.038 |        5.553 | 0.091477 | 0.075093 |  0.197364 |  0.003234 |
|    05 | 18041 |  17091 |   950 |   17.182 |        5.266 | 0.174432 | 0.135366 |  0.25355  |  0.009905 |
|    06 |  7976 |   7507 |   469 |    7.596 |        5.88  | 0.076617 | 0.066828 |  0.136693 |  0.001338 |
|    07 |  6734 |   6322 |   412 |    6.413 |        6.118 | 0.064523 | 0.058706 |  0.094472 |  0.00055  |
|    08 | 10744 |  10104 |   640 |   10.232 |        5.957 | 0.103122 | 0.091194 |  0.122923 |  0.001466 |
|    09 |  9754 |   9137 |   617 |    9.29  |        6.326 | 0.093253 | 0.087917 |  0.058923 |  0.000314 |
|    10 |  9670 |   9022 |   648 |    9.21  |        6.701 | 0.092079 | 0.092334 | -0.002765 |  1e-06    |

### `real_estate_loans`  —  IV = 0.0564

| bin   |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 0     | 39283 |  36010 |  3273 |   37.413 |        8.332 | 0.36752  | 0.466372 | -0.238206 |  0.023547 |
| 1     | 36645 |  34728 |  1917 |   34.9   |        5.231 | 0.354436 | 0.273155 |  0.26049  |  0.021173 |
| 2     | 22087 |  20850 |  1237 |   21.035 |        5.601 | 0.212796 | 0.176261 |  0.18837  |  0.006882 |
| 3+    |  6984 |   6393 |   591 |    6.651 |        8.462 | 0.065247 | 0.084212 | -0.255152 |  0.004839 |

### `revolving_util`  —  IV = 1.1354

| bin           |     n |   good |   bad |   % dòng |   bad rate % |    %good |     %bad |       WOE |   IV phần |
|:--------------|------:|-------:|------:|---------:|-------------:|---------:|---------:|----------:|----------:|
| 01            | 10483 |  10216 |   267 |    9.984 |        2.547 | 0.104265 | 0.038045 |  1.00817  |  0.066761 |
| 02            | 10483 |  10350 |   133 |    9.984 |        1.269 | 0.105633 | 0.018951 |  1.7181   |  0.148927 |
| 03            | 10482 |  10337 |   145 |    9.983 |        1.383 | 0.1055   | 0.020661 |  1.63046  |  0.138326 |
| 04            | 10482 |  10290 |   192 |    9.983 |        1.832 | 0.10502  | 0.027358 |  1.34514  |  0.104466 |
| 05            | 10482 |  10228 |   254 |    9.983 |        2.423 | 0.104388 | 0.036193 |  1.05926  |  0.072236 |
| 06            | 10482 |  10124 |   358 |    9.983 |        3.415 | 0.103326 | 0.051012 |  0.705836 |  0.036925 |
| 07            | 10482 |   9930 |   552 |    9.983 |        5.266 | 0.101346 | 0.078655 |  0.253472 |  0.005752 |
| 08            | 10482 |   9576 |   906 |    9.983 |        8.643 | 0.097733 | 0.129097 | -0.278319 |  0.008729 |
| 09            | 10482 |   8756 |  1726 |    9.983 |       16.466 | 0.089364 | 0.245939 | -1.01236  |  0.15851  |
| 10            | 10482 |   8011 |  2471 |    9.983 |       23.574 | 0.081761 | 0.352095 | -1.4601   |  0.394715 |
| X_IMPLAUSIBLE |   177 |    163 |    14 |    0.169 |        7.91  | 0.001664 | 0.001995 | -0.181602 |  6e-05    |

---

## Việc để lại cho bước dựng scorecard

- Kiểm hệ số hồi quy: mọi hệ số phải cùng dấu và độ lớn quanh 1. Hệ số lật dấu hoặc vọt lên
  3–4 là dấu hiệu chồng lấn giữa các biến, chỗ đầu tiên cần nhìn là cặp `debt_ratio_valid`
  và `monthly_income`.
- Chạy model có và không có bin sentinel để định lượng rủi ro leakage của `late_90`.
- Quy đổi hệ số ra thang điểm theo PDO và base-odds.

---

## Nguồn

- Agresti, A. (2013), *Categorical Data Analysis*, 3rd ed., Wiley — kiểm định hai tỉ lệ.
- Benjamini, Y. & Hochberg, Y. (1995), "Controlling the false discovery rate", *JRSS-B*, 57(1), 289–300.
- Berk, R., Brown, L., Buja, A., Zhang, K. & Zhao, L. (2013), "Valid post-selection inference",
  *Annals of Statistics*, 41(2), 802–837.
- Siddiqi, N. (2017), *Intelligent Credit Scoring*, 2nd ed., Wiley — binning và WOE trong scorecard.
- Regulation B, 12 CFR Part 1002 § 1002.9 và Appendix C — yêu cầu nêu lý do cụ thể khi từ chối tín dụng.
