# TỔNG HỢP KIẾN THỨC

## NGÀY 1

### Quy trình EDA cho dữ liệu bảng

#### **Thang EDA 7 tầng**

1. **Tầng 0: Đóng khung** (trước khi nhìn vào số liệu)
    - Câu hỏi: "Mục tiêu phân tích là gì? Cột nào là target? Mỗi dòng đại diện cho cái gì? Từng cột nghĩa là gì?"
    - Trả lời: Bối cảnh và bàng mô tả cột.

2. **Tầng 1: Tổng quan**
    - Câu hỏi: "Dữ liệu to cỡ nào? Kiểu gì? Trông ra sao?"
    - Trả lời: info, shape, dtype, head, describe.

3. **Tầng 2: Chất lượng dữ liệu**
    - Câu hỏi: "Điều gì khiến một dòng hoặc một giá trị trở nên vô nghĩa?"
    - Trả lời: thiếu, trùng, giá trị bất khả thi, vi phạm nguyên tắc ngành, mâu thuẫn giữa các cột.
    - ***Chú ý:*** nên tìm hiểu thêm các kiến thức chuyên ngành liên quan đến bộ dữ liệu mà ta đang phân tích.

4. **Tầng 3: Phân tích riêng từng biến** (univariate)
    - Câu hỏi: "Mỗi biến phân bố như thế nào?"
    - Trả lời: Tâm, độ phân tán, hình dạng (histogram), có bị lệch không.
    - ***Đặc biệt:*** phân phối chính của biến target, nếu bị lệch thì cân nhắc log.
    - *Kiến thức bên lề:* Lệch phải là triệu chứng, quan hệ nhân/lũy thừa là nguyên nhân, log là thuốc. Log biến phép nhân thành phép cộng, biến tỉ lệ thành khoảng cách đều.

5. **Tầng 4: Phân tích theo cặp** (bivariate)
    - Câu hỏi: "Mỗi biến có liên hệ với target không? Và liên hệ kiểu gì? (tuyến tính? đơn điệu? không có?)"
    - Trả lời: Scatter/boxplot theo target, ma trận tương quan.

6. **Tầng 5: Phân tích đa biến, kiểm soát nhiễu**
    - Câu hỏi: "Quan hệ vừa thấy là thật hay do một biến thứ ba giật dây?"
    - Trả lời: kiểm soát confounder, nghịch lý Simpson, tương tác.

7. **Tầng 6: Tổng hợp & kết luận**
    - Câu hỏi: "Câu trả lời cho câu hỏi ban đầu là gì? Điểm hạn chế? Bước tiếp theo?"

#### **Những phân tích quan trọng hay bị xót:**
- Phân phối chính của biến target.
- Kiểu của giá trị thiếu - thiếu ngẫu nhiên hay thiếu có hệ thống, thiếu có liên quan đến target không.
- Trùng dòng là lỗi hay thật.
- Outlier là lỗi hãy cự trị thật.
- Đa cộng tuyến (multicollinearity).
- Rò rỉ dữ liệu (data leakage): loại các biến "ăn gian" có chứa thông tin của target ra khỏi mô hình.

#### **Câu hỏi vạn năng:**
- "Cái gì khiến dòng này vô lý?"
- "Quan hệ này có thể do biến thứ 3 gây ra không?"

### Cách đánh giá một mô hình ML cơ bản

Hai bài toán cơ bản trong ML là hồi quy (regression) và phân loại (classification).

Để đánh giá một mô hình hồi quy, metric cơ bản nhất là:
- MSE = trung bình của (giá thật - giá dự đoán)^2.
- RMSE = \sqrt{MSE} (có cùng đơn vị với target).

**Chú ý:** Khi đánh giá một mô hình là "tốt" hay "tệ" thì cần so sánh vơi một mốc tham chiếu (baseline). ***Một chỉ số không có mốc so sánh thì không có ý nghĩa***.

Trong bài toán hồi quy, baseline cơ bản nhất là **"luôn đoán bằng trung bình"**. Trong trường hợp này, RMSE của baseline chính là **độ lệch chuẩn**.

Từ đó ta có chỉ số R² = 1 − (MSE của mô hình) / (MSE của baseline đoán-trung-bình). Chỉ số này dùng để đo mô hình của bạn giỏi hơn "baseline luôn đoán bằng trung bình" bao nhiêu. R² = 0 nghĩa là ngang baseline; R² < 0 nghĩa là mô hình còn tệ hơn cả đoán bừa bằng trung bình. 
- Ví dụ: trong một bài toán dự đoán giá cả, mô hình cho R² = 0.91 thì ta nói "mô hình giải thích được ~91% biến động giá".

### Bài học kinh nghiệm: 
- Luôn đọc kỹ thông tin, nhìn vào số liệu để trả lời. Không được đọc lướt qua và áng chừng để đưa ra kết luận.