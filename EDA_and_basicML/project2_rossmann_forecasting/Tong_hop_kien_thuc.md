# TỔNG HỢP KIẾN THỨC (PHẦN 2)

## QUY TRÌNH EDA CHO DỮ LIỆU TIME-SERIES 

### **Thang EDA 8 tầng**

1. **Tầng 0 - Hiểu dữ liệu & bài toán:** Mỗi dòng là gì? Target là gì? Tần suất? Phạm vi thời gian? Dự đoán bao xa?
2. **Tầng 1 - Trục thời gian:** Parse `Date` đúng kiểu, kiểm chuỗi dữ liệu liên tục hay đứt quãng, timestamp có trùng không, sắp xếp theo thời gian.
3. **Tầng 2 - Data quality audit:** Kiểm tra dữ liệu thiếu (phân biệt "không biết" và "không áp dụng"), giá trị bất thường, các dòng không phản ánh nhu cầu thật.
4. **Tầng 3 - Phân rã 3 thành phần:** Luôn nhìn chuỗi qua lăng kính **"Trend" + "Seasonally" + "Noise"**. Dùng `resample` để làm mượt theo từng tầng.
5. **Tầng 4 - Mùa vụ đa chu kỳ:** Khảo sát theo thứ trong tuần, theo quý, theo tháng, theo năm. Luôn cảnh giác bẫy compositioin khi tổng hợp trung bình trên một tập đối tượng thay đổi.
6. **Tầng 5 - Biến ngoại sinh (Exogenous):** ví dụ như khuyến mãi, ngày lễ, sự kiện. So sánh có/không, đóng đinh bằng số (chênh lệch, tương quan). Mỗi biến đều phải qua ***thước đó rò rỉ*** (lúc dự báo có biết trước không?).
7. **Tầng 6 - Tự tương quan (ACF/PACF):** Doanh thu phụ thuộc mức nào vào chính nó ở các độ trễ khác nhau -> trực tiếp chỉ ra nên tạo lag nào.
8. **Tầng 7 - Kết luận:** Nguyên tắc chốt: mỗi phát hiện EDA phải sinh ra một feature ứng viên.

### **Bài học kinh nghiệm**
- Thấy gì bằng mắt phải xác định lại bằng số.
- Cảnh giác composition bias.
- Khi có nhiều đối tượng (cửa hàng), kiểm ở đúng grain từng đối tượng nhưng nhìn phân phối/trung bình của chỉ số để chẩn đoán (mean ACF, phân phối max_gap), chứ không vẽ hàng loạt biểu đồ.
- Làm sạch bảng tra cứu (dimension) trước khi join vào bảng sự kiện (fact).
- `Restart` & `Run all` trước khi tin.

## "CÂY THƯỚC" CHỐNG RÒ RỈ DỮ LIỆU
> Lúc dự đoán một mẫu mới ta có thực sự sở hữu giá trị này không?

## NGUYÊN TẮC CHIA TRAIN/TEST CHO DỮ LIỆU TIME-SERIES
> Chia theo trục thời gian, cắt phần đầu để train, phần cuối để test. Tuyệt đối không để rò rỉ dữ liệu tương lai.