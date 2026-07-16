# TỔNG HỢP KIẾN THỨC (PHẦN 3) — DEEP LEARNING / CNN

## KIẾN TRÚC MẠNG

### Residual / Skip connection
- **Cơ chế:** cộng tensor đầu vào block (qua một "đường tắt") với tensor đầu ra sau chồng conv — `Add([x, shortcut])`.
- **Vì sao cần:** chống **vanishing gradient**. Mạng càng sâu, gradient lan truyền ngược qua càng nhiều lớp càng bị nhân liên tiếp với các số nhỏ → teo dần về 0 trước khi về tới lớp đầu → lớp đầu gần như không được cập nhật. Đường tắt tạo một "đường cao tốc" cho gradient chảy thẳng về lớp trước, không bị teo.
- **Bản chất học phần dư:** thay vì bắt chồng conv học toàn bộ ánh xạ `H(x)`, mạng chỉ cần học phần dư `F(x) = H(x) − x`. Nếu ánh xạ tối ưu gần với "giữ nguyên đầu vào" thì chỉ cần đẩy `F(x) → 0` → thêm lớp không làm mạng tệ đi (giải quyết *degradation problem*).
- **Identity vs projection shortcut:**
  - Số kênh **không đổi** → shortcut là **identity** (cộng thẳng đầu vào, đường cao tốc gradient sạch nhất).
  - Số kênh **đổi** (vd 64→128) → *không cộng identity được*, phải **chiếu (projection)** để khớp shape. ResNet dùng **conv 1×1** (rẻ); dùng conv 3×3 thì học được nhiều hơn nhưng tốn tham số hơn và bớt "thuần" về gradient.

### Đầu ra: Flatten vs Global Average Pooling (GAP)
- Phần lớn tham số của một CNN thường dồn vào lớp **Dense ngay sau `Flatten`** (Flatten `h×w×c` thành vector dài → Dense tốn `h×w×c × units` tham số).
- **GAP** bình quân toàn bộ mỗi feature map (`h×w`) thành 1 số → vector độ dài đúng bằng số kênh → **ít tham số hơn hẳn**; đồng thời là một *regularizer cấu trúc* (không có tham số để overfit). Các mạng hiện đại (ResNet, Inception) dùng GAP thay FC khổng lồ.
- **Đánh đổi:** `Flatten` **giữ** thông tin vị trí không gian; GAP **xóa** vị trí (bình quân hết). Trên feature map nhỏ, giữ vị trí đôi khi có ích → Flatten có thể tốt hơn nhưng tốn tham số.

## CHUẨN HÓA & KÍCH HOẠT

### Chuẩn hóa đầu vào (z-score)
- `(x − mean) / (std + eps)`; tính `mean`, `std` **trên tập train**, áp cho cả val/test. Giúp huấn luyện ổn định, hội tụ nhanh.

### Batch Normalization (BN)
- Công thức: `y = γ·((x − μ_batch)/σ_batch) + β` — chuẩn hóa activation mỗi batch về mean 0/std 1, rồi scale/shift bằng `γ, β` **học được** (để mạng khôi phục phân phối nếu cần).
- **Lợi:** hội tụ nhanh hơn, cho phép learning rate cao hơn, giảm phụ thuộc vào cách khởi tạo, regularize nhẹ.
- **Quan trọng — lúc inference:** BN **không** dùng thống kê batch, mà dùng **mean/var trung bình trượt** tích lũy trong lúc train.

### Activation & khởi tạo
- **ReLU** `max(0,x)`: chặt số âm về 0 (tạo sparsity nhưng neuron có thể "chết").
- **SiLU** `x·sigmoid(x)`: trơn, cho số âm nhỏ đi qua (không chặt cứng) → thường tốt hơn ở mạng sâu.
- **He initialization:** khởi tạo trọng số phù hợp cho họ ReLU, giữ phương sai activation ổn định qua các lớp.

## CHỐNG OVERFITTING (REGULARIZATION)
> Mạng dung lượng lớn + dữ liệu nhỏ → dễ **học vẹt** (train acc cao, test acc thấp). Cần chồng nhiều lớp phòng thủ.

- **Dropout / SpatialDropout2D:** tắt ngẫu nhiên neuron / feature-map khi train → mạng không phụ thuộc vào một đường duy nhất.
- **Weight decay (L2):** phạt trọng số lớn → mô hình đơn giản hơn.
- **Data augmentation** = regularization *trong không gian dữ liệu*: mở rộng dataset hiệu dụng, ép mạng học **bất biến (invariance)** thay vì thuộc lòng. Gồm: random crop, flip, contrast, **cutout**.
- **Cutout:** che một ô vuông ngẫu nhiên thành 0 → ép mạng **không dựa vào một vùng đặc trưng duy nhất**, phải dùng nhiều bộ phận của vật thể → khỏe hơn với che khuất (tương tự dropout nhưng ở không gian ảnh đầu vào).

## TỐI ƯU HÓA

### Adam vs AdamW
- **Adam + L2 (phạt trong loss):** vì Adam chia gradient theo trung bình bình phương gradient của từng tham số (thích nghi), lượng weight decay bị **cuốn theo** phép chia đó → phạt không đều, bị méo.
- **AdamW:** **tách (decouple)** weight decay khỏi bước gradient, co trọng số về 0 một cách trực tiếp và đều → weight decay đúng nghĩa, thường **generalize tốt hơn**.

### Lịch learning rate & clipping
- **LR cao lúc đầu:** học nhanh, tiến nhanh vào vùng tối ưu, nhảy qua được các cực tiểu cục bộ xấu.
- **Giảm LR dần về cuối:** bước nhỏ, *lắng* vào đáy thay vì dao động/nhảy vọt quanh nó.
- **Gradient clipping (`clipnorm`):** chặn gradient quá lớn → ổn định huấn luyện.

## LOSS & PHƯƠNG PHÁP LUẬN
- **`sparse_categorical_crossentropy`:** dùng khi nhãn là **số nguyên** (0..9); `categorical_crossentropy` mới dùng cho nhãn one-hot.
- Chia **train / validation / test:** validation để *chọn model* và theo dõi overfitting mà không đụng vào test.
- **`ModelCheckpoint(save_best_only=True)` chỉ lưu file model tốt nhất RA ĐĨA — KHÔNG tự nạp lại vào biến `model`.** Muốn chấm model tốt nhất phải **load lại tường minh**, hoặc dùng **`EarlyStopping(restore_best_weights=True)`** để tự khôi phục trọng số tốt nhất vào `model`.
- Luôn **đánh giá model TỐT NHẤT** (theo val_loss), không phải model ở epoch cuối.
- Con số `model.evaluate` **trả về** (`result`) mới là metric chuẩn tính trên toàn tập; số hiển thị trên thanh tiến trình chỉ là ước lượng chạy theo từng batch → có thể lệch nhẹ.
