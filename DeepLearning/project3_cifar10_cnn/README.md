# Phân loại ảnh CIFAR-10 bằng CNN tự thiết kế (from scratch)

## Mục tiêu
Tự thiết kế và huấn luyện một mạng CNN **từ đầu** (không dùng mạng pretrained) để phân loại ảnh CIFAR-10, dưới ràng buộc số tham số **≤ 4 triệu** — buộc phải tối ưu kiến trúc thay vì "phình" mạng.

## Dữ liệu
CIFAR-10: 60.000 ảnh màu 32×32, 10 lớp (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), mỗi lớp 6.000 ảnh. Chia sẵn 50.000 ảnh train / 10.000 ảnh test.

## Kiến trúc & kỹ thuật
- **Kiến trúc:** 4 khối tích chập (64 → 128 → 256 → 512 kênh), ba khối đầu có **residual/skip connection** (chống vanishing gradient); mỗi khối kết bằng MaxPooling + SpatialDropout; đầu ra Flatten → Dense → softmax.
- **Chuẩn hóa:** z-score đầu vào; **BatchNormalization** giữa các lớp; **He initialization**; activation **SiLU**.
- **Chống overfitting:** Dropout, SpatialDropout, weight decay (L2), và **data augmentation** mạnh (random crop, flip, contrast, **cutout**).
- **Tối ưu:** **AdamW** (decoupled weight decay) + lịch learning rate giảm bậc thang (PiecewiseConstantDecay) + gradient clipping.
- **Huấn luyện:** loss `sparse_categorical_crossentropy`; chia train/validation/test; `ModelCheckpoint` lưu model tốt nhất theo `val_loss`, nạp lại model tốt nhất trước khi đánh giá.

## Kết quả
Đạt **93.8% accuracy** trên tập test (giá trị `model.evaluate` trả về) với **3.999.070 tham số** — sát trần ràng buộc 4M. Mạng tự thiết kế hoàn toàn từ đầu.

## File
- `dl_cifar10_cnn.ipynb` — toàn bộ quy trình: xử lý dữ liệu, kiến trúc, huấn luyện, đánh giá
- `Tong_hop_kien_thuc_3.md` — tổng hợp kiến thức deep learning / CNN

## Công nghệ
Python, TensorFlow/Keras, NumPy, matplotlib.
