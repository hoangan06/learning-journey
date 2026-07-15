# Dự báo nhu cầu bán lẻ (Rossmann) — Time-series Forecasting

## Mục tiêu
Dự báo doanh số hằng ngày của từng cửa hàng cho 6 tuần tới, đặt trong bối cảnh demand planning của một chuỗi bán lẻ FMCG (lập lịch nhân sự, tồn kho, khuyến mãi).

## Dữ liệu
Rossmann Store Sales (Kaggle): 1.115 cửa hàng, ~1 triệu dòng, doanh số theo ngày từ 01/2013 đến 07/2015. Hai bảng: `train.csv` (giao dịch) + `store.csv` (hồ sơ cửa hàng).

## Quy trình
1. **EDA chuỗi thời gian:** phân rã trend / mùa vụ / nhiễu; đo ảnh hưởng promo & ngày lễ; phát hiện lỗ dữ liệu 180 cửa hàng (07–12/2014); ACF để chọn lag.
2. **Làm sạch:** loại ngày đóng cửa & doanh thu 0; xử lý missing (phân biệt "không biết" vs "không áp dụng"); làm sạch bảng `store` trước khi gộp.
3. **Feature engineering horizon-safe:** đặc trưng lịch, promo, ngày lễ, hồ sơ cửa hàng, và **target encoding** (`store_mean`, `store_dow_mean`); huấn luyện trên `log1p(Sales)` để khớp metric RMSPE.
4. **Chia train/test THEO THỜI GIAN** (6 tuần cuối làm test, không xáo trộn); **baseline** `Store × DayOfWeek × Promo`.
5. **Mô hình:** XGBoost, dự báo trực tiếp (không đệ quy → tránh tích lũy sai số).

## Kết quả
| Mô hình | RMSPE | MAPE | MAE |
|---|---|---|---|
| Baseline (Store×DayOfWeek×Promo) | 0.147 | — | — |
| **XGBoost + target encoding** | **0.140** | **9.9%** | **~647** |

Đặc trưng lag_42 (horizon-safe) đã thử nhưng *không* cải thiện — nhất quán với kết luận EDA rằng chuỗi đi ngang (không xu hướng) nên lag ít giá trị. Feature importance xác nhận target encoding là xương sống của mô hình.

## File
- `eda_rossmann.ipynb` — EDA & làm sạch chuỗi thời gian
- `predict_rossmann.ipynb` — feature engineering, baseline, mô hình, đánh giá
- `Tong_hop_kien_thuc_2.md` — tổng hợp kiến thức time-series
- `data/` — dữ liệu

## Công nghệ
Python, pandas, numpy, scikit-learn, xgboost, statsmodels, matplotlib.