# Dự đoán giá kim cương (Diamonds) — EDA + Regression

## Mục tiêu
Phân tích yếu tố quyết định giá kim cương và xây mô hình dự đoán giá, đặt trong bối cảnh một nhà bán lẻ muốn định giá dựa trên dữ liệu.

## Dữ liệu
Bộ `diamonds` (~54.000 viên, từ thư viện seaborn), gồm 4C (carat, cut, color, clarity), kích thước (x, y, z), depth, table, price.

## Quy trình
1. EDA & làm sạch: dùng kiến thức chuyên ngành (công thức hình học của kim cương) để bắt dữ liệu lỗi; kiểm soát biến gây nhiễu (carat) khi so sánh 4C.
2. Tiền xử lý: mã hóa thứ tự cho cut/color/clarity; loại biến rò rỉ (price_per_carat).
3. Mô hình & so sánh: Linear, Linear log-log, Random Forest.

## Kết quả
| Mô hình | R² (test) |
|---|---|
| Linear (price) | 0.905 |
| Linear (log-log) | 0.943 |
| **Random Forest** | **0.981** |
| **XGBoost** | **0.981** |

Feature importance của mô hình xác nhận kết luận EDA: kích thước áp đảo, trong nhóm 4C thì clarity ảnh hưởng mạnh nhất.

## File
- `EDA_diamonds.ipynb` — phân tích & làm sạch
- `diamonds_predict.ipynb` — mô hình dự đoán
- `data/` — dữ liệu đã làm sạch

## Công nghệ
Python, pandas, scikit-learn, seaborn, matplotlib, xgboost.