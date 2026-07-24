# Learning Journey — Chuyển ngành sang Data Science / AI

Nhật ký học tập và các dự án thực hành trên hành trình chuyển từ giảng dạy Toán sang Data Science / AI. Nền tảng Toán tốt, đang hoàn thành Thạc sĩ Khoa học dữ liệu.

## Dự án
- **Project 1 — Dự đoán giá kim cương (Diamonds):** EDA làm sạch bằng kiến thức chuyên ngành + so sánh mô hình hồi quy (Linear, log-log, Random Forest, XGBoost). Kết quả tốt nhất: Random Forest và XGBoost, R² = 0.98. [Xem thư mục »](EDA_and_basicML/project1_diamonds_dataset)

- **Project 2 — Dự báo nhu cầu bán lẻ (Rossmann, time-series):** Dự báo doanh số 6 tuần cho 1.115 cửa hàng. EDA chuỗi thời gian, feature engineering horizon-safe + target encoding, chia train/test theo thời gian. XGBoost đạt RMSPE 0.140 / MAPE 9.9%, vượt baseline. [Xem thư mục »](EDA_and_basicML/project2_rossmann_forecasting)

- **Project 3 — Phân loại ảnh CIFAR-10 bằng CNN (deep learning):** Tự thiết kế mạng CNN từ đầu bằng TensorFlow/Keras (residual blocks, BatchNorm, data augmentation, AdamW), 3.999.070 tham số (sát trần 4M), đạt 93.8% accuracy trên tập test. [Xem thư mục »](DeepLearning/project3_cifar10_cnn)

- **Project 4 — Trợ lý hỏi–đáp RAG trên văn bản pháp quy FMCG (AI/LLM):** Pipeline RAG end-to-end trên 5 văn bản luật (hải quan, an toàn thực phẩm, ghi nhãn) — chunking theo cấu trúc Điều/Khoản, embedding đa ngữ (multilingual-e5), truy hồi ngữ nghĩa, sinh câu trả lời **có trích dẫn nguồn** bằng Gemini API. Đánh giá bám nguồn: 6/6 groundedness, có phân tích lỗi trích dẫn & retrieval. [Xem thư mục »](AI_LLM/project4_rag_assistant)

## Kỹ năng thể hiện
Python, pandas, scikit-learn, xgboost, TensorFlow/Keras, EDA, phân tích chuỗi thời gian, feature engineering & target encoding, deep learning / CNN (thị giác máy tính), **RAG / LLM (embedding, vector search, prompt engineering, đánh giá faithfulness)**, chống rò rỉ dữ liệu, đánh giá mô hình, Git.

## Cấu trúc
- `EDA_and_basicML/` — các project phân tích dữ liệu & ML cơ bản
- `DeepLearning/` — các project deep learning
- `AI_LLM/` — các project AI / mô hình ngôn ngữ lớn
