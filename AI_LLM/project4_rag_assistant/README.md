# Trợ lý hỏi–đáp RAG trên văn bản pháp quy chuỗi cung ứng FMCG

> **Trạng thái:** hoàn thành — pipeline RAG end-to-end chạy được, trả lời kèm trích dẫn nguồn, có đánh giá bám nguồn (faithfulness).

## Mục tiêu
Xây dựng trợ lý hỏi–đáp (RAG) trả lời câu hỏi về **quy định pháp luật áp dụng cho vận hành chuỗi cung ứng ngành hàng tiêu dùng nhanh (FMCG)**, câu trả lời **bắt buộc kèm trích dẫn nguồn** đến từng Điều/Khoản của văn bản gốc.

Bối cảnh nghiệp vụ bao trọn vòng đời sản phẩm:

```
Nhập nguyên liệu (hải quan) → Sản xuất (an toàn thực phẩm) → Ra thị trường (ghi nhãn)
```

Bài toán thực tế: nhân viên vận hành / QA / xuất nhập khẩu cần tra cứu nhanh một quy định cụ thể trong hàng trăm điều luật rải rác nhiều văn bản, và cần biết **căn cứ ở đâu** chứ không chỉ một câu trả lời trôi chảy.

## Dữ liệu
5 văn bản quy phạm pháp luật công khai, tải từ nguồn chính thức (Công báo Chính phủ, Cơ sở dữ liệu quốc gia về pháp luật, Thư viện pháp luật):

| Văn bản | Vai trò trong chuỗi | Định dạng |
|---|---|---|
| Luật An toàn thực phẩm — VBHN 61/VBHN-VPQH (2025) | Sản xuất, kinh doanh thực phẩm | `.docx` |
| Luật Hải quan — VBHN 54/VBHN-VPQH (2026) | Nhập nguyên liệu, thông quan | `.html` |
| Nghị định 15/2018/NĐ-CP | Hướng dẫn Luật ATTP (tự công bố, GMP, nhập khẩu) | `.html` |
| Nghị định 43/2017/NĐ-CP | Nhãn hàng hóa | `.html` |
| Nghị định 111/2021/NĐ-CP | Sửa đổi NĐ 43 về nhãn hàng hóa | `.html` |

Ưu tiên **văn bản hợp nhất** để tránh trích dẫn điều khoản đã hết hiệu lực.

## Quy trình

### 1. Thu thập & khảo sát dữ liệu
- Kiểm tra định dạng PDF bằng phép đo (`len(page.get_text())`) thay vì phán đoán bằng mắt → phát hiện 3/4 PDF là **bản scan** (không có lớp text) ⇒ chuyển sang nguồn HTML toàn văn thay vì OCR.
- Convert `.doc` → `.docx` (LibreOffice headless) để `python-docx` đọc được.
- Kiểm tra tính đầy đủ: đối chiếu số Điều thu được với số Điều thực của từng văn bản.

### 2. Trích xuất văn bản (tách I/O khỏi logic)
- `read_docx_paragraphs()` — đọc `.docx`; kiểm tra riêng `doc.tables` (khối chữ ký chứa số hiệu & ngày ban hành) và `word/footnotes.xml` (15 footnote = bản đồ sửa đổi).
- `read_html_paragraphs()` — BeautifulSoup, khoanh vùng đúng khối toàn văn (`div.cldivContentDocVn`) trước khi lấy thẻ `<p>`, tránh kéo theo menu và văn bản liên quan.
- Cả hai cùng trả về một dạng `list[str]` → dùng chung một hàm chunking.

### 3. Chunking theo cấu trúc văn bản pháp luật
- `chunk_paragraphs()` — duyệt một lượt với biến trạng thái, nhận diện ranh giới **Chương** (số La Mã) và **Điều** bằng regex neo đầu dòng; tự động dừng tại khối chữ ký ("Nơi nhận:").
- `split_amendment_chunk()` — tách Điều 1 của NĐ 111 (nghị định sửa đổi) thành từng **Khoản**, mỗi khoản là một sửa đổi, kèm metadata `sua_doi_cho` trỏ tới điều khoản bị sửa.
- `split_long_chunk()` — cắt **đệ quy theo tầng** (Điều → Khoản → Điểm) cho các chunk vượt giới hạn token, kèm **gom nhóm tham lam** để mảnh vụn không quá nhỏ.
- **Contextual chunking:** mọi chunk mang tiền tố tiêu đề Điều cha trong `text_for_embedding`, giữ `noi_dung` nguyên bản để hiển thị trích dẫn.

### 4. Kiểm soát giới hạn token
Đo bằng tokenizer thật của mô hình embedding (`intfloat/multilingual-e5-base`, `max_seq_length = 512`):

| | Trước khi cắt | Sau khi cắt |
|---|---|---|
| Số chunk | 257 | **291** |
| Token trung vị | — | 230 |
| Token tối đa | 1342 | **505** |
| Chunk vượt 512 token | 26 | **0** |

Với mỗi chunk vượt ngưỡng, phần bị cắt được **giải mã ra để đọc và đánh giá mức thiệt hại** trước khi quyết định cắt tiếp hay chấp nhận — vì nội dung cuối điều khoản thường là điều kiện phủ định phần đầu.

### 5. Vector hóa & truy hồi ngữ nghĩa
- Mã hóa 291 chunk bằng `intfloat/multilingual-e5-base` (chọn theo benchmark VN-MTEB; đa ngữ, context 512 token, chạy local trên GPU RTX 4050), chuẩn hóa L2 để cosine similarity = tích vô hướng.
- Truy hồi brute-force (`emb @ q`) — chính xác tuyệt đối, đủ nhanh ở quy mô vài trăm vector; ghi rõ khi nào cần chuyển sang FAISS/ANN.
- **Đánh giá định tính** trên bộ 5 câu hỏi (4 trong phạm vi + 1 ngoài phạm vi): 3/4 câu trong phạm vi có chunk đúng ở top-1, câu còn lại ở top-1 & top-3.

**Phát hiện quan trọng:** câu hỏi ngoài phạm vi vẫn đạt cosine 0.80–0.81 (chỉ thấp hơn chút so với 0.84–0.90 của câu trong phạm vi) ⇒ **không thể dùng ngưỡng điểm cố định** để phát hiện "không có trong tài liệu"; việc này phải giao cho LLM ở khâu sinh câu trả lời.

### 6. Sinh câu trả lời có trích dẫn & đánh giá bám nguồn
- Sinh câu trả lời bằng **Gemini API** (gọi kèm retry + exponential backoff cho lỗi 503). Prompt ràng buộc bốn chân: chỉ dùng ngữ cảnh, trả lời từng phần, ưu tiên bản sửa đổi, fallback khi không có căn cứ. Khóa API đọc từ `.env` (đã loại trừ trong `.gitignore`).
- Chạy bộ **6 câu hỏi** (5 trong phạm vi + 1 ngoài phạm vi), lưu artifact `eval_runs.json` (ghim input–output cùng lần chạy vì LLM sinh không tất định).
- **Đánh giá faithfulness** trên hai trục: *groundedness* (nội dung có trong ngữ cảnh?) và *độ chính xác trích dẫn* (đường dẫn Điều/Khoản đúng?).

**Kết quả đánh giá:**

| Chỉ số | Kết quả |
|---|---|
| Groundedness | **6/6** — không câu nào bịa nội dung; fallback đúng cho câu ngoài phạm vi |
| Độ chính xác trích dẫn | **4/6** đúng hoàn toàn (các câu không dính sửa đổi); 2/6 sai đường dẫn ở chunk sửa đổi |

Hai phát hiện: (1) **lỗi trích dẫn ở chunk sửa đổi** do cấu trúc lồng hai tầng (NĐ 111 Điều 1 Khoản 5 → Điều 10 mới Khoản 1) — LLM bỏ mất tầng giữa; (2) **retrieval trượt** ở câu về nhãn lưu thông nội địa — chunk đúng nhất không lọt top-k do một Điều bị cắt thành nhiều part. Cả hai đều có hướng khắc phục ghi trong phần Hạn chế.

## Kết quả
- `data/chunks.json` — 291 chunk + metadata phục vụ trích dẫn.
- `data/embeddings.npy` — ma trận embedding (291 × 768), đã chuẩn hóa L2.
- `data/eval_runs.json` — artifact 6 lần chạy (câu hỏi, chunk truy hồi, ngữ cảnh, câu trả lời).
- Pipeline RAG end-to-end: câu hỏi → truy hồi → sinh câu trả lời **có trích dẫn**, tự từ chối khi ngoài phạm vi.

## Hướng cải tiến (đã xác định từ đánh giá)
- **Parent-child retrieval:** khi một part trúng, nở về Điều cha (`parent_id` đã lưu sẵn) — khắc phục retrieval trượt do cắt part.
- **Metadata trích dẫn chuẩn** cho chunk sửa đổi (`trich_dan_chuan`) — khắc phục lỗi đường dẫn lồng hai tầng.
- **So sánh 2–3 mô hình embedding** (gte-multilingual-base, halong_embedding) trên bộ câu hỏi thật, đo Recall@k.
- **Hybrid search** (BM25 + vector) cho mã văn bản/số hiệu.

## Hạn chế đã biết
- **Phụ lục dạng bảng chưa được xử lý** — trong đó có Phụ lục I của NĐ 43 (danh mục nội dung bắt buộc ghi nhãn theo nhóm hàng hóa). Hướng xử lý: chuyển mỗi dòng bảng thành câu văn xuôi trước khi chunk.
- **Footnote chưa đưa vào chunk.** Với văn bản hợp nhất, nội dung sửa đổi đã nằm trong phần thân; footnote chỉ ghi xuất xứ sửa đổi. Hướng cải tiến: trích footnote làm giàu metadata lịch sử hiệu lực.
- **NĐ 43 + NĐ 111 chưa được hợp nhất** (không có bản hợp nhất chính thức công khai). Hiện gắn cờ metadata `sua_doi`; độ chính xác phụ thuộc việc LLM có sử dụng ghi chú đó hay không — sẽ kiểm ở khâu đánh giá.
- **Điều 101 Luật Hải quan** sửa Luật Quản lý thuế, nằm ngoài phạm vi FMCG.
- Hệ thống trả lời theo **đúng phiên bản văn bản đã nạp**, không tự cập nhật khi pháp luật thay đổi.

## File
- `rag_fmcg_chunking.ipynb` — thu thập, khảo sát, trích xuất và chunking dữ liệu
- `rag_fmcg_embedding.ipynb` — vector hóa & truy hồi ngữ nghĩa
- `rag_fmcg_qa_eval.ipynb` — sinh câu trả lời có trích dẫn & đánh giá bám nguồn
- `Tong_hop_kien_thuc_4.md` — tổng hợp kiến thức RAG & xử lý dữ liệu
- `data/` — văn bản gốc (`.docx`, `raw_html/`), `chunks.json`, `embeddings.npy`, `eval_runs.json`

## Công nghệ
Python, BeautifulSoup, python-docx, PyMuPDF, regex, pandas, sentence-transformers (multilingual-e5), HuggingFace tokenizers, FAISS (khái niệm), Gemini API. Kỹ thuật RAG: chunking theo cấu trúc, contextual chunking, kiểm soát token, truy hồi ngữ nghĩa, prompt engineering, đánh giá faithfulness.
