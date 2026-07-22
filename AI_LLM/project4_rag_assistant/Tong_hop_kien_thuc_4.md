# Tổng hợp kiến thức — Project 4: Trợ lý hỏi–đáp RAG trên văn bản pháp quy FMCG

> Phần I–IV là kiến thức nền RAG; phần V–VI là kỹ thuật xử lý dữ liệu thực tế đã gặp trong project.

---

## I. Vì sao cần RAG

### 1. Giới hạn của LLM thuần
- Kiến thức của LLM nằm trong **trọng số (weights)**, được "đóng băng" tại thời điểm kết thúc huấn luyện. Tài liệu nội bộ / văn bản mới không có trong dữ liệu huấn luyện → mô hình không biết.
- Nguy hiểm hơn "không biết": mô hình **không nói là không biết** mà bịa ra câu trả lời trôi chảy (**hallucination — ảo giác**). Với tài liệu pháp lý, trả lời sai mà nghe thuyết phục còn tệ hơn không trả lời.

### 2. Ba hướng đưa kiến thức riêng vào LLM

| Hướng | Cách làm | Nhược điểm |
|---|---|---|
| **Fine-tune** | Huấn luyện tiếp trên tài liệu | Tốn tài nguyên; rủi ro *catastrophic forgetting*; tài liệu đổi là phải train lại; **không trích dẫn được nguồn** |
| **Nhồi hết vào prompt** | Dán toàn bộ tài liệu mỗi lần hỏi | Vượt **context window**; hiện tượng **lost in the middle**; đốt token/chi phí |
| **RAG** | Truy hồi vài đoạn liên quan → đưa cho LLM | Phụ thuộc chất lượng retrieval; vẫn gửi dữ liệu lên API nếu dùng LLM đám mây |

- **Context window:** giới hạn cứng số token đầu vào của LLM.
- **Lost in the middle:** kể cả khi nội dung lọt trong context window, LLM chú ý tốt phần đầu và cuối prompt, kém phần giữa → *nhồi được ≠ dùng tốt*.
- **Bảo mật:** RAG dùng API **không** giải quyết triệt để — mỗi lần trả lời vẫn gửi các đoạn liên quan lên server bên thứ ba. Muốn triệt để phải self-host LLM.

### 3. Pipeline RAG
```
Nạp tài liệu → Chunking → Embedding → Vector store
   → (câu hỏi) → Retrieval top-k → Generation có trích dẫn nguồn
```

---

## II. Tìm kiếm theo nghĩa

### 1. Vì sao tìm từ khóa không đủ
Ctrl+F / khớp chữ **không đáng tin cậy**: cùng một ý có nhiều cách diễn đạt ("nghỉ mấy ngày" ↔ "5 ngày làm việc"). Lúc trúng lúc trượt tùy người hỏi có tình cờ dùng đúng chữ trong tài liệu hay không.

> Lưu ý cân bằng: tìm từ khóa **không vô dụng**. Hệ RAG sản xuất thường dùng **hybrid search** (BM25 + vector), vì từ khóa rất mạnh với mã sản phẩm, tên riêng, số hiệu văn bản (vd "43/2017/NĐ-CP").

### 2. Cosine similarity
$$\cos(u,v) = \frac{u \cdot v}{\lVert u\rVert \lVert v\rVert}$$
- Đo **mức độ cùng hướng** của hai vector, giá trị trong [-1, 1].
- Chia cho tích độ dài = tách riêng phần *hướng*, loại bỏ ảnh hưởng của *độ lớn*. Nếu chỉ dùng tích vô hướng thô, hai vector dài nhưng lệch hướng vẫn có tích lớn → sai.
- **Hệ quả thực dụng:** nếu vector đã được chuẩn hóa về độ dài 1 (`normalize_embeddings=True`), thì cosine similarity = **tích vô hướng** → tính rất rẻ.

### 3. Vì sao bag-of-words không đủ
Vector đếm từ chỉ thấy phần **trùng chữ**; phần đồng nghĩa khác chữ đóng góp đúng bằng 0. Muốn "hiểu nghĩa" thì phép biến văn bản → vector phải có tính chất *gần nghĩa ⇒ gần vector*.

### 4. Tính chất đó từ đâu ra
Mô hình embedding là mạng neural được **huấn luyện có chủ đích** cho mục tiêu này, thường bằng **contrastive learning**: cho xem hàng trăm triệu cặp (câu hỏi–đoạn trả lời, cặp diễn giải…), kéo cặp dương lại gần, đẩy cặp âm ra xa. "Nghĩa" được nén từ dữ liệu huấn luyện vào trọng số.

### 5. Language model ≠ Embedding model (điểm hay bị nhầm)
- **PhoBERT** là *pretrained language model* (masked LM), dùng làm nền để fine-tune cho phân loại, NER, POS… Lấy `[CLS]`/mean-pooling của nó rồi tính cosine cho kết quả **kém** — vì MLM không hề dạy "hai câu cùng nghĩa thì vector gần nhau" (kết quả kinh điển từ bài *Sentence-BERT*, 2019: BERT thuần mean-pooling còn thua trung bình GloVe trên STS).
- **Embedding model = language model + giai đoạn huấn luyện tương phản.**
- `bkai-foundation-models/vietnamese-bi-encoder` chính là PhoBERT đã fine-tune thành bi-encoder.

---

## III. Chunking — cắt tài liệu

### 1. Vì sao phải cắt (hai bờ vực)
| Chunk quá to | Chunk quá nhỏ |
|---|---|
| **Pha loãng ngữ nghĩa**: 1 vector chỉ chỉ một hướng, embed 20 chủ đề → xấp xỉ trung bình 20 hướng, gần đều mọi thứ, không gần hẳn cái gì | **Mất ngữ cảnh**: câu điều kiện bị tách khỏi câu quyền lợi |
| Nhồi cả rác vào prompt | Retrieval trúng mảnh vụn không đủ trả lời |
| Nếu để cả tài liệu = 1 vector → luôn là top-1 → quay về "dán hết vào prompt" | |

**Ví dụ mất ngữ cảnh:** "Công ty hoàn trả 100% chi phí…" / "Tuy nhiên, chỉ áp dụng với chuyến công tác nước ngoài đã được phê duyệt." Tách hai câu là sinh ra câu trả lời sai nguy hiểm.

### 2. Các chiến lược cắt (thô → khôn)
1. Cắt cứng theo độ dài.
2. Cắt theo ranh giới tự nhiên (*recursive splitting*: ưu tiên `\n\n` → `\n` → giữa câu).
3. **Cắt theo cấu trúc tài liệu** — tốt nhất với văn bản pháp luật: mỗi **Điều** là một chunk, Điều quá dài thì xuống **Khoản**, rồi **Điểm**.

### 3. Overlap (chồng lấn)
Chunk sau lặp ~10–20% cuối chunk trước.
- **Cứu được:** vết cắt rơi đúng giữa hai đoạn bổ nghĩa cho nhau.
- **Không cứu được:** phụ thuộc xa (điều kiện nằm cách 2 trang).
- **Giá phải trả:** lưu trùng lặp, tốn thêm token embedding.

### 4. Metadata cho mỗi chunk (văn bản pháp luật)
`chunk_id`, `noi_dung` (hiển thị), `text_for_embedding` (để mã hóa), `so_dieu`, `ten_dieu`, `so_chuong`, `ten_chuong`, `bo_luat`, `so_hieu`, `ngay_ky`, `nguon`, `sua_doi`, `sua_doi_cho`, `parent_id`, `part`.

**Hai nguyên tắc rút ra:**
- **Tách "văn bản để embed" khỏi "văn bản để hiển thị"** (`text_for_embedding` vs `noi_dung`) — chuẩn mực trong RAG.
- **Contextual chunking:** chunk con tự mang theo tiêu đề cha (`"Điều 2. Giải thích từ ngữ: …"`) → giữ ngữ cảnh mà không cần cơ chế tra ngược lúc retrieval. Rẻ hơn parent-child thật.

### 5. Parent-child retrieval (small-to-big)
Tách rời **đơn vị truy hồi** khỏi **đơn vị đưa cho LLM**: embed chunk nhỏ (chính xác, khớp giới hạn token), nhưng khi trúng thì đưa cho LLM **cả Điều cha**. Giải quyết mâu thuẫn "cắt nhỏ thì mất ngữ cảnh". Trường `parent_id` được lưu sẵn để bật tính năng này khi cần.

### 6. Bẫy văn bản pháp luật
- **Hiệu lực & sửa đổi:** nạp bản gốc mà bỏ bản sửa đổi → chatbot trích dẫn điều khoản đã hết hiệu lực. Ưu tiên **văn bản hợp nhất**; nếu không có, gắn cờ metadata `sua_doi` + ghi chú trong chunk.
- **Điều "được bãi bỏ":** text rỗng → embed ra vector vô nghĩa (nhiễu) hoặc không embed được. Xử lý: đưa chính ghi chú bãi bỏ vào nội dung chunk → hệ thống trả lời đúng sự thật kèm trích dẫn.
- **Trích dẫn lồng nhau:** trong nghị định sửa đổi, các Điều được trích lại nằm trong ngoặc kép `"Điều 10. …"`. Dùng `re.match` (neo đầu chuỗi) thì các dòng này **không** bị nhận nhầm là ranh giới Điều mới — đúng như mong muốn, không cần thêm code.

---

## IV. Embedding & giới hạn token

### 1. Truncation âm thầm — lỗi nguy hiểm nhất
Mỗi mô hình embedding có `max_seq_length`. Text vượt ngưỡng bị **cắt cụt không báo lỗi**: phần đuôi vẫn nằm trong dữ liệu nhưng **vô hình với retrieval**.
- Đo bằng **tokenizer của chính mô hình đó**, không ước lượng theo ký tự.
- Tiếng Việt ≈ 3–4 ký tự/token (chỉ để ước lượng thô).
- **Mọi thứ nối vào text đều tính vào ngân sách token**, kể cả tiền tố `"passage: "`.

**Kiểm tra đúng cách:** không chỉ đếm số chunk vượt ngưỡng, mà **giải mã phần bị cắt ra để đọc**. Nội dung nằm ở cuối thường là *điều kiện phủ định* cái nằm ở đầu ("trừ trường hợp…", điểm g/h của một danh sách điều kiện).

### 2. Điều khoản dạng danh sách
Điều "Giải thích từ ngữ" / danh mục / liệt kê điều kiện vừa **dài nhất** vừa **tệ nhất khi gộp 1 vector** (20 chủ đề). Cắt nhỏ nhóm này vừa né truncate vừa **cải thiện chất lượng embedding** — hai lợi ích một công.

### 3. Chọn mô hình embedding — VN-MTEB (EACL 2026)

| Mô hình | VN-MTEB | Context | Ghi chú |
|---|---|---|---|
| `Alibaba-NLP/gte-multilingual-base` | 65.22 | 8192 | cao nhất nhóm base |
| **`intfloat/multilingual-e5-base`** | **62.42** | **512** | đang dùng |
| `hiieu/halong_embedding` | 61.60 | 512 | fine-tune tiếng Việt từ chính e5-base |
| `intfloat/multilingual-e5-small` | 60.66 | 512 | nhẹ hơn |
| `bkai-foundation-models/vietnamese-bi-encoder` | 54.89 | 256 | PhoBERT fine-tune, yếu nhất |

**Tiêu chí chọn:** (1) mạnh tiếng Việt; (2) context đủ cho chunk theo Điều; (3) chạy local, chi phí 0, dữ liệu không rời máy; (4) có benchmark công khai đối chứng.

> **"Chuyên tiếng Việt" ≠ "tốt hơn cho tiếng Việt".** Mô hình đa ngữ lớn học cấu trúc ngữ nghĩa từ khối dữ liệu khổng lồ rồi chuyển giao sang tiếng Việt, thắng mô hình thuần Việt nhỏ.
>
> **Benchmark công khai không thay thế được đo trên dữ liệu của chính mình.**

### 4. Tiền tố của dòng e5
- `"query: "` cho câu hỏi, `"passage: "` cho tài liệu — vì đây là **asymmetric task** (câu hỏi ngắn ↔ đoạn văn dài).
- Tác vụ **symmetric** (so hai câu cùng dạng, phân loại, clustering): dùng `"query: "` cho cả hai.
- Bỏ tiền tố → tụt chất lượng mà **không có lỗi báo ra**.

### 5. `normalize_embeddings=True`
Chuẩn hóa vector về độ dài 1 → cosine similarity rút gọn thành tích vô hướng (`emb @ q`). Kiểm chứng: `np.linalg.norm(emb[0]) == 1.0`.

---

## V. Xử lý dữ liệu thực tế

### 1. Nhận dạng định dạng file
- **PDF text-based vs scan:** không đoán bằng mắt (con dấu đỏ chỉ là ảnh chèn lên, không nói gì về lớp text). Kiểm bằng `len(page.get_text())` — bằng 0 là scan. Hoặc thử bôi đen chữ trong PDF.
- Cần đo **toàn bộ trang**, không chỉ một trang mẫu (một file có thể vừa text-based vừa có trang scan).
- File scan → cần **OCR**; nhưng thường **rẻ hơn nhiều là tìm nguồn khác** có sẵn text (HTML/docx).
- `.doc` (nhị phân cũ) ≠ `.docx` (XML): `python-docx` chỉ đọc `.docx`. Convert bằng Word / LibreOffice (`soffice --headless --convert-to docx`).

### 2. `python-docx` — hai lỗ hổng im lặng
- `doc.paragraphs` **không chứa** nội dung trong **bảng** → phải duyệt riêng `doc.tables`.
- **Không chứa footnote** → nằm trong `word/footnotes.xml` bên trong file zip. Với văn bản hợp nhất, footnote chính là **bản đồ sửa đổi**.

### 3. Đọc HTML (BeautifulSoup)
- **Khoanh vùng khối nội dung trước** (`select_one('div.cldivContentDocVn')`) rồi mới `find_all('p')`; lấy cả trang sẽ kéo theo menu, danh sách văn bản liên quan (212K ký tự so với 103K nội dung thật).
- `re.sub(r'\s+', ' ', t)` — trong HTML, xuống dòng/tab trong source không có nghĩa hiển thị, phải gộp về một khoảng trắng.
- Trang web **đổi giao diện** làm chết mọi URL cũ → lưu HTML về local rồi parse offline: không phụ thuộc mạng, kết quả tái lập được.

### 4. Rác trong văn bản pháp quy
- Header/footer lặp mỗi trang, số trang, khối chữ ký ("Nơi nhận:", "TM. CHÍNH PHỦ") → nếu mọi chunk đều dính cùng một dòng rác, **mọi vector bị kéo về cùng một hướng nhiễu chung** → các chunk "giống nhau giả tạo" → retrieval mất khả năng phân biệt.
- **Rác hay vàng là tùy khâu sau cần gì:** dòng `'Chương I'` / `'NHỮNG QUY ĐỊNH CHUNG'` tách riêng trông như rác, nhưng chính là **tín hiệu cấu trúc** để chunking và điền metadata.

---

## VI. Nguyên tắc kỹ thuật rút ra

1. **Khi phải vá cùng một kiểu lỗi ở nhiều chỗ, lỗi nằm ở hàm chứ không nằm ở dữ liệu.** Thay 4 lần dán tay bằng 1 quy tắc dừng (`break` tại "Nơi nhận:").
2. **Tách I/O khỏi logic:** `read_docx_paragraphs` / `read_html_paragraphs` → cùng đổ vào một `chunk_paragraphs`. Thêm định dạng mới không phải viết lại logic.
3. **Bug flush phần tử cuối:** phần tử cuối không có "phần tử tiếp theo" đến đóng nó → phải append thủ công sau vòng lặp. Kiểm bằng cách đếm và đối chiếu số lượng kỳ vọng.
4. **Khi số liệu mâu thuẫn với kỳ vọng, nghi ngờ cả hai phía** — dữ liệu có thể sai, mà con số kỳ vọng trong đầu cũng có thể sai. Kiểm bằng một phép đo thứ hai độc lập.
5. **Khớp một con số không phải là đã kiểm tra xong** (4 văn bản khớp mà quên mất văn bản thứ 5).
6. **Bất thường trong dữ liệu: truy nguyên nhân trước khi sửa code.** Chunk độ dài 0 hóa ra là Điều "(được bãi bỏ)" — sự thật của dữ liệu, không phải bug.
7. **Đúng kết luận nhưng sai bằng chứng vẫn là chưa đạt** — ở phỏng vấn người ta xoáy vào bằng chứng.
8. **Việc dọn dữ liệu phải in log**, không được im lặng.
9. **Ưu tiên end-to-end trước khi tối ưu một khâu:** một RAG hoàn chỉnh còn thô có giá trị hơn một khâu tiền xử lý hoàn hảo mà chưa có gì để demo.

---

## Thuật ngữ Anh–Việt

| Tiếng Anh | Tiếng Việt |
|---|---|
| Retrieval-Augmented Generation (RAG) | Sinh câu trả lời có tăng cường truy hồi |
| Hallucination | Ảo giác (mô hình bịa) |
| Context window | Cửa sổ ngữ cảnh (giới hạn token đầu vào) |
| Lost in the middle | Mất thông tin ở giữa prompt |
| Chunking / chunk | Cắt đoạn / đoạn văn bản |
| Overlap | Chồng lấn |
| Embedding | Nhúng / vector hóa ngữ nghĩa |
| Vector store / vector database | Kho vector |
| Retrieval / top-k | Truy hồi / k kết quả gần nhất |
| Groundedness / faithfulness | Tính bám nguồn / trung thực với tài liệu |
| Truncation | Cắt cụt (do vượt giới hạn token) |
| Contrastive learning | Học tương phản |
| Bi-encoder | Mã hóa hai nhánh (câu hỏi & tài liệu riêng) |
| Hybrid search | Tìm kiếm lai (từ khóa + vector) |
| Parent-child retrieval | Truy hồi con–cha (small-to-big) |
