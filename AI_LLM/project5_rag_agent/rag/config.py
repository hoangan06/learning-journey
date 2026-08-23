"""Cấu hình dùng chung cho toàn bộ Project 5.

Mọi đường dẫn suy ra từ vị trí file này -> chạy được từ bất kỳ thư mục nào,
không phụ thuộc ổ đĩa hay cwd.
"""

from pathlib import Path

# --- Đường dẫn -------------------------------------------------------------
RAG_DIR = Path(__file__).resolve().parent            # .../project5_rag_agent/rag
PROJECT5_DIR = RAG_DIR.parent                        # .../project5_rag_agent
AI_LLM_DIR = PROJECT5_DIR.parent                     # .../AI_LLM
PROJECT4_DATA = AI_LLM_DIR / "project4_rag_assistant" / "data"

# Nguồn dữ liệu: TÁI DÙNG Project 4, KHÔNG copy, KHÔNG sửa (read-only).
CHUNKS_JSON = PROJECT4_DATA / "chunks.json"
EMBEDDINGS_NPY = PROJECT4_DATA / "embeddings.npy"

DATA_DIR = PROJECT5_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"                  # đã .gitignore
BENCHMARK_JSONL = DATA_DIR / "benchmark.jsonl"       # Khối 2
RESULTS_DIR = PROJECT5_DIR / "results"

RUNS_DIR = RESULTS_DIR / "runs"

COLLECTION_NAME = "fmcg_regulations"

# --- LLM -------------------------------------------------------------------
# Đổi model qua biến môi trường GEMINI_MODEL nếu tài khoản không có model này.
# `python -m eval.doctor` liệt kê các model mà tài khoản thực sự gọi được.
import os as _os

GEMINI_MODEL = _os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MAX_TOKENS = 2048
GEMINI_CONTEXT_WINDOW = 1_000_000
LLM_TEMPERATURE = 0.0          # 0 để giảm phương sai giữa các lần chạy eval

# Free tier bị giới hạn số request/phút. Nghỉ giữa các lần gọi cho an toàn;
# chỉnh bằng --sleep. Hạn mức thật xem tại https://aistudio.google.com/rate-limit
LLM_SLEEP_SECONDS = 4.0
LLM_MAX_RETRIES = 5

# Câu từ chối CỐ ĐỊNH. Bắt buộc model trả lời đúng chuỗi này khi không đủ căn cứ
# -> việc đo "có từ chối hay không" thành phép so chuỗi xác định, không cần LLM chấm.
REFUSAL_SENTINEL = "Không tìm thấy quy định liên quan trong 5 văn bản được cung cấp."

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu quy định pháp luật cho chuỗi cung ứng FMCG tại Việt Nam.
Kho tài liệu chỉ gồm 5 văn bản: Luật An toàn thực phẩm (VBHN 61), Luật Hải quan (VBHN 54),
Nghị định 15/2018/NĐ-CP, Nghị định 43/2017/NĐ-CP và Nghị định 111/2021/NĐ-CP.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng thông tin có trong CÁC ĐOẠN TRÍCH được cung cấp. Không dùng kiến thức bên ngoài.
2. Mọi khẳng định phải kèm số đoạn trích, dạng [1], [2].
3. Nếu một đoạn có ghi chú "LƯU Ý: đã bị sửa bởi ...", phải nêu rõ điều đó trong câu trả lời
   và ưu tiên nội dung sửa đổi nếu nội dung sửa đổi có trong các đoạn trích.
4. Nếu các đoạn trích KHÔNG đủ căn cứ để trả lời, hãy trả lời DUY NHẤT bằng đúng câu sau,
   không thêm gì khác:
   "{refusal}"
5. Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm câu hỏi.""".format(
    refusal=REFUSAL_SENTINEL
)

# --- Embedding -------------------------------------------------------------
# PHẢI trùng model đã sinh ra embeddings.npy ở Project 4, nếu không thì
# vector câu hỏi và vector văn bản nằm ở hai không gian khác nhau -> rác.
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIM = 768

# e5 được huấn luyện bất đối xứng: văn bản gắn "passage: ", câu hỏi gắn "query: ".
# Bỏ prefix hoặc dùng nhầm prefix làm chất lượng truy hồi tụt rõ rệt.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# --- Mặc định cho retrieval ------------------------------------------------
DEFAULT_TOP_K = 5

# 5 văn bản trong kho. Lọc theo `so_hieu`, KHÔNG lọc theo `bo_luat`:
# NĐ 43/2017 và NĐ 111/2021 dùng chung một giá trị `bo_luat`.
DOC_TITLES = {
    "61/VBHN-VPQH": "Luật An toàn thực phẩm (VBHN 61/2025)",
    "54/VBHN-VPQH": "Luật Hải quan (VBHN 54/2026)",
    "15/2018/NĐ-CP": "NĐ 15/2018 (hướng dẫn Luật ATTP)",
    "43/2017/NĐ-CP": "NĐ 43/2017 (nhãn hàng hóa)",
    "111/2021/NĐ-CP": "NĐ 111/2021 (sửa đổi NĐ 43/2017)",
}
