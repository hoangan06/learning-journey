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

COLLECTION_NAME = "fmcg_regulations"

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
