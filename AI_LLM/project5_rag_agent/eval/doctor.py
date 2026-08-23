"""Kiểm tra môi trường TRƯỚC khi đốt quota — và in đúng lệnh cần chạy để sửa.

Có file này vì một lý do cụ thể đã kiểm chứng: `pip install ragas==0.4.3` trên
môi trường sạch sẽ kéo về langchain 1.x, và khi đó `import ragas` HỎNG NGAY:

    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

(ragas 0.4.3 vẫn import đường dẫn đã bị xoá ở langchain-community 0.4.)
Không phát hiện sớm thì rất dễ mất hàng giờ đi tìm lỗi trong code của chính dự án.

Chạy:  python -m eval.doctor
"""

from __future__ import annotations

import importlib.metadata as md
import sys

OK, BAD, WARN = "  [OK]  ", "  [LỖI] ", "  [!]   "

FIX_RAGAS = (
    'pip install "langchain<1.0" "langchain-core<1.0" "langchain-community<0.4" '
    '"langchain-openai<1.0" "instructor>=1.6" rapidfuzz pandas'
)


def ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def check_packages() -> list[str]:
    fixes: list[str] = []
    print("== Thư viện ==")
    need = ["chromadb", "sentence-transformers", "llama-index-core",
            "llama-index-llms-google-genai", "ragas", "openai"]
    for p in need:
        v = ver(p)
        print(f"{OK if v else BAD}{p:32s} {v or 'CHƯA CÀI'}")
        if not v:
            fixes.append(f"pip install {p}")

    print("\n== Ràng buộc phiên bản của ragas 0.4.3 ==")
    pins = {"langchain": (1, 0), "langchain-core": (1, 0),
            "langchain-community": (0, 4), "langchain-openai": (1, 0)}
    bad_pin = False
    for p, maxv in pins.items():
        v = ver(p)
        if v is None:
            print(f"{WARN}{p:32s} chưa cài")
            continue
        parts = tuple(int(x) for x in v.split(".")[:2])
        good = parts < maxv
        bad_pin |= not good
        print(f"{OK if good else BAD}{p:32s} {v}  (cần < {maxv[0]}.{maxv[1]})")

    iv = ver("instructor")
    if iv:
        good = tuple(int(x) for x in iv.split(".")[:2]) >= (1, 6)
        bad_pin |= not good
        print(f"{OK if good else BAD}{'instructor':32s} {iv}  (cần >= 1.6)")
    for p in ("rapidfuzz", "pandas"):
        v = ver(p)
        print(f"{OK if v else BAD}{p:32s} {v or 'CHƯA CÀI (ragas cần)'}")
        bad_pin |= v is None
    if bad_pin:
        fixes.append(FIX_RAGAS)
    return fixes


def check_imports() -> list[str]:
    fixes: list[str] = []
    print("\n== Import thử ==")
    for label, mod in [("chromadb", "chromadb"),
                       ("llama_index core agent", "llama_index.core.agent.workflow"),
                       ("GoogleGenAI", "llama_index.llms.google_genai"),
                       ("ragas", "ragas")]:
        try:
            __import__(mod)
            print(f"{OK}{label}")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD}{label}: {type(e).__name__}: {str(e)[:100]}")
            if mod == "ragas":
                fixes.append(FIX_RAGAS)
    return fixes


def check_data() -> list[str]:
    fixes: list[str] = []
    print("\n== Dữ liệu ==")
    from rag.config import BENCHMARK_JSONL, CHROMA_DIR, CHUNKS_JSON, EMBEDDINGS_NPY

    for label, p in [("chunks.json (Project 4)", CHUNKS_JSON),
                     ("embeddings.npy (Project 4)", EMBEDDINGS_NPY),
                     ("benchmark.jsonl", BENCHMARK_JSONL),
                     ("chroma_db/", CHROMA_DIR)]:
        print(f"{OK if p.exists() else BAD}{label:32s} {p}")
        if not p.exists() and "chroma" in label:
            fixes.append("python -m rag.build_index --force")

    try:
        import chromadb

        from rag.config import COLLECTION_NAME
        n = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME).count()
        print(f"{OK if n == 291 else WARN}{'số bản ghi trong index':32s} {n} (mong đợi 291)")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD}{'đọc index':32s} {type(e).__name__}: {str(e)[:70]}")
        fixes.append("python -m rag.build_index --force")

    try:
        from eval.benchmark import load_benchmark

        items = load_benchmark()
        n_ver = sum(1 for i in items if i.verified_by_human)
        print(f"{OK}{'benchmark':32s} {len(items)} câu, "
              f"{n_ver} đã xác minh tay")
        if n_ver < len(items):
            print(f"{WARN}{'':32s} còn {len(items) - n_ver} câu chưa xác minh "
                  "-> phải nêu trong README")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD}benchmark: {e}")
    return fixes


def check_llm() -> list[str]:
    print("\n== Khóa API và model ==")
    try:
        from rag.pipeline import load_api_key

        key = load_api_key()
        print(f"{OK}{'tìm thấy khóa Gemini':32s} (…{key[-4:]}, không in ra đầy đủ)")
    except SystemExit as e:
        print(f"{BAD}{str(e)[:200]}")
        return ["đặt GOOGLE_API_KEY hoặc tạo .env"]

    try:
        from google import genai

        from rag.config import GEMINI_MODEL

        client = genai.Client(api_key=key)
        names = []
        for m in client.models.list():
            nm = getattr(m, "name", "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions or not actions:
                names.append(nm)
        gem = sorted(n for n in names if n.startswith("gemini"))
        print(f"{OK}{'gọi được API':32s} {len(gem)} model gemini khả dụng")
        print(f"         đang cấu hình dùng: {GEMINI_MODEL}"
              f"  {'(có trong danh sách)' if GEMINI_MODEL in gem else '(KHÔNG THẤY — đổi GEMINI_MODEL)'}")
        print("         ví dụ: " + ", ".join(gem[:8]))
    except Exception as e:  # noqa: BLE001
        print(f"{WARN}không liệt kê được model: {type(e).__name__}: {str(e)[:110]}")
    return []


def main() -> None:
    print(f"Python {sys.version.split()[0]}\n")
    fixes = check_packages() + check_imports() + check_data() + check_llm()
    print("\n" + "=" * 70)
    if fixes:
        print("CẦN CHẠY:")
        for f in dict.fromkeys(fixes):
            print(f"  {f}")
    else:
        print("Môi trường sẵn sàng. Thứ tự chạy:")
        print("  python -m eval.experiments                      # không cần LLM")
        print("  python -m eval.generate --mode baseline")
        print("  python -m eval.generate --mode agent")
        print("  python -m eval.run_eval --tags baseline,agent")
        print("  python -m eval.ragas_eval --tag baseline")


if __name__ == "__main__":
    main()
