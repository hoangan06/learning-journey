"""Khối 1 — Kiểm chứng: Chroma có trả về đúng kết quả như brute-force của Project 4 không?

Đây là bước dễ bị bỏ qua nhất và cũng là bước đáng giá nhất. Nếu chuyển sang
vector store mà kết quả truy hồi lệch đi, thì MỌI con số so sánh ở Khối 3–4
đều đang đo lẫn một lỗi migration — và ta sẽ tưởng đó là hiệu ứng của biến
thí nghiệm. Chạy cái này trước khi tin bất kỳ bảng kết quả nào.

Chạy:
    python -m rag.verify_migration              # nhanh, KHÔNG cần tải model
    python -m rag.verify_migration --queries    # thêm 5 câu hỏi thật (cần model + GPU/CPU)
"""

from __future__ import annotations

import argparse
import json

import chromadb
import numpy as np

from rag.config import (
    CHROMA_DIR,
    CHUNKS_JSON,
    COLLECTION_NAME,
    E5_QUERY_PREFIX,
    EMBED_MODEL_NAME,
    EMBEDDINGS_NPY,
)

TOP_K = 5
N_PROBE = 25  # số vector đem ra dò

# 5 câu của Project 4 — câu cuối cố tình NGOÀI PHẠM VI (luật lao động),
# giữ nguyên để đối chiếu hành vi giữa hai project.
QUERIES = [
    "thực phẩm bao gói sẵn là gì",
    "hồ sơ tự công bố sản phẩm gồm những gì",
    "nhãn hàng hóa nhập khẩu bắt buộc ghi những nội dung nào",
    "thời hạn nộp tờ khai hải quan",
    "nhân viên nghỉ thai sản mấy ngày",
]


def brute_force(emb: np.ndarray, q: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Đúng công thức Project 4: vector đã chuẩn hoá nên tích vô hướng = cosine."""
    scores = emb @ q
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def compare(tag, bf_ids, ch_ids, bf_scores, ch_scores, verbose=False) -> tuple[int, float]:
    overlap = len(set(bf_ids) & set(ch_ids))
    same_order = bf_ids == ch_ids
    diff = max(abs(a - b) for a, b in zip(bf_scores, ch_scores)) if same_order else float("nan")
    if verbose or overlap < len(bf_ids):
        flag = "OK " if overlap == len(bf_ids) else "!! "
        print(f"  {flag}{tag}: overlap {overlap}/{len(bf_ids)}"
              f"{'' if same_order else '  (thứ tự khác)'}"
              f"{f'  Δscore max {diff:.2e}' if same_order else ''}")
        if overlap < len(bf_ids):
            print(f"      brute : {bf_ids}")
            print(f"      chroma: {ch_ids}")
    return overlap, diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", action="store_true", help="chạy thêm 5 câu hỏi thật")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    emb = np.load(EMBEDDINGS_NPY)
    ids = [c["chunk_id"] for c in chunks]

    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)

    print("== Kiểm chứng migration Project 4 -> Chroma ==")
    print(f"  index: {col.count()} bản ghi | nguồn: {len(chunks)} chunk")
    assert col.count() == len(chunks), "Số bản ghi không khớp!"

    # -- 1. Văn bản lưu trong Chroma có đúng text đã embed không --------------
    sample = col.get(ids=ids[:50], include=["documents"])
    by_id = dict(zip(sample["ids"], sample["documents"]))
    bad = [c["chunk_id"] for c in chunks[:50]
           if by_id.get(c["chunk_id"]) != c["text_for_embedding"]]
    print(f"  [1] document khớp text_for_embedding: {'OK' if not bad else f'LỆCH {bad[:3]}'}")

    # -- 2. Láng giềng gần nhất: brute-force vs Chroma ------------------------
    # Dùng chính vector của chunk làm câu hỏi -> không cần tải embedding model.
    step = max(1, len(emb) // N_PROBE)
    probes = list(range(0, len(emb), step))[:N_PROBE]
    overlaps, diffs = [], []
    print(f"  [2] so khớp top-{TOP_K} trên {len(probes)} vector mẫu:")
    for i in probes:
        q = emb[i]
        bf = brute_force(emb, q, TOP_K)
        res = col.query(query_embeddings=[q.astype(float).tolist()],
                        n_results=TOP_K, include=["distances"])
        o, d = compare(
            ids[i][:40],
            [ids[j] for j, _ in bf],
            res["ids"][0],
            [s for _, s in bf],
            [1.0 - x for x in res["distances"][0]],
            verbose=args.verbose,
        )
        overlaps.append(o)
        if d == d:  # loại NaN
            diffs.append(d)

    mean_overlap = sum(overlaps) / len(overlaps) / TOP_K
    print(f"      overlap trung bình : {mean_overlap:.1%}"
          f"  ({sum(1 for o in overlaps if o == TOP_K)}/{len(probes)} trùng khít)")
    if diffs:
        print(f"      sai số điểm tối đa : {max(diffs):.2e}")
    if mean_overlap < 0.99:
        print("      [!] Lệch đáng kể — KHÔNG chạy Khối 3/4 trước khi tìm ra nguyên nhân.")

    # -- 3. Liên kết cấu trúc -------------------------------------------------
    all_meta = col.get(include=["metadatas"])["metadatas"]
    n_multi = sum(1 for m in all_meta if m["n_parts"] > 1)
    n_amended = sum(1 for m in all_meta if m["is_amended"])
    n_linked = sum(1 for m in all_meta if m["amend_refs"])
    print(f"  [3] parent-child: {n_multi} chunk thuộc Điều nhiều part")
    print(f"      amendment   : {n_amended} chunk bị sửa, {n_linked} nối được tới khoản sửa đổi")
    if n_amended != n_linked:
        print(f"      [!] {n_amended - n_linked} chunk có sua_doi nhưng không phân giải được")

    # -- 4. Câu hỏi thật (tuỳ chọn) ------------------------------------------
    if args.queries:
        from sentence_transformers import SentenceTransformer

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        print(f"\n  [4] 5 câu hỏi thật (model trên {device}):")
        model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        for q in QUERIES:
            v = model.encode([E5_QUERY_PREFIX + q], normalize_embeddings=True)[0]
            bf = brute_force(emb, v, TOP_K)
            res = col.query(query_embeddings=[v.astype(float).tolist()],
                            n_results=TOP_K, include=["distances"])
            compare(q, [ids[j] for j, _ in bf], res["ids"][0],
                    [s for _, s in bf], [1.0 - x for x in res["distances"][0]],
                    verbose=True)
            top = bf[0]
            print(f"      top1 = {ids[top[0]]}  sim={top[1]:.4f}")

    print("\n  Xong. Nếu mục [2] đạt 100% thì Chroma tương đương brute-force của Project 4.")


if __name__ == "__main__":
    main()
