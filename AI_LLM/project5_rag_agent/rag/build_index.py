"""Khối 1 — Nạp 291 chunk của Project 4 vào Chroma (persistent).

Chạy:  python -m rag.build_index          (từ thư mục project5_rag_agent)
       python -m rag.build_index --force  (xoá collection cũ, dựng lại)

KHÔNG tính lại embedding: tái dùng nguyên `embeddings.npy` của Project 4.
Việc build vì thế chỉ mất vài giây và không cần GPU.

Ngoài việc nạp vector, script còn dựng sẵn hai liên kết ở tầng metadata:
  - parent/part : nhóm các chunk bị cắt nhỏ của cùng một Điều (parent-child)
  - amendment   : nối Điều gốc <-> khoản sửa đổi ở văn bản khác
Cả hai đều tính một lần lúc build để lúc truy vấn khỏi phải xử lý chuỗi.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any

import numpy as np

try:
    import chromadb
except ImportError:  # pragma: no cover
    sys.exit("Thiếu chromadb. Cài: pip install chromadb")

from rag.config import (
    CHROMA_DIR,
    CHUNKS_JSON,
    COLLECTION_NAME,
    EMBED_DIM,
    EMBEDDINGS_NPY,
)

# "đã bị sửa bởi Khoản 4, Điều 1, Nghị định 111/2021/NĐ-CP"
# "đã bị sửa bởi Điều 2, Nghị định 111/2021/NĐ-CP"   (không có Khoản)
AMEND_RE = re.compile(
    r"(?:Khoản\s+(\d+)\s*,\s*)?Điều\s+(\d+)\s*,\s*Nghị định\s+(\S+)",
    re.IGNORECASE,
)
PART_RE = re.compile(r"_part_(\d+)$")

SEP = "|"  # metadata của Chroma chỉ nhận scalar -> list phải nối thành chuỗi


# --------------------------------------------------------------------------
# Nạp và kiểm tra dữ liệu nguồn
# --------------------------------------------------------------------------
def load_source() -> tuple[list[dict[str, Any]], np.ndarray]:
    if not CHUNKS_JSON.exists():
        sys.exit(f"Không thấy {CHUNKS_JSON}")
    if not EMBEDDINGS_NPY.exists():
        sys.exit(f"Không thấy {EMBEDDINGS_NPY}")

    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    emb = np.load(EMBEDDINGS_NPY)

    # Ba bất biến. Sai bất kỳ cái nào thì mọi số đo sau này đều vô nghĩa,
    # nên dừng ngay tại đây thay vì để nó hỏng âm thầm.
    if len(chunks) != emb.shape[0]:
        sys.exit(f"Lệch số dòng: {len(chunks)} chunk vs {emb.shape[0]} vector")
    if emb.shape[1] != EMBED_DIM:
        sys.exit(f"Sai số chiều: {emb.shape[1]}, mong đợi {EMBED_DIM}")

    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        sys.exit(
            f"Vector chưa chuẩn hoá L2 (norm {norms.min():.4f}..{norms.max():.4f}). "
            "Cosine sẽ sai."
        )

    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        sys.exit("chunk_id bị trùng — Chroma sẽ ghi đè âm thầm")

    return chunks, emb


# --------------------------------------------------------------------------
# Liên kết sửa đổi (amendment)
# --------------------------------------------------------------------------
def resolve_amendments(
    chunks: list[dict[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Parse trường `sua_doi` thành chunk_id thật.

    Trả về:
      amend_refs : Điều gốc      -> [chunk_id của khoản sửa đổi]
      amends     : khoản sửa đổi -> [chunk_id của Điều gốc]   (chiều ngược)
      unresolved : các chuỗi sua_doi không parse/không khớp được
    """
    all_ids = {c["chunk_id"] for c in chunks}
    amend_refs: dict[str, list[str]] = {}
    amends: dict[str, list[str]] = defaultdict(list)
    unresolved: list[str] = []

    for c in chunks:
        raw = (c.get("sua_doi") or "").strip()
        if not raw:
            continue
        m = AMEND_RE.search(raw)
        if not m:
            unresolved.append(f"{c['chunk_id']}: {raw}")
            continue

        khoan, dieu, so_hieu = m.group(1), m.group(2), m.group(3).rstrip(".,;")
        base = f"{so_hieu}_Điều_{dieu}"
        if khoan:
            base += f"_Khoản_{khoan}"

        # Khoản sửa đổi cũng có thể bị cắt thành nhiều part.
        if base in all_ids:
            targets = [base]
        else:
            targets = sorted(i for i in all_ids if i.startswith(base + "_part_"))

        if not targets:
            unresolved.append(f"{c['chunk_id']}: {raw} -> không tìm thấy '{base}'")
            continue

        amend_refs[c["chunk_id"]] = targets
        for t in targets:
            amends[t].append(c["chunk_id"])

    return amend_refs, dict(amends), unresolved


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def build_metadatas(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_sizes: dict[str, int] = defaultdict(int)
    for c in chunks:
        parent_sizes[c["parent_id"]] += 1

    amend_refs, amends, unresolved = resolve_amendments(chunks)
    if unresolved:
        print(f"  [!] {len(unresolved)} liên kết sửa đổi không phân giải được:")
        for u in unresolved:
            print(f"      {u}")

    metas: list[dict[str, Any]] = []
    for c in chunks:
        m = PART_RE.search(c["chunk_id"])
        # part_index: dùng để xếp lại đúng thứ tự khi ghép các part của một Điều.
        # KHÔNG lấy từ trường `part` của Project 4 — 281/291 chunk bỏ trống nó.
        part_index = int(m.group(1)) if m else 0

        metas.append(
            {
                "chunk_id": c["chunk_id"],
                "so_hieu": c["so_hieu"],          # khoá phân biệt văn bản (KHÔNG dùng bo_luat)
                "bo_luat": c["bo_luat"],
                "so_dieu": str(c["so_dieu"]),
                "ten_dieu": c["ten_dieu"],
                "so_chuong": c["so_chuong"],
                "ten_chuong": c["ten_chuong"],
                "ngay_ky": c["ngay_ky"],
                "nguon": c["nguon"],
                "parent_id": c["parent_id"],
                "part_index": part_index,
                "n_parts": parent_sizes[c["parent_id"]],
                "sua_doi": c.get("sua_doi", "") or "",
                "is_amended": bool(c.get("sua_doi")),
                "amend_refs": SEP.join(amend_refs.get(c["chunk_id"], [])),
                "amends": SEP.join(amends.get(c["chunk_id"], [])),
                "n_chars": len(c["text_for_embedding"]),
            }
        )

    n_multi = sum(1 for v in parent_sizes.values() if v > 1)
    print(f"  parent-child : {n_multi} Điều bị cắt thành nhiều part")
    print(f"  amendment    : {len(amend_refs)} Điều gốc -> {len(amends)} khoản sửa đổi")
    return metas


# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------
def get_collection(client, force: bool):
    """Tạo collection với cosine space.

    Chroma 1.x đổi cách khai báo space (`configuration=` thay cho `metadata=`)
    nên thử bản mới trước, lỗi thì rơi về bản cũ — khỏi phải ghim version.
    """
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  đã xoá collection cũ '{COLLECTION_NAME}'")
        except Exception:
            pass

    kwargs = dict(name=COLLECTION_NAME, embedding_function=None)
    try:
        return client.get_or_create_collection(
            **kwargs, configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:
        return client.get_or_create_collection(
            **kwargs, metadata={"hnsw:space": "cosine"}
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="xoá collection cũ rồi dựng lại")
    args = ap.parse_args()

    print("== Khối 1: dựng Chroma index ==")
    chunks, emb = load_source()
    print(f"  nguồn        : {len(chunks)} chunk, vector {emb.shape} (đã chuẩn hoá L2)")

    metas = build_metadatas(chunks)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = get_collection(client, force=args.force)

    if col.count() > 0 and not args.force:
        sys.exit(
            f"  Collection đã có {col.count()} bản ghi. Dùng --force để dựng lại."
        )

    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=emb.astype(float).tolist(),
        documents=[c["text_for_embedding"] for c in chunks],  # đúng text đã embed
        metadatas=metas,
    )

    print(f"  -> đã nạp {col.count()} bản ghi vào {CHROMA_DIR}")
    print("  Bước tiếp: python -m rag.verify_migration")


if __name__ == "__main__":
    main()
