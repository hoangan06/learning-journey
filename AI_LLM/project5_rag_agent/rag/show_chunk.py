"""Xem nguyên văn một hoặc nhiều chunk — công cụ để XÁC MINH nhãn benchmark.

Không có nó thì việc "kiểm tra lại đáp án chuẩn" phải mở file JSON 900KB và
tìm bằng mắt. Có nó thì một lệnh là ra.

Chạy:
    python -m rag.show_chunk 15/2018/NĐ-CP_Điều_6
    python -m rag.show_chunk --grep "36 tháng tuổi"
    python -m rag.show_chunk --dieu 15 --doc 43/2017/NĐ-CP
    python -m rag.show_chunk --bench mh03        # xem mọi chunk chuẩn của một câu benchmark
"""

from __future__ import annotations

import argparse
import json
import sys

from rag.config import CHUNKS_JSON


def load() -> list[dict]:
    return json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))


def show(c: dict, full: bool = True) -> None:
    print("=" * 78)
    print(f"{c['chunk_id']}")
    print(f"  văn bản : {c['so_hieu']} — {c['bo_luat']}")
    print(f"  chương  : {c['so_chuong']} {c['ten_chuong']}")
    print(f"  điều    : {c['so_dieu']} — {c['ten_dieu']}")
    print(f"  parent  : {c['parent_id']}")
    if c["sua_doi"]:
        print(f"  !! SỬA ĐỔI: {c['sua_doi']}")
    print(f"  nguồn   : {c['nguon']}")
    print("-" * 78)
    print(c["text_for_embedding"] if full else c["text_for_embedding"][:400] + "...")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk_ids", nargs="*")
    ap.add_argument("--grep", help="tìm chuỗi trong nội dung")
    ap.add_argument("--dieu", help="lọc theo số Điều")
    ap.add_argument("--doc", help="lọc theo so_hieu, vd 43/2017/NĐ-CP")
    ap.add_argument("--bench", metavar="ID", help="xem chunk chuẩn của một câu benchmark")
    ap.add_argument("--short", action="store_true", help="chỉ in 400 ký tự đầu")
    a = ap.parse_args()

    chunks = load()
    by_id = {c["chunk_id"]: c for c in chunks}
    picked: list[dict] = []

    if a.bench:
        from eval.benchmark import load_benchmark

        it = next((i for i in load_benchmark() if i.id == a.bench), None)
        if not it:
            sys.exit(f"Không có câu benchmark id={a.bench}")
        print(f"\n### {it.id} [{it.type}] {it.user_input}")
        print(f"### Đáp án chuẩn (CẦN XÁC MINH):\n{it.reference}\n")
        picked = [by_id[c] for c in it.reference_chunk_ids if c in by_id]

    for cid in a.chunk_ids:
        if cid in by_id:
            picked.append(by_id[cid])
        else:
            near = [i for i in by_id if i.startswith(cid)]
            if near:
                print(f"[!] '{cid}' không có; khớp tiền tố: {near}")
                picked += [by_id[i] for i in near]
            else:
                print(f"[!] không tìm thấy '{cid}'")

    if a.grep:
        picked += [c for c in chunks if a.grep.lower() in c["text_for_embedding"].lower()]
    if a.dieu:
        picked += [c for c in chunks if c["so_dieu"] == a.dieu
                   and (not a.doc or c["so_hieu"] == a.doc)]
    elif a.doc and not picked:
        picked += [c for c in chunks if c["so_hieu"] == a.doc]

    seen: set[str] = set()
    n = 0
    for c in picked:
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"])
        show(c, full=not a.short)
        n += 1
    if n == 0:
        print("Không có kết quả.")
    else:
        print(f"({n} chunk)")


if __name__ == "__main__":
    main()
