"""Khối 2 — Nạp, kiểm tra và thống kê benchmark.

Benchmark là thứ mọi con số ở Khối 3–4 dựa vào. Một nhãn sai ở đây làm hỏng
toàn bộ bảng kết quả mà không có lỗi nào báo, nên file này kiểm tra khá gắt.

Chạy:
    python -m eval.benchmark                     # kiểm tra + thống kê
    python -m eval.benchmark --retrieval-check   # + đo recall của retriever (cần model)
    python -m eval.benchmark --show mh03         # xem chi tiết một câu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from rag.config import BENCHMARK_JSONL

VALID_TYPES = {"single_hop", "multi_hop", "out_of_scope"}
REQUIRED_FIELDS = [
    "id", "type", "user_input", "reference", "reference_chunk_ids",
    "reference_docs", "min_steps", "should_refuse", "tags", "verified_by_human",
]


@dataclass
class BenchmarkItem:
    id: str
    type: str
    user_input: str                 # tên trường đặt theo Ragas -> Khối 3 khỏi ánh xạ lại
    reference: str                  # đáp án chuẩn (ground truth)
    reference_chunk_ids: list[str]  # nhãn để tính context recall KHÔNG cần LLM
    reference_docs: list[str]
    # Nhãn quỹ đạo — CẬN DƯỚI của số lần truy hồi ĐỘC LẬP VỀ NGỮ NGHĨA, tức số
    # câu hỏi con phải tra riêng. KHÔNG phải:
    #   - số chunk chuẩn (nhiều chunk có thể về trong cùng một lần tra: các part
    #     của cùng một Điều, hoặc hai Điều liền nhau lọt cùng top-k), và
    #   - KHÔNG phải kỳ vọng số lần gọi tool của agent (agent có thể cần thêm một
    #     lần `check_amendment` để xác nhận hiệu lực — đó là đường đi ĐÚNG, dài hơn).
    # Vì vậy `steps_over_min` ở eval/metrics_local.py là tín hiệu so với cận dưới,
    # không phải phán quyết về hiệu quả. Xem README §5 "Limitations".
    min_steps: int
    should_refuse: bool
    tags: list[str] = field(default_factory=list)
    verified_by_human: bool = False
    note: str = ""

    @property
    def is_in_scope(self) -> bool:
        return self.type != "out_of_scope"


def load_benchmark(path=BENCHMARK_JSONL) -> list[BenchmarkItem]:
    if not path.exists():
        sys.exit(f"Không thấy {path}")
    items: list[BenchmarkItem] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"Dòng {lineno} không phải JSON hợp lệ: {e}")
        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        if missing:
            sys.exit(f"Dòng {lineno} thiếu trường: {missing}")
        items.append(BenchmarkItem(**{k: raw[k] for k in REQUIRED_FIELDS + ["note"] if k in raw}))
    return items


# --------------------------------------------------------------------------
def validate(items: list[BenchmarkItem], chunk_meta: dict[str, dict] | None) -> list[str]:
    """Trả về danh sách lỗi. Rỗng = benchmark dùng được."""
    errs: list[str] = []

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for it in items:
        if it.id in seen_ids:
            errs.append(f"{it.id}: id bị trùng")
        seen_ids.add(it.id)

        q = it.user_input.strip().lower()
        if q in seen_questions:
            errs.append(f"{it.id}: câu hỏi trùng với {seen_questions[q]}")
        seen_questions[q] = it.id

        if it.type not in VALID_TYPES:
            errs.append(f"{it.id}: type '{it.type}' không hợp lệ")
        if not it.user_input.strip():
            errs.append(f"{it.id}: câu hỏi rỗng")
        if not it.reference.strip():
            errs.append(f"{it.id}: thiếu đáp án chuẩn")

        # Ràng buộc theo loại — đây là chỗ dễ sai nhất khi thêm câu mới
        if it.type == "out_of_scope":
            if not it.should_refuse:
                errs.append(f"{it.id}: out_of_scope phải có should_refuse=true")
            if it.reference_chunk_ids:
                errs.append(f"{it.id}: out_of_scope không được có reference_chunk_ids")
        else:
            if it.should_refuse:
                errs.append(f"{it.id}: câu trong phạm vi không được should_refuse=true")
            if not it.reference_chunk_ids:
                errs.append(f"{it.id}: thiếu reference_chunk_ids -> không tính được recall")

        if it.type == "multi_hop" and it.min_steps < 2:
            errs.append(f"{it.id}: multi_hop mà min_steps={it.min_steps} (phải >= 2)")
        if it.type == "single_hop" and it.min_steps != 1:
            errs.append(f"{it.id}: single_hop mà min_steps={it.min_steps} (phải = 1)")

        # Đối chiếu với index thật: chunk_id có tồn tại không, so_hieu có khớp không
        if chunk_meta is not None:
            for cid in it.reference_chunk_ids:
                if cid not in chunk_meta:
                    errs.append(f"{it.id}: chunk_id không tồn tại trong index -> {cid}")
            docs = {chunk_meta[c]["so_hieu"] for c in it.reference_chunk_ids if c in chunk_meta}
            if docs and set(it.reference_docs) != docs:
                errs.append(
                    f"{it.id}: reference_docs {sorted(it.reference_docs)} "
                    f"không khớp so_hieu suy từ chunk {sorted(docs)}"
                )
    return errs


DEFERRING_PHRASES = [
    "nêu tại Điều", "các tài liệu khác", "các trường hợp khác",
    "theo quy định về nội dung bắt buộc", "và các nội dung khác",
]


def lint_references(items: list[BenchmarkItem]) -> list[str]:
    """Cảnh báo (không phải lỗi): đáp án chuẩn HOÃN lại thay vì trả lời.

    Một `reference` viết kiểu "... và các trường hợp khác nêu tại Điều này" thì
    người đọc không rút ra được câu trả lời, và tệ hơn: nó CHE MẤT việc nhãn
    chunk bị thiếu. Đúng cách này đã lộ ra ở câu mh08 — đáp án vòng vo, và khi
    viết lại cho cụ thể mới phát hiện chunk chứa nội dung cốt lõi chưa được liệt kê.

    Nguyên tắc: `reference` phải suy được TRỌN VẸN từ `reference_chunk_ids`.
    """
    warns: list[str] = []
    for it in items:
        if not it.reference_chunk_ids:
            continue
        for p in DEFERRING_PHRASES:
            if p in it.reference:
                warns.append(f"{it.id}: đáp án chuẩn hoãn lại — chứa \"{p}\"")
    return warns


def stats(items: list[BenchmarkItem]) -> None:
    from collections import Counter

    by_type = Counter(i.type for i in items)
    print(f"  tổng: {len(items)} câu")
    for t in ["single_hop", "multi_hop", "out_of_scope"]:
        print(f"    {t:14s}: {by_type.get(t, 0)}")

    n_ver = sum(1 for i in items if i.verified_by_human)
    flag = "OK" if n_ver == len(items) else "!! CHƯA ĐỦ"
    print(f"  đã người xác minh: {n_ver}/{len(items)}  [{flag}]")

    docs = Counter(d for i in items for d in i.reference_docs)
    print("  phân bố văn bản:")
    for d, n in docs.most_common():
        print(f"    {d:18s} {n}")

    tags = Counter(t for i in items for t in i.tags)
    print("  nhãn:", ", ".join(f"{k}={v}" for k, v in tags.most_common()))

    n_amend = sum(1 for i in items if "amended" in i.tags)
    if n_amend < 3:
        print(f"  [!] chỉ {n_amend} câu gắn 'amended' — thí nghiệm amendment sẽ thiếu tín hiệu")


# --------------------------------------------------------------------------
def retrieval_check(items: list[BenchmarkItem], top_k_list=(3, 5, 10)) -> None:
    """Trần recall của retriever — KHÔNG cần LLM, không tốn quota.

    Trả lời câu: 'nếu retrieval hoàn hảo ở khâu sau, benchmark này còn khả thi
    không?'. Nếu recall đã thấp ở đây thì mọi điểm faithfulness/answer ở Khối 3
    đều bị chặn trên bởi con số này — và tối ưu prompt sẽ vô ích.
    """
    from rag.retriever import RegulationRetriever, RetrievalConfig

    r = RegulationRetriever()
    in_scope = [i for i in items if i.is_in_scope]
    cache: dict[tuple[str, str], list] = {}

    def run(it: BenchmarkItem, cfg) -> list:
        # Một câu + một cấu hình chỉ truy hồi một lần (embed câu hỏi khá đắt)
        key = (it.id, cfg.name)
        if key not in cache:
            cache[key] = r.search(it.user_input, cfg)
        return cache[key]

    print(f"\n== Trần recall trên {len(in_scope)} câu trong phạm vi ==")
    header = f"  {'cấu hình':22s} {'recall':>8s} {'đủ bộ':>8s} {'ký tự ctx':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for k in top_k_list:
        for pc in (False, True):
            cfg = RetrievalConfig(top_k=k, use_parent_child=pc)
            fracs, full, ctx = [], 0, []
            for it in in_scope:
                hits = run(it, cfg)
                got = {c.chunk_id for c in hits}
                want = set(it.reference_chunk_ids)
                found = len(want & got)
                fracs.append(found / len(want))
                full += int(found == len(want))
                ctx.append(sum(len(c.text) for c in hits))
            print(f"  {cfg.name:22s} {sum(fracs)/len(fracs):7.1%} "
                  f"{full:>4}/{len(in_scope):<3} {sum(ctx)//len(ctx):>10,}")

    # Câu nào chưa lấy đủ — danh sách cần xem lại bằng mắt
    cfg = RetrievalConfig(top_k=5, use_parent_child=True)
    print(f"\n  Câu chưa lấy đủ chunk ở {cfg.name}:")
    misses = [
        (it, sorted(set(it.reference_chunk_ids) - {c.chunk_id for c in run(it, cfg)}))
        for it in in_scope
    ]
    misses = [(it, m) for it, m in misses if m]
    for it, m in misses:
        print(f"    {it.id} ({it.type}) thiếu: {m}")
    if not misses:
        print("    (không có — benchmark có thể đang quá dễ)")

    # Out-of-scope: ngưỡng similarity có tách được không?
    oos = [i for i in items if not i.is_in_scope]
    if oos:
        print("\n  Điểm top-1, câu ngoài phạm vi:")
        for it in oos:
            print(f"    {it.id}: {run(it, cfg)[0].score:.4f}  <- {it.user_input[:52]}")
        best_in = min(run(i, cfg)[0].score for i in in_scope)
        worst_oos = max(run(i, cfg)[0].score for i in oos)
        print(f"    thấp nhất TRONG phạm vi = {best_in:.4f} | "
              f"cao nhất NGOÀI phạm vi = {worst_oos:.4f}")
        print(f"    => ngưỡng similarity "
              f"{'TÁCH ĐƯỢC' if best_in > worst_oos else 'KHÔNG tách được'}")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-check", action="store_true")
    ap.add_argument("--show", metavar="ID", help="in chi tiết một câu")
    args = ap.parse_args()

    items = load_benchmark()

    if args.show:
        it = next((i for i in items if i.id == args.show), None)
        if not it:
            sys.exit(f"Không có câu id={args.show}")
        print(json.dumps(it.__dict__, ensure_ascii=False, indent=2))
        return

    # Đối chiếu với index nếu đã dựng
    chunk_meta = None
    try:
        import chromadb
        from rag.config import CHROMA_DIR, COLLECTION_NAME

        col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
        got = col.get(include=["metadatas"])
        chunk_meta = dict(zip(got["ids"], got["metadatas"]))
    except Exception as e:
        print(f"  [!] không đọc được index ({e}) — bỏ qua kiểm tra chunk_id")

    print("== Kiểm tra benchmark ==")
    errs = validate(items, chunk_meta)
    if errs:
        print(f"  {len(errs)} LỖI:")
        for e in errs:
            print(f"    - {e}")
    else:
        print("  không có lỗi cấu trúc")

    warns = lint_references(items)
    if warns:
        print(f"  {len(warns)} CẢNH BÁO (không chặn, nhưng nên sửa):")
        for w in warns:
            print(f"    - {w}")

    stats(items)

    if args.retrieval_check:
        retrieval_check(items)

    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
