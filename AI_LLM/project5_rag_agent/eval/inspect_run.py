"""Đối chiếu tay — công cụ để KIỂM TRA LẠI điểm Ragas bằng mắt.

Ragas là LLM-as-judge: nó dùng một LLM để chấm output của một LLM. Ba rủi ro đã
biết — tự quy chiếu, phương sai giữa các lần chấm, và điểm số trông chính xác hơn
bản chất. Cách xử lý duy nhất trung thực là mở vài câu ra đọc và so bằng tay.

File này in cạnh nhau: câu hỏi, đáp án chuẩn, câu trả lời của hệ thống, quỹ đạo,
các chunk đã lấy về (đánh dấu chunk chuẩn), và điểm Ragas nếu đã chấm.

Chạy:
    python -m eval.inspect_run --tag baseline --id mh03
    python -m eval.inspect_run --tag agent --type multi_hop
    python -m eval.inspect_run --tag baseline --wrong        # chỉ câu từ chối sai / thiếu chunk
"""

from __future__ import annotations

import argparse
import json

from eval.benchmark import load_benchmark
from eval.generate import run_path
from eval.metrics_local import score_one
from rag.config import RESULTS_DIR


def _fmt(v, nd: int = 3) -> str:
    """In số an toàn khi giá trị có thể là None.

    `score_one()` trả None cho recall/precision/mrr ở câu ngoài phạm vi — đúng
    theo thiết kế: không có gold chunk thì các đại lượng đó không xác định, và
    trả 0.0 sẽ là nói dối. Nhưng None mà đem format `:.3f` thì ném TypeError.
    """
    return "—" if v is None else f"{float(v):.{nd}f}"



def load_ragas(tag: str) -> dict[str, dict]:
    p = RESULTS_DIR / f"ragas_{tag}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {r["id"]: r for r in data.get("per_sample", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--id", default="")
    ap.add_argument("--type", default="")
    ap.add_argument("--wrong", action="store_true")
    ap.add_argument("--full-context", action="store_true")
    a = ap.parse_args()

    bench = {i.id: i for i in load_benchmark()}
    ragas = load_ragas(a.tag)
    p = run_path(a.tag)
    if not p.exists():
        raise SystemExit(f"Chưa có {p}")

    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        it = bench.get(r["id"])
        if it is None:
            continue
        if a.id and r["id"] != a.id:
            continue
        if a.type and r["type"] != a.type:
            continue
        s = score_one(it, r)
        if a.wrong and s["refusal_correct"] and not s["missing_chunks"]:
            continue

        n += 1
        print("=" * 80)
        print(f"{it.id}  [{it.type}]  tags={it.tags}")
        print(f"CÂU HỎI     : {it.user_input}")
        print(f"\nĐÁP ÁN CHUẨN:\n  {it.reference}")
        print(f"\nHỆ THỐNG TRẢ LỜI ({r['mode']}):\n  " + (r["answer"] or "(rỗng)").replace("\n", "\n  "))
        if r.get("error"):
            print(f"\nLỖI: {r['error']}")

        if r.get("trajectory"):
            print("\nQUỸ ĐẠO:")
            for i, st in enumerate(r["trajectory"], 1):
                if st["type"] == "tool_call":
                    print(f"  {i}. -> {st['name']}({st['args']})")
                elif st["type"] == "tool_result":
                    print(f"  {i}. <- {st['name']}: {st['output_chars']} ký tự"
                          f"{' [LỖI]' if st.get('is_error') else ''}")
                elif st["type"] == "retrieve":
                    print(f"  {i}. truy hồi {st['n_chunks']} đoạn ({st['config']})")
                else:
                    print(f"  {i}. LLM: {str(st.get('text', ''))[:100]}...")

        want = set(it.reference_chunk_ids)
        print("\nCÁC ĐOẠN LẤY VỀ (* = chunk chuẩn):")
        for i, (cid, org) in enumerate(zip(r["chunk_ids"], r["origins"]), 1):
            mark = "*" if cid in want else " "
            print(f"  {mark}[{i}] {cid}  ({org})")
        if s["missing_chunks"]:
            print(f"  THIẾU: {s['missing_chunks']}")

        print(f"\nMETRIC TẤT ĐỊNH: recall={_fmt(s['context_recall_id'])} "
              # mrr/recall/precision là None ở câu ngoài phạm vi (không có gold chunk).
              # Format thẳng bằng :.3f sẽ ném TypeError -> chết đúng ở `--id oos01`.
              f"prec={_fmt(s['context_precision_id'])} mrr={_fmt(s['mrr'])} "
              f"| từ chối: nên={s['should_refuse']} thực tế={s['refused']} "
              f"-> {'ĐÚNG' if s['refusal_correct'] else 'SAI'}"
              f" | trích dẫn={s['has_citation']}")
        if r["id"] in ragas:
            sc = {k: v for k, v in ragas[r["id"]].items() if k != "id"}
            print(f"ĐIỂM RAGAS      : {sc}")
            print("  ^ đối chiếu điểm này với đánh giá thủ công ở trên."
                  "\n    Chênh lệch lớn = LLM-as-judge không đáng tin ở ca này;"
                  "\n    ghi ca đó vào phần limitations của README.")

        if a.full_context:
            print("\nNGỮ CẢNH ĐẦY ĐỦ:")
            for i, c in enumerate(r["contexts"], 1):
                print(f"  --- [{i}] ---\n{c}\n")
        print()

    print(f"({n} câu)")


if __name__ == "__main__":
    main()