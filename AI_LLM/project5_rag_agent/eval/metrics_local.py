"""Metric KHÔNG cần LLM — chạy trong vài giây, không tốn quota, không có phương sai.

Vì sao tách riêng khỏi Ragas:
  - Ragas dùng LLM-as-judge: chậm, tốn quota, và chấm lại có thể ra điểm khác.
  - Các đại lượng ở đây là phép so tập hợp và so chuỗi -> tất định, tái lập 100%.
  - Chúng làm ĐỐI CHỨNG cho Ragas: hai cách đo cùng một đại lượng mà lệch nhau
    là một phát hiện đáng viết vào README, không phải chuyện bỏ qua.

Nhóm 1 — truy hồi (dựa trên nhãn `reference_chunk_ids`):
    context_recall_id, context_precision_id, full_recall, mrr
Nhóm 2 — hành vi từ chối (dựa trên câu từ chối cố định trong prompt):
    refused, refusal_correct
Nhóm 3 — trích dẫn:
    has_citation, citation_valid
Nhóm 4 — quỹ đạo:
    n_tool_calls, n_llm_calls, steps_over_min, hit_step_cap
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from rag.config import REFUSAL_SENTINEL

CITE_RE = re.compile(r"\[(\d{1,2})\]")
# So khớp câu từ chối: lấy 40 ký tự đầu để chịu được khác biệt dấu câu cuối câu.
REFUSAL_KEY = REFUSAL_SENTINEL[:40].lower()


def is_refusal(answer: str) -> bool:
    return REFUSAL_KEY in (answer or "").lower()


def score_one(item, run: dict[str, Any]) -> dict[str, Any]:
    """item: BenchmarkItem. run: một dòng trong file kết quả sinh câu trả lời."""
    got = list(dict.fromkeys(run.get("chunk_ids", [])))
    want = set(item.reference_chunk_ids)
    answer = run.get("answer", "") or ""

    inter = want & set(got)
    recall = len(inter) / len(want) if want else None
    precision = len(inter) / len(got) if got and want else None

    # MRR: nghịch đảo thứ hạng của chunk chuẩn ĐẦU TIÊN. Đo "chunk đúng nằm cao
    # hay thấp", thứ mà recall không phân biệt được.
    mrr = 0.0
    for rank, cid in enumerate(got, 1):
        if cid in want:
            mrr = 1.0 / rank
            break

    refused = is_refusal(answer)
    cited = [int(m) for m in CITE_RE.findall(answer)]

    return {
        "id": item.id,
        "type": item.type,
        "tags": item.tags,
        # -- truy hồi
        "context_recall_id": recall,
        "context_precision_id": precision,
        "full_recall": (len(inter) == len(want)) if want else None,
        "mrr": mrr if want else None,
        "n_chunks": len(got),
        "ctx_chars": sum(len(c) for c in run.get("contexts", [])),
        "missing_chunks": sorted(want - inter),
        # -- từ chối
        "should_refuse": item.should_refuse,
        "refused": refused,
        "refusal_correct": refused == item.should_refuse,
        # -- trích dẫn (chỉ tính khi có trả lời thật)
        "has_citation": bool(cited) if not refused else None,
        # citation_valid CHỈ có nghĩa với pipeline tĩnh.
        # Baseline: một lần truy hồi -> [1..n] ánh xạ 1-1 với `got`, kiểm được thật.
        # Agent: mỗi lần gọi tool lại đánh số [1] từ đầu, trong khi `got` là hợp đã
        # khử trùng của MỌI lần gọi -> "1 <= c <= len(got)" gần như luôn đúng và
        # không chứng minh điều gì. Đặt None cho agent thay vì in ra 100% giả.
        "citation_valid": (
            None if run.get("n_tool_calls", 1) > 1
            else (all(1 <= c <= len(got) for c in cited) if cited else False)
        ) if not refused else None,
        # -- quỹ đạo
        "n_tool_calls": run.get("n_tool_calls", 0),
        "n_llm_calls": run.get("n_llm_calls", 0),
        "min_steps": item.min_steps,
        "steps_over_min": run.get("n_tool_calls", 0) - item.min_steps,
        "hit_step_cap": bool(run.get("hit_step_cap", False)),
        "error": run.get("error", ""),
        "answer_chars": len(answer),
    }


# --------------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Gộp điểm, và LUÔN tách theo loại câu hỏi.

    Trung bình trên cả 20 câu là con số dễ gây hiểu nhầm: câu ngoài phạm vi
    không có recall, câu một-bước và nhiều-bước hỏng theo cách khác nhau.
    Báo cáo gộp mà không tách loại là giấu thông tin.
    """
    out: dict[str, Any] = {}
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["type"]].append(r)
        for t in r["tags"]:
            groups[f"tag:{t}"].append(r)

    for name, g in groups.items():
        in_scope = [r for r in g if not r["should_refuse"]]
        out[name] = {
            "n": len(g),
            "context_recall_id": _mean([r["context_recall_id"] for r in in_scope]),
            "context_precision_id": _mean([r["context_precision_id"] for r in in_scope]),
            "full_recall_rate": _mean([r["full_recall"] for r in in_scope]),
            "mrr": _mean([r["mrr"] for r in in_scope]),
            "refusal_correct": _mean([r["refusal_correct"] for r in g]),
            "false_refusal_rate": _mean([r["refused"] for r in in_scope]),
            "has_citation": _mean([r["has_citation"] for r in g]),
            "citation_valid": _mean([r["citation_valid"] for r in g]),
            "avg_tool_calls": _mean([r["n_tool_calls"] for r in g]),
            "avg_llm_calls": _mean([r["n_llm_calls"] for r in g]),
            "avg_ctx_chars": _mean([r["ctx_chars"] for r in g]),
            "step_cap_rate": _mean([r["hit_step_cap"] for r in g]),
            "error_rate": _mean([bool(r["error"]) for r in g]),
        }
        # ĐỘ PHỦ cho những metric có thể là None ở một số câu. Không có phần này
        # thì hai cột "100.0%" cạnh nhau trông như nhau trong khi mẫu số lệch 40%
        # (đã gặp: citation_valid 15/20 ở baseline vs 9/20 ở agent, vì agent gọi
        # tool nhiều lần thì phép kiểm chỉ số trích dẫn không xác định).
        # Cùng nguyên tắc với `n_ok/n` ở tầng Ragas: một điểm trung bình không có
        # mẫu số đi kèm thì không phải là kết quả.
        out[name]["coverage"] = {
            "context_recall_id": sum(1 for r in in_scope if r["context_recall_id"] is not None),
            "context_precision_id": sum(1 for r in in_scope if r["context_precision_id"] is not None),
            "full_recall_rate": sum(1 for r in in_scope if r["full_recall"] is not None),
            "mrr": sum(1 for r in in_scope if r["mrr"] is not None),
            "has_citation": sum(1 for r in g if r["has_citation"] is not None),
            "citation_valid": sum(1 for r in g if r["citation_valid"] is not None),
        }
    return out


def fmt(v, pct: bool = True) -> str:
    if v is None:
        return "  —  "
    if isinstance(v, float) and pct and 0.0 <= v <= 1.0:
        return f"{v:6.1%}"
    if isinstance(v, float):
        return f"{v:6.2f}"
    return f"{v:>6}"


def print_summary(agg: dict[str, Any], title: str = "") -> None:
    if title:
        print(f"\n== {title} ==")
    # (nhãn, khoá, có phải tỉ lệ phần trăm không) — cột đếm KHÔNG được in ra %,
    # nếu không thì "trung bình 1.0 lần gọi tool" hiện thành "100.0%".
    cols = [
        ("recall", "context_recall_id", True),
        ("prec", "context_precision_id", True),
        ("đủ bộ", "full_recall_rate", True),
        ("mrr", "mrr", False),
        ("từ chối đúng", "refusal_correct", True),
        ("từ chối nhầm", "false_refusal_rate", True),
        ("trích dẫn", "has_citation", True),
        ("tool/câu", "avg_tool_calls", False),
    ]
    head = f"  {'nhóm':16s} {'n':>3s} " + " ".join(f"{c[0]:>13s}" for c in cols)
    print(head)
    print("  " + "-" * (len(head) - 2))

    def line(name: str) -> str:
        a = agg[name]
        return f"  {name:16s} {a['n']:>3} " + " ".join(
            f"{fmt(a[k], pct):>13s}" for _, k, pct in cols)

    for name in ["all", "single_hop", "multi_hop", "out_of_scope"]:
        if name in agg:
            print(line(name))
    for name in sorted(k for k in agg if k.startswith("tag:")):
        if agg[name]["n"] >= 2:
            print(line(name))
