"""Kiểm thử tầng metric TẤT ĐỊNH.

Vì sao chỉ test đúng phần này, và test được: `eval/metrics_local.py` là code
thuần — không I/O, không mạng, không LLM. Cho cùng một input thì luôn cho cùng
một output. Đó chính là lý do tầng này tồn tại (README §3), và cũng là lý do nó
là phần DUY NHẤT trong repo đáng viết unit test.

Ba lỗi trong README §4.7 đã có thể bị bắt bởi một test ở đây thay vì bị bắt bởi
một con số trông hợp lý: mục 4 (từ chối bị chấm 0.0), mục 5 (dòng trùng làm
phình mẫu số), mục 6 (phân loại bằng cách parse output thay vì bằng nhãn).

Chạy:  python -m pytest tests/ -q      (hoặc: python tests/test_metrics_local.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.benchmark import BenchmarkItem              # noqa: E402
from eval.metrics_local import aggregate, is_refusal, score_one  # noqa: E402
from rag.config import REFUSAL_SENTINEL              # noqa: E402


# ---------------------------------------------------------------------------
def item(**kw) -> BenchmarkItem:
    """BenchmarkItem tối thiểu, chỉ ghi đè trường cần cho từng test."""
    base = dict(
        id="t1", type="single_hop", user_input="hỏi gì đó", reference="đáp án",
        reference_chunk_ids=["A", "B"], reference_docs=["43/2017/NĐ-CP"],
        min_steps=1, should_refuse=False, tags=[], verified_by_human=True, note="",
    )
    base.update(kw)
    return BenchmarkItem(**base)


def run(**kw) -> dict:
    """Bản ghi kết quả tối thiểu."""
    base = dict(id="t1", answer="Câu trả lời [1].", chunk_ids=["A", "B"],
                contexts=["x", "y"], n_tool_calls=1, n_llm_calls=1,
                trajectory=[], error="")
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# 1. Câu ngoài phạm vi: recall/precision/MRR phải là None, KHÔNG phải 0.0
#    Đây là ranh giới quan trọng nhất của file: không có gold chunk thì các đại
#    lượng đó không xác định. Trả 0.0 sẽ kéo tụt điểm trung bình của cả benchmark
#    bằng những câu mà hệ thống làm ĐÚNG — đúng lỗi README §4.7 mục 2.
def test_out_of_scope_metrics_are_none_not_zero():
    s = score_one(item(type="out_of_scope", reference_chunk_ids=[], should_refuse=True),
                  run(answer=REFUSAL_SENTINEL, chunk_ids=["Z"]))
    assert s["context_recall_id"] is None
    assert s["context_precision_id"] is None
    assert s["mrr"] is None
    assert s["refusal_correct"] is True


# 2. MRR: gold chunk đầu tiên ở hạng 3 -> 1/3
def test_mrr_uses_rank_of_first_gold_chunk():
    s = score_one(item(reference_chunk_ids=["C"]),
                  run(chunk_ids=["A", "B", "C", "D"]))
    assert abs(s["mrr"] - 1 / 3) < 1e-9


def test_mrr_is_zero_when_no_gold_chunk_retrieved():
    s = score_one(item(reference_chunk_ids=["C"]), run(chunk_ids=["A", "B"]))
    assert s["mrr"] == 0.0
    assert s["context_recall_id"] == 0.0


# 3. Recall vs full-recall: lấy được 1/2 chunk chuẩn thì recall 0.5 nhưng
#    full_recall phải False — hai đại lượng này KHÔNG được lẫn nhau.
def test_partial_recall_is_not_full_recall():
    s = score_one(item(reference_chunk_ids=["A", "B"]), run(chunk_ids=["A", "X"]))
    assert s["context_recall_id"] == 0.5
    assert s["full_recall"] is False


# 4. Nhận diện từ chối: khớp theo TIỀN TỐ nên chịu được khác biệt ở cuối câu.
#    README §3 từng ghi nhầm là "exact match" — test này cố định hành vi thật.
def test_refusal_detection_tolerates_trailing_differences():
    assert is_refusal(REFUSAL_SENTINEL)
    assert is_refusal(REFUSAL_SENTINEL.rstrip(".") + "!")
    assert is_refusal("  " + REFUSAL_SENTINEL.upper() + "  ")
    assert not is_refusal("Theo Điều 8 Nghị định 43/2017 thì phải ghi nhãn phụ [1].")
    assert not is_refusal("")
    assert not is_refusal(None)


# 5. Từ chối SAI ở câu trong phạm vi phải bị tính là từ chối nhầm,
#    và các metric về trích dẫn phải là None (không có gì để trích dẫn).
def test_false_refusal_on_in_scope_question():
    s = score_one(item(should_refuse=False), run(answer=REFUSAL_SENTINEL))
    assert s["refused"] is True
    assert s["refusal_correct"] is False
    assert s["has_citation"] is None
    assert s["citation_valid"] is None


# 6. Trích dẫn ngoài phạm vi: model ghi [9] mà chỉ có 6 đoạn -> không hợp lệ.
def test_citation_index_out_of_range_is_invalid():
    s = score_one(item(), run(answer="Theo quy định [9].",
                              chunk_ids=list("ABCDEF"), n_tool_calls=1))
    assert s["has_citation"] is True
    assert s["citation_valid"] is False


def test_citation_index_in_range_is_valid():
    s = score_one(item(), run(answer="Theo quy định [2].",
                              chunk_ids=list("ABCDEF"), n_tool_calls=1))
    assert s["citation_valid"] is True


# 7. Nhiều lần gọi tool -> citation_valid KHÔNG xác định.
#    Mỗi lần gọi tool đánh số [1] lại từ đầu, còn `chunk_ids` là hợp đã khử
#    trùng của mọi lần gọi, nên phép kiểm "1 <= n <= len(got)" đúng một cách
#    rỗng nghĩa. Trả None thay vì in ra 100% giả.
def test_citation_validity_undefined_for_multi_call_answers():
    s = score_one(item(), run(answer="Theo quy định [1].",
                              chunk_ids=list("ABCDEF"), n_tool_calls=3))
    assert s["citation_valid"] is None
    assert s["has_citation"] is True


# 8. aggregate() phải bỏ qua None chứ không coi là 0, và phải tách nhóm
#    theo type — trộn câu ngoài phạm vi vào recall là lỗi §4.7 mục 2.
def test_aggregate_excludes_out_of_scope_from_recall():
    rows = [
        score_one(item(id="a", reference_chunk_ids=["A"]), run(id="a", chunk_ids=["A"])),
        score_one(item(id="b", reference_chunk_ids=["B"]), run(id="b", chunk_ids=["X"])),
        score_one(item(id="c", type="out_of_scope", reference_chunk_ids=[], should_refuse=True),
                  run(id="c", answer=REFUSAL_SENTINEL, chunk_ids=["X"])),
    ]
    agg = aggregate(rows)
    # 2 câu trong phạm vi: 1 trúng, 1 trượt -> 0.5. Câu ngoài phạm vi KHÔNG được
    # kéo xuống thành 1/3.
    assert abs(agg["all"]["context_recall_id"] - 0.5) < 1e-9
    assert agg["all"]["n"] == 3
    assert agg["all"]["refusal_correct"] == 1.0   # cả 3 câu đều xử lý đúng
    assert agg["out_of_scope"]["n"] == 1


# 9. Khử trùng bản ghi — bảo vệ trực tiếp cho README §4.7 mục 5.
#    Một câu chết vì 429 rồi chạy lại để lại HAI dòng. Chấm cả hai thì mẫu số
#    phình và mọi tỉ lệ bị kéo xuống bởi dòng chưa từng có câu trả lời.
def test_dedupe_prefers_successful_row_over_errored_one():
    import json
    import tempfile

    import eval.generate as G

    lines = [
        {"id": "a", "answer": "ok"},
        {"id": "b", "answer": "", "error": "429 Too Many Requests"},
        {"id": "b", "answer": "chay lai thanh cong"},
        {"id": "c", "answer": "cu"},
        {"id": "c", "answer": "moi"},
        {"id": "d", "answer": "", "error": "loi 1"},
        {"id": "d", "answer": "", "error": "loi 2"},
    ]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "tmp.jsonl"
        f.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines),
                     encoding="utf-8")
        original = G.run_path
        G.run_path = lambda tag: f
        try:
            out = G.load_run_deduped("tmp")
        finally:
            G.run_path = original

    assert len(out) == 4, f"7 dòng phải rút về 4 câu, nhận {len(out)}"
    assert out["b"]["answer"] == "chay lai thanh cong"   # dòng chạy được thắng dòng lỗi
    assert out["c"]["answer"] == "moi"                   # hai dòng tốt -> lấy dòng sau
    assert out["d"].get("error")                         # toàn lỗi -> vẫn báo là lỗi


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'HỎNG ' + str(fails) if fails else 'TẤT CẢ ĐỀU ĐẠT'}")
    sys.exit(1 if fails else 0)
