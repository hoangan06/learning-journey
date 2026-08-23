"""Khối 3/5 — Chấm điểm KHÔNG cần LLM trên các file kết quả đã sinh, và so sánh các chế độ.

Đây là điểm vào chính của harness. Chạy trong vài giây, không tốn quota:

    python -m eval.run_eval --tags baseline
    python -m eval.run_eval --tags baseline,agent          # sinh bảng so sánh
    python -m eval.run_eval --tags baseline,agent --md agent_vs_baseline

Điểm Ragas (LLM-as-judge) nằm ở `eval/ragas_eval.py`, tách riêng có chủ đích:
metric ở đây tất định, metric ở kia có phương sai. Trộn chung vào một lệnh sẽ
khiến người đọc tưởng mọi con số đều đáng tin như nhau.
"""

from __future__ import annotations

import argparse
import json

from eval.benchmark import load_benchmark
from eval.generate import load_run_deduped, run_path
from eval.metrics_local import aggregate, print_summary, score_one
from rag.config import RESULTS_DIR


def score_tag(tag: str) -> tuple[list[dict], dict]:
    p = run_path(tag)
    if not p.exists():
        raise SystemExit(f"Chưa có {p}. Chạy trước: python -m eval.generate --mode {tag}")
    bench = {i.id: i for i in load_benchmark()}

    # Khử trùng theo id — xem `eval/generate.py:load_run_deduped`. Dùng chung một
    # hàm với `ragas_eval` là có chủ đích: hai nơi chấm điểm mà khử trùng khác nhau
    # thì sớm muộn cũng lệch mẫu số, và đó đúng là lỗi ở README §4.7 mục 5.
    n_lines = sum(1 for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
    latest = {k: v for k, v in load_run_deduped(tag).items() if k in bench}
    if n_lines != len(latest):
        print(f"  [dedupe] {n_lines} dòng -> {len(latest)} câu duy nhất "
              f"({n_lines - len(latest)} dòng cũ/lỗi bị bỏ qua)")
    still_failing = [k for k, v in latest.items() if v.get("error")]
    if still_failing:
        print(f"  [!] {len(still_failing)} câu KHÔNG có câu trả lời: {', '.join(sorted(still_failing))}")
        print("      Chạy lại `python -m eval.generate` để bổ sung trước khi tin bảng này.")

    rows = [score_one(bench[i], r) for i, r in latest.items()]
    return rows, aggregate(rows)


# --------------------------------------------------------------------------
COMPARE_ROWS = [
    ("Context recall (labelled chunks)", "context_recall_id", True),
    ("Full-recall rate", "full_recall_rate", True),
    ("Context precision (labelled)", "context_precision_id", True),
    ("MRR", "mrr", False),
    ("Refusal accuracy (all 20)", "refusal_correct", True),
    ("False-refusal rate (in-scope)", "false_refusal_rate", True),
    ("Citation present", "has_citation", True),
    ("Citation indices valid", "citation_valid", True),
    ("Mean tool calls / question", "avg_tool_calls", False),
    ("Mean LLM calls / question", "avg_llm_calls", False),
    ("Mean context chars", "avg_ctx_chars", False),
    ("Hit step cap", "step_cap_rate", True),
    ("Generation errors", "error_rate", True),
]


def _cell(v, pct: bool) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    return f"{v:,.2f}" if v < 1000 else f"{v:,.0f}"


def _cell_with_n(agg: dict, key: str, pct: bool) -> str:
    """Ô của bảng so sánh, KÈM mẫu số khi metric không phủ hết nhóm.

    Không có phần này thì "100.0%" trên 9 câu và "100.0%" trên 15 câu trông giống
    hệt nhau — đã gặp thật ở `citation_valid` (baseline 15/20 vs agent 9/20).
    Đây là cùng một nguyên tắc với `n_ok/n` ở tầng Ragas, áp cho tầng tất định.
    """
    cell = _cell(agg.get(key), pct)
    cov = agg.get("coverage", {}).get(key)
    if cov is not None and cov != agg.get("n"):
        cell += f" ({cov}/{agg['n']})"
    return cell


def to_markdown(aggs: dict[str, dict], n_in_scope: int) -> str:
    tags = list(aggs)
    L = ["# Agent vs baseline RAG — evaluation harness results\n"]
    L.append("All metrics on this page are **deterministic**: they come from set comparison "
             "against hand-labelled gold chunks, from a prefix match on a fixed refusal "
             "sentence, and from counters recorded in the agent trajectory. No LLM judge is "
             "involved, so re-running produces identical numbers. LLM-judged metrics (Ragas) "
             "are reported separately.\n")
    L.append("> **`Citation indices valid` is only defined for single-retrieval answers.** "
             "A citation `[n]` can be checked against the retrieved set only when there was one "
             "retrieval to number against. When the agent calls a tool more than once each result "
             "block restarts at `[1]`, while the scored set is the deduplicated union of all "
             "calls, so the check would pass vacuously. Those questions are scored `None` and "
             "excluded from this row — meaning the row has a smaller denominator for the agent "
             "column than for the baseline, and the two are not directly comparable.\n")

    L.append("## Overall\n")
    L.append("| metric | " + " | ".join(f"`{t}`" for t in tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    for label, key, pct in COMPARE_ROWS:
        L.append(f"| {label} | "
                 + " | ".join(_cell_with_n(aggs[t]["all"], key, pct) for t in tags) + " |")

    for grp, title in [("single_hop", "Single-hop questions"),
                       ("multi_hop", "Multi-hop questions"),
                       ("out_of_scope", "Out-of-scope questions"),
                       ("tag:amended", "Questions on amended articles")]:
        if not all(grp in aggs[t] for t in tags):
            continue
        n = aggs[tags[0]][grp]["n"]
        L.append(f"\n## {title} (n={n})\n")
        L.append("| metric | " + " | ".join(f"`{t}`" for t in tags) + " |")
        L.append("|---|" + "---|" * len(tags))
        for label, key, pct in COMPARE_ROWS:
            vals = [aggs[t][grp][key] for t in tags]
            if all(v is None for v in vals):
                continue
            L.append(f"| {label} | "
                     + " | ".join(_cell_with_n(aggs[t][grp], key, pct) for t in tags) + " |")

    step = 1.0 / n_in_scope
    L.append(
        f"\n## How to read these numbers\n\n"
        f"The in-scope benchmark has **{n_in_scope} questions**, so one question is worth "
        f"**{step:.1%}**. Any gap smaller than that is a single question changing its mind, "
        f"not a real difference — and even a one-question gap is within the noise of a "
        f"non-deterministic agent. Differences are only worth discussing when they are "
        f"several questions wide, and the honest framing for anything smaller is "
        f"\"indistinguishable on this benchmark\".\n"
    )
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="baseline")
    ap.add_argument("--md", default="", help="tên file markdown so sánh trong results/")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    aggs, all_rows = {}, {}
    for t in tags:
        rows, agg = score_tag(t)
        aggs[t], all_rows[t] = agg, rows
        print_summary(agg, title=f"{t}  ({len(rows)} câu)")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"eval_{t}.json").write_text(
            json.dumps({"tag": t, "aggregate": agg, "per_sample": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")

        if args.show_misses:
            print(f"\n  -- câu thiếu chunk chuẩn ({t}) --")
            for r in rows:
                if r["missing_chunks"]:
                    print(f"    {r['id']:6s} thiếu {r['missing_chunks']}")
            bad = [r for r in rows if not r["refusal_correct"]]
            if bad:
                print(f"  -- từ chối sai ({t}) --")
                for r in bad:
                    print(f"    {r['id']:6s} should_refuse={r['should_refuse']} "
                          f"refused={r['refused']}")

    if len(tags) > 1 or args.md:
        n_in = sum(1 for r in all_rows[tags[0]] if not r["should_refuse"])
        md = to_markdown(aggs, n_in)
        name = args.md or ("_vs_".join(tags) if len(tags) > 1 else f"eval_{tags[0]}")
        path = RESULTS_DIR / f"{name}.md"
        path.write_text(md, encoding="utf-8")
        print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
