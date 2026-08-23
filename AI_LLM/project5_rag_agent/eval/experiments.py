"""Khối 4 — Thí nghiệm tinh chỉnh retrieval, ĐO BẰNG NHÃN, KHÔNG CẦN LLM.

Toàn bộ file này chạy offline trong vài chục giây và không tốn một request
Gemini nào. Đó là chủ ý: tách tầng truy hồi khỏi tầng sinh thì mới kiểm soát
được biến — thay đổi ở tầng sinh (prompt, nhiệt độ, model) không lẫn vào kết
quả, và chạy lại bao nhiêu lần cũng ra đúng con số cũ.

Các biến thể quét:
    top_k              3 / 5 / 10
    parent-child       bật / tắt
    amendment-aware    bật / tắt

Đầu ra:
    results/retrieval_experiments.md    bảng để dán vào README
    results/retrieval_experiments.json  số thô

Chạy:
    python -m eval.experiments
    python -m eval.experiments --top-k 3,5,10,20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from statistics import mean

from eval.benchmark import BenchmarkItem, load_benchmark
from rag.config import RESULTS_DIR
from rag.retriever import RegulationRetriever, RetrievalConfig


@dataclass
class VariantResult:
    name: str
    top_k: int
    parent_child: bool
    amendments: bool
    recall: float
    precision: float
    full_recall: float
    mrr: float
    ctx_chars: float
    n_chunks: float
    recall_single: float
    recall_multi: float
    recall_amended: float
    misses: list[str]


def evaluate_variant(r: RegulationRetriever, items: list[BenchmarkItem],
                     cfg: RetrievalConfig) -> VariantResult:
    rec, prec, full, mrrs, chars, counts = [], [], [], [], [], []
    by_type: dict[str, list[float]] = {"single_hop": [], "multi_hop": []}
    amended: list[float] = []
    misses: list[str] = []

    for it in items:
        hits = r.search(it.user_input, cfg)
        got = [c.chunk_id for c in hits]
        want = set(it.reference_chunk_ids)
        inter = want & set(got)

        v = len(inter) / len(want)
        rec.append(v)
        prec.append(len(inter) / len(got) if got else 0.0)
        full.append(float(len(inter) == len(want)))
        mrrs.append(next((1.0 / i for i, c in enumerate(got, 1) if c in want), 0.0))
        chars.append(sum(len(c.text) for c in hits))
        counts.append(len(hits))
        by_type.setdefault(it.type, []).append(v)
        if "amended" in it.tags:
            amended.append(v)
        if inter != want:
            misses.append(f"{it.id}:{','.join(sorted(want - inter))}")

    return VariantResult(
        name=cfg.name, top_k=cfg.top_k,
        parent_child=cfg.use_parent_child, amendments=cfg.expand_amendments,
        recall=mean(rec), precision=mean(prec), full_recall=mean(full),
        mrr=mean(mrrs), ctx_chars=mean(chars), n_chunks=mean(counts),
        recall_single=mean(by_type["single_hop"]) if by_type["single_hop"] else 0.0,
        recall_multi=mean(by_type["multi_hop"]) if by_type["multi_hop"] else 0.0,
        recall_amended=mean(amended) if amended else 0.0,
        misses=misses,
    )


def oos_analysis(r: RegulationRetriever, items: list[BenchmarkItem],
                 cfg: RetrievalConfig) -> dict:
    """Ngưỡng similarity có tách được câu ngoài phạm vi không?

    Đây là phép kiểm quyết định cơ chế từ chối: nếu KHÔNG tách được thì việc
    từ chối phải do LLM quyết định qua prompt, không được dùng ngưỡng số.
    """
    ins = [i for i in items if i.is_in_scope]
    oos = [i for i in items if not i.is_in_scope]
    s_in = {i.id: r.search(i.user_input, cfg)[0].score for i in ins}
    s_oos = {i.id: r.search(i.user_input, cfg)[0].score for i in oos}
    lo_in, hi_oos = min(s_in.values()), max(s_oos.values())
    return {
        "config": cfg.name,
        "in_scope_top1": s_in,
        "out_of_scope_top1": s_oos,
        "min_in_scope": lo_in,
        "max_out_of_scope": hi_oos,
        "separable": bool(lo_in > hi_oos),
        "gap": round(lo_in - hi_oos, 4),
    }


# --------------------------------------------------------------------------
def to_markdown(rows: list[VariantResult], oos: dict, n_in: int, n_all: int) -> str:
    L: list[str] = []
    L.append("# Retrieval tuning experiments\n")
    L.append(f"Benchmark: **{n_all} questions** ({n_in} in-scope, {n_all - n_in} out-of-scope). ")
    L.append("All numbers below are computed by **set comparison against hand-labelled "
             "`reference_chunk_ids`** — no LLM is involved, so they are fully deterministic "
             "and reproducible.\n")
    L.append("`recall` = fraction of labelled gold chunks retrieved. "
             "`precision` = fraction of retrieved chunks that are gold — see *Limitations*, "
             "this metric penalises useful-but-unlabelled chunks. "
             "`MRR` = mean reciprocal rank of the first gold chunk. "
             "`ctx chars` = mean characters of context handed to the LLM, i.e. **the cost "
             "of the recall**.\n")

    L.append("## Main table\n")
    L.append("| variant | top-k | parent-child | amendment | recall | full-recall | precision | MRR | chunks | ctx chars |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in rows:
        L.append(
            f"| `{v.name}` | {v.top_k} | {'yes' if v.parent_child else 'no'} "
            f"| {'yes' if v.amendments else 'no'} | {v.recall:.1%} | {v.full_recall:.1%} "
            f"| {v.precision:.1%} | {v.mrr:.3f} | {v.n_chunks:.1f} | {v.ctx_chars:,.0f} |"
        )

    L.append("\n## Recall broken down by question type\n")
    L.append("| variant | single-hop | multi-hop | amended-article questions |")
    L.append("|---|---|---|---|")
    for v in rows:
        L.append(f"| `{v.name}` | {v.recall_single:.1%} | {v.recall_multi:.1%} "
                 f"| {v.recall_amended:.1%} |")
    L.append("\nThe breakdown matters more than the headline number: a variant can lift the "
             "average while making one class of question worse.\n")

    L.append("## Out-of-scope separability\n")
    L.append(f"Config used: `{oos['config']}`\n")
    L.append("| question | type | top-1 similarity |")
    L.append("|---|---|---|")
    for k, v in sorted(oos["out_of_scope_top1"].items()):
        L.append(f"| {k} | out-of-scope | {v:.4f} |")
    L.append(f"| **lowest in-scope** | in-scope | **{oos['min_in_scope']:.4f}** |")
    L.append(
        f"\nLowest in-scope top-1 = **{oos['min_in_scope']:.4f}**, highest out-of-scope "
        f"top-1 = **{oos['max_out_of_scope']:.4f}** (gap = {oos['gap']:+.4f}). "
        + ("**The two classes overlap, so a similarity threshold cannot be used to decide "
           "when to refuse.** Refusal therefore has to be driven by the prompt and measured "
           "on the out-of-scope set, not by a score cutoff."
           if not oos["separable"] else
           "The classes are separable on this benchmark, but the margin is small and the "
           "sample is tiny — a threshold tuned on 4 out-of-scope questions would not "
           "generalise.")
    )

    L.append("\n## Questions still missing gold chunks\n")
    for v in rows:
        if v.misses:
            L.append(f"- `{v.name}`: {'; '.join(v.misses)}")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", default="3,5,10")
    ap.add_argument("--out", default="retrieval_experiments")
    args = ap.parse_args()

    items = load_benchmark()
    in_scope = [i for i in items if i.is_in_scope]
    r = RegulationRetriever()

    ks = [int(x) for x in args.top_k.split(",")]
    configs = [RetrievalConfig(top_k=k, use_parent_child=pc, expand_amendments=am)
               for k in ks for pc in (False, True) for am in (False, True)]

    print(f"== Thí nghiệm retrieval | {len(configs)} biến thể × {len(in_scope)} câu ==\n")
    rows = []
    for cfg in configs:
        v = evaluate_variant(r, in_scope, cfg)
        rows.append(v)
        print(f"  {v.name:26s} recall {v.recall:6.1%}  đủ bộ {v.full_recall:6.1%}  "
              f"prec {v.precision:6.1%}  mrr {v.mrr:.3f}  ctx {v.ctx_chars:8,.0f}")

    oos = oos_analysis(r, items, RetrievalConfig(top_k=5, use_parent_child=True))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, oos, len(in_scope), len(items))
    (RESULTS_DIR / f"{args.out}.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"{args.out}.json").write_text(
        json.dumps({"variants": [v.__dict__ for v in rows], "out_of_scope": oos},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n  Ngưỡng similarity tách được out-of-scope? "
          f"{'CÓ' if oos['separable'] else 'KHÔNG'} (gap {oos['gap']:+.4f})")
    print(f"  -> {RESULTS_DIR / (args.out + '.md')}")


if __name__ == "__main__":
    main()
