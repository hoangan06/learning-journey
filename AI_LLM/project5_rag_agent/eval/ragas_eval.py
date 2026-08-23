"""Khối 3 — Chấm điểm bằng Ragas trên một file kết quả ĐÃ SINH SẴN.

Không gọi lại LLM để sinh câu trả lời: đọc results/runs/<tag>.jsonl. Nhờ vậy
chấm lại bao nhiêu lần cũng được mà không đốt thêm quota sinh.

CÁC QUYẾT ĐỊNH ĐÃ KIỂM CHỨNG TRÊN ragas 0.4.3 (đừng chép tutorial 0.1/0.2):

1. ragas 0.4 có HAI hệ metric không trộn được:
     - `ragas.metrics`             (legacy)      -> dùng với `evaluate()`
     - `ragas.metrics.collections` (mới)         -> chỉ dùng `await m.ascore()`
   Đưa metric "collections" vào `evaluate()` sẽ báo `TypeError: All metrics must
   be initialised metric objects` dù đã khởi tạo. Ở đây chọn LEGACY + `evaluate()`.

2. Judge = Gemini qua ENDPOINT TƯƠNG THÍCH OPENAI của Google.
   Lý do: `llm_factory(provider="google", client=genai.Client(...))` cho ra client
   ĐỒNG BỘ, và `agenerate()` sẽ ném `TypeError: Cannot use agenerate() with a
   synchronous client`. Đi qua `AsyncOpenAI` thì client bất đồng bộ, chạy được.

3. `llm_factory()` BẮT BUỘC có tham số `client`. Mọi ví dụ kiểu
   `llm_factory("gpt-4o")` là của bản cũ và sẽ lỗi.

4. Embedding cho `answer_relevancy`: dùng LẠI e5 của retriever thay vì gọi thêm
   API embedding. Vừa đỡ một phụ thuộc, vừa nhất quán với tầng truy hồi.
   Metric legacy gọi `embed_query`/`embed_documents` nên phải bọc đúng giao diện đó.

Chạy:
    python -m eval.ragas_eval --tag baseline
    python -m eval.ragas_eval --tag baseline --limit 5      # tiết kiệm quota
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any

from eval.benchmark import load_benchmark
from eval.generate import load_run_deduped, run_path
from rag.config import CHUNKS_JSON, GEMINI_MODEL, RESULTS_DIR

RAGAS_CACHE_DIR = RESULTS_DIR / ".ragas_cache"

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


# --------------------------------------------------------------------------
def build_e5_embeddings():
    """Bọc SentenceTransformer e5 theo giao diện embedding LEGACY của ragas.

    Metric `answer_relevancy` gọi `self.embeddings.embed_query(...)`. Embedding
    kiểu MỚI của ragas chỉ có `embed_text` -> đưa vào sẽ lỗi AttributeError.
    Nên phải kế thừa `BaseRagasEmbeddings` và cài đủ 4 phương thức trừu tượng.
    """
    from ragas.embeddings.base import BaseRagasEmbeddings

    from rag.config import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX, EMBED_MODEL_NAME

    class E5RagasEmbeddings(BaseRagasEmbeddings):
        # CẢNH BÁO ĐÃ TRẢ GIÁ: KHÔNG được đặt tên thuộc tính là `self.model` cho
        # đối tượng SentenceTransformer. Ragas đọc `getattr(embeddings, "model")`
        # để ghi telemetry (EmbeddingUsageEvent) và trường đó phải là CHUỖI.
        # Đặt nhầm -> mọi lần embed đều ném:
        #   ValidationError: model - Input should be a valid string
        #     [input_value=SentenceTransformer(...)]
        # và metric `answer_relevancy` chết im lặng, trả về NaN.
        def __init__(self) -> None:
            from sentence_transformers import SentenceTransformer

            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            self.model = EMBED_MODEL_NAME          # chuỗi — ragas đọc trường này
            self._st = SentenceTransformer(EMBED_MODEL_NAME, device=device)

        def embed_query(self, text: str) -> list[float]:
            v = self._st.encode([E5_QUERY_PREFIX + text], normalize_embeddings=True)[0]
            return v.astype(float).tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vs = self._st.encode([E5_PASSAGE_PREFIX + t for t in texts],
                                 normalize_embeddings=True)
            return [v.astype(float).tolist() for v in vs]

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    return E5RagasEmbeddings()


def build_judge(model: str, api_key: str, use_cache: bool = True,
                max_tokens: int = 0, temperature: float = 0.0):
    """Judge = Gemini qua cổng tương thích OpenAI, CÓ CACHE ĐĨA.

    Cache là thứ quan trọng nhất ở đây, không phải retry. Free tier giới hạn theo
    SỐ REQUEST MỖI NGÀY (đã đo: 20/ngày với gemini-3.6-flash). Hết quota giữa
    chừng thì retry vô nghĩa — phải chạy lại vào hôm sau. Có cache thì lần chạy
    sau KHÔNG gọi lại những cặp (prompt, model) đã chấm xong, nên tiến độ cộng dồn
    qua nhiều ngày thay vì mất trắng.

    `max_tokens`: ĐÃ TRẢ GIÁ MỘT LẦN. Chạy thật với gemini-3.5-flash-lite, 3/20 mẫu
    faithfulness chết với `IncompleteOutputException: The output is incomplete due to
    a max_tokens length limit`. Nguyên nhân: faithfulness bắt judge tách câu trả lời
    thành danh sách mệnh đề dưới dạng JSON; câu trả lời dài (mh01, mh07, sh07 liệt kê
    5-9 khoản) sinh JSON dài hơn mặc định của model. Instructor không parse được →
    mẫu đó thành NaN. Đây KHÔNG phải lỗi quota, và retry không cứu được vì lần nào
    cũng bị cắt ở đúng chỗ đó. Cách sửa là nới trần token của judge.
    `llm_factory(**kwargs)` đẩy thẳng kwargs xuống client (đã kiểm bằng inspect).
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    cache = None
    if use_cache:
        from ragas.cache import DiskCacheBackend

        RAGAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = DiskCacheBackend(cache_dir=str(RAGAS_CACHE_DIR))

    client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE, max_retries=5)
    kwargs: dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return llm_factory(model, provider="openai", client=client, cache=cache, **kwargs)


# --------------------------------------------------------------------------
def build_dataset(tag: str, limit: int = 0, include_oos: bool = False,
                  context_source: str = "raw", drop_refused: bool = False):
    """Ghép file kết quả với nhãn benchmark thành EvaluationDataset.

    MẶC ĐỊNH LOẠI 4 CÂU NGOÀI PHẠM VI — đây là một lỗi đã đo được, không phải sở thích.

    Câu ngoài phạm vi không có `reference_chunk_ids` (đúng theo thiết kế: không đoạn
    nào trong kho trả lời được nó). Bản trước nhét chuỗi giữ chỗ "(ngoài phạm vi)" vào
    `reference_contexts`; rapidfuzz so chuỗi đó với context thật ra ~0, nên mỗi câu
    ngoài phạm vi bị chấm 0 điểm context recall như thể retrieval trượt.

    Số bị bóp méo đúng bằng:
        full-recall trong phạm vi = 15/16 = 0.9375
        điểm báo cáo               = 0.9375 * 16/20 = 0.75
    Tức 0.75 KHÔNG phải chất lượng truy hồi; nó là chất lượng truy hồi nhân với tỉ lệ
    câu trong phạm vi. Metric đo cả thứ nó không được phép đo.

    Câu ngoài phạm vi vẫn được chấm — nhưng bằng metric tất định đúng loại
    (refusal accuracy trong `eval/metrics_local.py`, hiện 4/4), chứ không bằng
    context recall. Chấm faithfulness cho một câu từ chối cũng vô nghĩa tương tự:
    không có mệnh đề nào để đối chiếu.

    `context_source`:
      "raw"   -> `contexts`       = văn bản THÔ của chunk. Đúng cho các metric so
                                    với nhãn (`non_llm_context_recall`), vì nhãn
                                    cũng là văn bản thô.
      "shown" -> `contexts_shown` = ĐÚNG các khối đã đưa vào prompt (tiêu đề trích
                                    dẫn + dòng `LƯU Ý: {sua_doi}` + text). Đúng cho
                                    `faithfulness`, vì faithfulness hỏi "câu trả lời
                                    có căn cứ trong thứ model được đưa không".
    Dùng nhầm nguồn là một lỗi đã đo được: `mh03` bị chấm faithfulness 0,5 vì câu
    trả lời trích đúng dòng LƯU Ý, mà dòng đó KHÔNG có trong `contexts`. Hệ thống
    đúng, harness chấm sai. Xem notes/03 §3.6 (f).

    `drop_refused`: bỏ các câu mà hệ thống TỪ CHỐI trả lời. Câu từ chối không phát
    biểu mệnh đề nào, nên faithfulness của nó không xác định — ragas 0.4.3 trả về
    **0.0**, tức phạt như thể bịa đặt. `mh02` bị đúng chuyện này. Chỉ nên bật khi
    chạy các metric LLM về câu trả lời; các metric context không cần.
    """
    from ragas import EvaluationDataset

    p = run_path(tag)
    if not p.exists():
        raise SystemExit(f"Chưa có {p}. Chạy trước: python -m eval.generate --mode {tag}")

    bench = {i.id: i for i in load_benchmark()}
    chunk_text = {c["chunk_id"]: c["text_for_embedding"]
                  for c in json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))}

    from rag.config import REFUSAL_SENTINEL

    rows, ids, skipped_oos, skipped_refused = [], [], [], []
    missing_shown = 0
    # Khử trùng bằng ĐÚNG hàm mà `run_eval` dùng. Trước đây file này duyệt thẳng
    # từng dòng; hiện an toàn chỉ vì các dòng trùng đều có `error` và bị bỏ ở
    # dưới — tức là may, không phải thiết kế.
    deduped = load_run_deduped(tag)
    for r in deduped.values():
        it = bench.get(r["id"])
        if it is None or r.get("error"):
            continue
        if not it.reference_chunk_ids and not include_oos:
            skipped_oos.append(r["id"])
            continue
        if drop_refused and REFUSAL_SENTINEL[:40].lower() in (r["answer"] or "").lower():
            skipped_refused.append(r["id"])
            continue

        ctx = r["contexts"]
        if context_source == "shown":
            shown = r.get("contexts_shown")
            if shown:
                ctx = shown
            else:
                # Bản ghi cũ chạy trước khi có trường này. KHÔNG im lặng dùng tạm:
                # số sẽ không so được với bản ghi mới, và không có gì báo.
                missing_shown += 1

        # Ragas cần retrieved_contexts không rỗng; câu bị từ chối vẫn có context.
        rows.append({
            "user_input": it.user_input,
            "retrieved_contexts": ctx or ["(không có đoạn nào)"],
            "reference_contexts": [chunk_text[c] for c in it.reference_chunk_ids
                                   if c in chunk_text] or ["(ngoài phạm vi)"],
            "response": r["answer"] or "(không có câu trả lời)",
            "reference": it.reference,
        })
        ids.append(r["id"])
        if limit and len(rows) >= limit:
            break
    if missing_shown:
        raise SystemExit(
            f"  DỪNG: {missing_shown}/{len(rows) + missing_shown} bản ghi trong '{tag}' "
            "chưa có trường `contexts_shown`.\n"
            "  Bản ghi được sinh trước khi trường này tồn tại. Dùng tạm `contexts` sẽ tạo "
            "ra một con số KHÔNG so được với các lần chạy khác mà không có gì cảnh báo.\n"
            f"  Cách sửa: chạy lại `python -m eval.generate --mode {tag} --force`, "
            "hoặc dùng `--context-source raw`."
        )
    return EvaluationDataset.from_list(rows), ids, skipped_oos, skipped_refused


def build_metrics(names: list[str], judge) -> list:
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NonLLMContextPrecisionWithReference,
        NonLLMContextRecall,
        ResponseRelevancy,
    )

    # Gán judge ngay lúc khởi tạo: `evaluate()` chỉ bơm llm vào metric nào có
    # .llm is None, rồi RESET về None sau khi chạy xong.
    table = {
        "faithfulness": lambda: Faithfulness(llm=judge),
        "answer_relevancy": lambda: ResponseRelevancy(llm=judge),
        "context_precision": lambda: LLMContextPrecisionWithReference(llm=judge),
        "context_recall": lambda: LLMContextRecall(llm=judge),
        # Hai metric dưới KHÔNG dùng LLM: so chuỗi mờ bằng rapidfuzz giữa
        # retrieved_contexts và reference_contexts -> đối chứng cho bản LLM.
        "non_llm_context_recall": NonLLMContextRecall,
        "non_llm_context_precision": NonLLMContextPrecisionWithReference,
    }
    out = []
    for n in names:
        if n not in table:
            raise SystemExit(f"Metric '{n}' không có. Chọn trong: {', '.join(table)}")
        out.append(table[n]())
    return out



# --------------------------------------------------------------------------
# Số request LLM mỗi mẫu, theo từng metric (ước lượng từ cách ragas 0.4.3 hoạt động)
COST_PER_SAMPLE = {
    "faithfulness": 2,              # tách mệnh đề + phán đoán NLI
    "answer_relevancy": 1,          # sinh câu hỏi ngược từ câu trả lời
    "context_precision": None,      # 1 request MỖI ĐOẠN context -> đắt nhất
    "context_recall": 1,
    "non_llm_context_recall": 0,    # rapidfuzz, không gọi LLM
    "non_llm_context_precision": 0,
}


def estimate_requests(dataset, names: list[str]) -> tuple[int, dict[str, int]]:
    """Ước lượng số request TRƯỚC khi chạy.

    Có hàm này vì một lý do đã trả giá: free tier giới hạn theo NGÀY (20 request
    với gemini-3.6-flash). Chạy mù thì đốt hết quota trong 60 giây rồi nhận về
    một file kết quả toàn NaN trông như đã đo.
    """
    per_metric: dict[str, int] = {}
    for n in names:
        c = COST_PER_SAMPLE.get(n, 1)
        if c is None:  # context_precision: theo số đoạn context
            per_metric[n] = sum(len(sm.retrieved_contexts or []) for sm in dataset.samples)
        else:
            per_metric[n] = c * len(dataset)
    return sum(per_metric.values()), per_metric


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=GEMINI_MODEL)
    # Mặc định là bộ RẺ. `context_precision` tốn 1 request MỖI ĐOẠN context
    # (~6-7 lần/mẫu) nên không để mặc định — bật bằng --metrics full.
    ap.add_argument("--metrics", default="faithfulness,non_llm_context_recall",
                    help="danh sách, hoặc 'full' để lấy đủ 5 metric")
    ap.add_argument("--estimate-only", action="store_true",
                    help="chỉ in ước lượng số request rồi thoát, không gọi LLM")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="trần token của judge; đặt 0 để dùng mặc định của model. "
                         "Mặc định 4096 vì 3/20 mẫu faithfulness từng chết vì "
                         "IncompleteOutputException khi câu trả lời có nhiều mệnh đề")
    ap.add_argument("--context-source", choices=["raw", "shown"], default="raw",
                    help="'raw' = văn bản thô của chunk (đúng cho non_llm_context_recall). "
                         "'shown' = đúng khối đã đưa vào prompt, kèm dòng LƯU Ý "
                         "(đúng cho faithfulness). Xem docstring build_dataset")
    ap.add_argument("--drop-refused", action="store_true",
                    help="bỏ các câu hệ thống TỪ CHỐI. Câu từ chối không có mệnh đề nào "
                         "để đối chiếu nên ragas chấm faithfulness = 0.0, tức phạt như "
                         "thể bịa. Dùng khi chạy metric về câu trả lời")
    ap.add_argument("--include-oos", action="store_true",
                    help="giữ cả câu ngoài phạm vi trong dataset Ragas. MẶC ĐỊNH LÀ "
                         "LOẠI: chúng không có reference_contexts thật nên bị chấm 0 "
                         "và kéo tụt context recall (xem docstring build_dataset)")
    ap.add_argument("--budget", type=int, default=0,
                    help="số request tối đa cho phép; vượt thì dừng trước khi chạy")
    ap.add_argument("--workers", type=int, default=1,
                    help="1 = tuần tự, an toàn với free tier")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        import ragas  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Không import được ragas: {type(e).__name__}: {e}\n"
            "Chạy `python -m eval.doctor` để xem cách sửa."
        )

    from ragas import RunConfig, evaluate

    from rag.pipeline import load_api_key

    api_key = load_api_key()
    os.environ.setdefault("OPENAI_API_KEY", api_key)  # chặn ragas tự tìm khóa OpenAI thật

    dataset, ids, skipped_oos, skipped_refused = build_dataset(
        args.tag, args.limit, include_oos=args.include_oos,
        context_source=args.context_source, drop_refused=args.drop_refused)
    if args.metrics.strip() == "full":
        names = ["faithfulness", "answer_relevancy", "context_precision",
                 "context_recall", "non_llm_context_recall"]
    else:
        names = [s.strip() for s in args.metrics.split(",") if s.strip()]
    judge = build_judge(args.model, api_key, use_cache=not args.no_cache,
                        max_tokens=args.max_tokens)
    metrics = build_metrics(names, judge)

    total, per_metric = estimate_requests(dataset, names)
    print(f"== Ragas | tag={args.tag} | {len(dataset)} mẫu | judge={args.model} ==")
    print(f"  metrics: {', '.join(names)}")
    print(f"  cache  : {'TẮT' if args.no_cache else RAGAS_CACHE_DIR}")
    print(f"  judge max_tokens: {args.max_tokens or 'mặc định của model'}")
    print(f"  nguồn context   : {args.context_source}"
          + ("  (đúng thứ model đã nhìn thấy, có dòng LƯU Ý)" if args.context_source == "shown"
             else "  (văn bản thô của chunk — KHÔNG có dòng LƯU Ý)"))
    if skipped_refused:
        print(f"  ĐÃ LOẠI {len(skipped_refused)} câu bị TỪ CHỐI: {', '.join(skipped_refused)}")
        print("     (không phát biểu mệnh đề nào -> faithfulness bị chấm 0.0 oan)")
    if skipped_oos:
        print(f"  ĐÃ LOẠI {len(skipped_oos)} câu ngoài phạm vi: {', '.join(skipped_oos)}")
        print("     (không có reference_contexts thật -> context recall sẽ chấm 0 oan.")
        print("      Chúng được chấm bằng refusal accuracy tất định ở eval.run_eval.)")
    print(f"  ƯỚC LƯỢNG SỐ REQUEST: ~{total}")
    for n, c in per_metric.items():
        print(f"     {n:28s} ~{c}")
    print("  Hạn mức thật của tài khoản: https://aistudio.google.com/rate-limit")
    print("  LƯU Ý: đây là LLM-as-judge — điểm có phương sai, cần đối chiếu tay.\n")

    if args.budget and total > args.budget:
        raise SystemExit(f"  DỪNG: ước lượng {total} request > ngân sách {args.budget}. "
                         "Giảm --limit hoặc bớt --metrics.")
    if args.estimate_only:
        return

    # Nạp e5 SAU chốt chặn: `--estimate-only` và `--budget` phải rẻ tuyệt đối,
    # không tải 1GB trọng số chỉ để in một con số ước lượng.
    embeddings = build_e5_embeddings() if "answer_relevancy" in names else None

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=args.workers, timeout=300,
                             max_retries=5, max_wait=90),
        show_progress=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_sample: list[dict[str, Any]] = []
    for i, sc in enumerate(result.scores):
        row = {"id": ids[i] if i < len(ids) else f"row{i}"}
        row.update({k: (None if v != v else round(float(v), 4)) for k, v in sc.items()})
        per_sample.append(row)

    # ĐỘ PHỦ — phần quan trọng nhất của báo cáo này.
    # `evaluate()` biến mọi mẫu lỗi (429, parse hỏng, embedding lỗi) thành NaN và
    # vẫn trả về kết quả. Lấy trung bình trên phần còn lại rồi in ra như một con
    # số bình thường là BÁO CÁO SAI: "faithfulness = 1.0" trên 1/5 mẫu trông y hệt
    # "faithfulness = 1.0" trên 5/5 mẫu. Nên mọi điểm ở đây đều đi kèm n_ok/n.
    keys = sorted({k for r in per_sample for k in r if k != "id"})
    n = len(per_sample)
    agg, coverage = {}, {}
    for k in keys:
        vals = [r[k] for r in per_sample if r.get(k) is not None]
        coverage[k] = len(vals)
        agg[k] = round(sum(vals) / len(vals), 4) if vals else None

    full = [k for k in keys if coverage[k] == n]
    partial = [k for k in keys if 0 < coverage[k] < n]
    empty = [k for k in keys if coverage[k] == 0]

    # GỘP với kết quả cũ thay vì ghi đè.
    # Lý do đã trả giá: mỗi metric cần một cấu hình khác nhau — `faithfulness` phải
    # chấm trên `contexts_shown` và bỏ câu từ chối, còn `non_llm_context_recall` phải
    # chấm trên văn bản thô và giữ đủ câu. Không chạy chung một lệnh được. Nếu file
    # bị ghi đè thì lệnh sau XOÁ MẤT kết quả của lệnh trước, và `eval.report` in ra
    # một bảng thiếu metric mà không có gì báo là đã thiếu.
    path = RESULTS_DIR / f"ragas_{args.tag}.json"
    prior: dict[str, Any] = {}
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - file hỏng thì coi như chưa có
            prior = {}

    merged_rows: dict[str, dict[str, Any]] = {r["id"]: dict(r)
                                              for r in prior.get("per_sample", [])}
    for r in per_sample:
        merged_rows.setdefault(r["id"], {"id": r["id"]}).update(r)

    # Mỗi metric mang theo ĐÚNG cấu hình đã sinh ra nó. Không có phần này thì
    # hai con số nằm cạnh nhau trong cùng một file trông như cùng một phép đo.
    metric_config = dict(prior.get("metric_config", {}))
    for name in names:
        metric_config[name] = {
            "judge_model": args.model,
            "n": n,
            "coverage": coverage.get(name),
            "context_source": args.context_source,
            "drop_refused": bool(args.drop_refused),
            "include_oos": bool(args.include_oos),
            "excluded_out_of_scope": skipped_oos,
            "excluded_refused": skipped_refused,
        }

    out = {"tag": args.tag,
           # Giữ hai khoá phẳng này cho `eval/report.py` đọc — đừng bỏ.
           "judge_model": args.model, "n": n,
           "last_run": {"judge_model": args.model, "metrics": names, "n": n,
                        "judge_max_tokens": args.max_tokens or None,
                        "context_source": args.context_source,
                        "drop_refused": bool(args.drop_refused),
                        "scope": "in_scope_only" if not args.include_oos else "all"},
           "metric_config": metric_config,
           "coverage": {**prior.get("coverage", {}), **coverage},
           "reliable": full, "partial": partial, "failed": empty,
           "aggregate": {**prior.get("aggregate", {}), **agg},
           "per_sample": list(merged_rows.values())}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    kept = [k for k in out["aggregate"] if k not in names]
    if kept:
        print(f"\n  (giữ lại metric đã đo trước đó: {', '.join(kept)})")

    print("\n== Điểm trung bình (kèm độ phủ) ==")
    for k in keys:
        c = coverage[k]
        flag = "" if c == n else ("  <-- KHÔNG ĐỦ DỮ LIỆU, ĐỪNG BÁO CÁO" if c else "  <-- HỎNG HOÀN TOÀN")
        val = f"{agg[k]}" if agg[k] is not None else "—"
        print(f"  {k:38s} {val:>8s}   ({c}/{n} mẫu){flag}")

    if partial or empty:
        print(f"\n  [!] {len(partial) + len(empty)}/{len(keys)} metric KHÔNG đủ độ phủ.")
        print("      Nguyên nhân thường gặp: hết quota theo ngày (429 RESOURCE_EXHAUSTED).")
        print("      Chạy lại đúng lệnh này vào ngày mai — cache đĩa giữ lại phần đã chấm,")
        print("      nên lần sau chỉ tốn quota cho phần còn thiếu.")
        print("      TUYỆT ĐỐI không đưa các con số phủ một phần vào README.")
    print(f"\n  -> {path}")
    print("  Đối chiếu tay vài câu trước khi tin: python -m eval.inspect_run --tag "
          f"{args.tag} --id mh03")


if __name__ == "__main__":
    main()
