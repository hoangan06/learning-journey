"""Giai đoạn SINH câu trả lời — tách hẳn khỏi giai đoạn CHẤM ĐIỂM.

Vì sao tách:
  1. Sinh câu trả lời tốn quota Gemini. Chấm điểm lại nhiều lần (đổi metric,
     sửa công thức, chạy Ragas) thì KHÔNG được phép gọi lại LLM để sinh.
  2. Có file kết quả trên đĩa thì mọi con số đều truy vết được: mở ra đọc đúng
     câu trả lời nào đã sinh ra điểm nào.
  3. Chạy dở bị ngắt (429, mất mạng) thì lần sau chạy tiếp, không mất công.

Kết quả ghi vào results/runs/<tag>.jsonl — mỗi dòng một câu benchmark.

Chạy:
    python -m eval.generate --mode baseline
    python -m eval.generate --mode agent
    python -m eval.generate --mode baseline --limit 3      # thử 3 câu cho rẻ
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from eval.benchmark import load_benchmark
from rag.config import LLM_SLEEP_SECONDS, RUNS_DIR


def run_path(tag: str) -> Path:
    return RUNS_DIR / f"{tag}.jsonl"


def load_run_deduped(tag: str) -> dict[str, dict]:
    """Đọc file kết quả, trả về ĐÚNG MỘT dòng cho mỗi câu hỏi.

    Dùng chung cho mọi thứ chấm điểm — `run_eval` và `ragas_eval` — vì đây là
    lỗi đã trả giá thật (README §4.7 mục 5): file mở ở chế độ append và dòng có
    `error` được coi là chưa xong, nên một câu chết vì 429 rồi chạy lại để lại
    HAI dòng. Chấm cả hai thì mẫu số phình (27 dòng trên benchmark 20 câu) và
    mọi tỉ lệ bị kéo xuống bởi những dòng chưa từng có câu trả lời.

    Quy tắc: ưu tiên dòng CHẠY ĐƯỢC; trong các dòng chạy được thì lấy dòng mới
    nhất. Trả về dict giữ nguyên thứ tự xuất hiện lần đầu.
    """
    p = run_path(tag)
    if not p.exists():
        return {}
    latest: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prev = latest.get(r["id"])
        if prev is None:
            latest[r["id"]] = r          # lần đầu gặp id này
        elif not r.get("error"):
            latest[r["id"]] = r          # dòng chạy được luôn thắng dòng đang giữ
        # dòng lỗi mà đã có dòng trước: bỏ qua, giữ nguyên dòng cũ
    return latest


def load_existing(tag: str) -> dict[str, dict]:
    p = run_path(tag)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            # Dòng lỗi coi như chưa chạy -> lần sau tự thử lại
            if not r.get("error"):
                out[r["id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "agent", "agent_react"], default="baseline")
    ap.add_argument("--tag", default="", help="tên file kết quả (mặc định = mode)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-parent-child", action="store_true")
    ap.add_argument("--amendments", action="store_true")
    ap.add_argument("--no-amendment-notice", action="store_true",
                    help="ABLATION: bỏ dòng 'LƯU Ý: đã bị sửa bởi...' khỏi context. "
                         "Dùng kèm --only mh03 --tag mh03_noNotice để đo xem dòng đó "
                         "có thật sự là thứ giúp trả lời đúng câu bị bãi bỏ không")
    ap.add_argument("--max-iterations", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu")
    ap.add_argument("--only", default="", help="chỉ chạy các id, cách nhau bằng dấu phẩy")
    ap.add_argument("--sleep", type=float, default=LLM_SLEEP_SECONDS)
    ap.add_argument("--force", action="store_true", help="chạy lại cả câu đã có")
    args = ap.parse_args()

    tag = args.tag or args.mode
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    items = load_benchmark()
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        items = [i for i in items if i.id in keep]
    if args.limit:
        items = items[: args.limit]

    done = {} if args.force else load_existing(tag)
    todo = [i for i in items if i.id not in done]
    print(f"== Sinh câu trả lời | mode={args.mode} tag={tag} ==")
    print(f"  {len(items)} câu | đã có {len(items) - len(todo)} | cần chạy {len(todo)}")
    if not todo:
        print("  Không còn gì để chạy. Dùng --force để chạy lại.")
        return

    from rag.retriever import RegulationRetriever, RetrievalConfig

    retriever = RegulationRetriever()
    cfg = RetrievalConfig(
        top_k=args.top_k,
        use_parent_child=not args.no_parent_child,
        expand_amendments=args.amendments,
    )

    if args.mode == "baseline":
        from rag.pipeline import BaselineRAG

        engine = BaselineRAG(retriever=retriever, config=cfg, sleep=args.sleep,
                             amendment_notice=not args.no_amendment_notice)
    elif args.no_amendment_notice:
        raise SystemExit("--no-amendment-notice chỉ áp dụng cho --mode baseline.")
    else:
        from agent.app import RegulationAgent, _fix_windows_loop

        _fix_windows_loop()
        engine = RegulationAgent(
            retriever=retriever,
            top_k=args.top_k,
            use_parent_child=not args.no_parent_child,
            expand_amendments=args.amendments,
            max_iterations=args.max_iterations,
            use_react=(args.mode == "agent_react"),
        )

    t0 = time.time()
    # Mở ở chế độ append: bị ngắt giữa chừng vẫn giữ được phần đã chạy.
    # CHỈ ghi đè khi thật sự chạy lại TOÀN BỘ. `--force --only mh03` mà mở "w"
    # sẽ xoá trắng 19 câu còn lại để ghi đúng 1 dòng — mất cả lần chạy trước mà
    # không có gì cảnh báo.
    # `--force` đặt done = {}, nên vế `or not done` cũ vô hiệu hoá luôn `truncate`:
    # `--force --only mh03` vẫn mở "w" và xoá trắng 19 câu còn lại. Chỉ xét `truncate`.
    # Mở "a" trên file chưa tồn tại vẫn tạo file bình thường.
    truncate = args.force and not args.only and not args.limit
    with run_path(tag).open("w" if truncate else "a", encoding="utf-8") as f:
        for n, it in enumerate(todo, 1):
            t1 = time.time()
            ans = engine.answer(it.user_input)
            rec = asdict(ans)
            rec.update({
                "id": it.id,
                "type": it.type,
                "mode": args.mode,
                "retrieval_config": cfg.name,
                "hit_step_cap": "max_iterations" in (ans.error or ""),
                "latency_s": round(time.time() - t1, 2),
            })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            flag = "TỪ CHỐI" if ans.refused else f"{len(ans.answer)} ký tự"
            print(f"  [{n}/{len(todo)}] {it.id:6s} {it.type:12s} "
                  f"tool={ans.n_tool_calls} chunk={len(ans.chunk_ids)} "
                  f"{rec['latency_s']:5.1f}s {flag}"
                  + (f"  LỖI: {ans.error[:60]}" if ans.error else ""))

    print(f"\n  Xong sau {time.time() - t0:.0f}s -> {run_path(tag)}")
    print(f"  Chấm điểm: python -m eval.run_eval --tag {tag}")


if __name__ == "__main__":
    main()
