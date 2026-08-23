"""Chạy toàn bộ project theo đúng thứ tự, một lệnh.

    python run_all.py            # chạy tất cả (cần khóa Gemini cho giai đoạn 3-5)
    python run_all.py --no-llm   # chỉ chạy phần KHÔNG cần LLM (nhanh, không tốn quota)
    python run_all.py --from 3   # chạy tiếp từ giai đoạn 3

Thiết kế: mỗi giai đoạn ghi kết quả ra đĩa và giai đoạn sau đọc lại từ đĩa.
Bị ngắt ở giữa (hết quota, mất mạng) thì chạy lại chỉ làm tiếp phần còn thiếu,
không mất công đã bỏ ra.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = [
    # Giai đoạn 0 chạy TRƯỚC mọi thứ và không cần khoá API: nếu tầng metric tất
    # định đã sai thì mọi bảng phía sau đều vô nghĩa, nên hỏng ở đây phải dừng
    # sớm chứ không phải sau 20 phút gọi LLM.
    (0, "Kiểm thử tầng metric tất định", ["-m", "pytest", "tests/", "-q"], False),
    (1, "Dựng vector index (Chroma)", ["-m", "rag.build_index", "--force"], False),
    (1, "Kiểm chứng migration vs Project 4", ["-m", "rag.verify_migration"], False),
    (2, "Kiểm tra benchmark", ["-m", "eval.benchmark"], False),
    (2, "Trần recall của benchmark", ["-m", "eval.benchmark", "--retrieval-check"], False),
    (3, "Thí nghiệm retrieval (không cần LLM)", ["-m", "eval.experiments"], False),
    (4, "Sinh câu trả lời — baseline RAG", ["-m", "eval.generate", "--mode", "baseline"], True),
    (5, "Sinh câu trả lời — agent", ["-m", "eval.generate", "--mode", "agent"], True),
    (6, "Chấm điểm tất định + bảng so sánh",
     ["-m", "eval.run_eval", "--tags", "baseline,agent", "--md", "agent_vs_baseline",
      "--show-misses"], False),
    # Ragas chạy LÀM HAI LƯỢT cho mỗi tag, và đó không phải chuyện thẩm mỹ:
    # hai metric cần hai cấu hình loại trừ nhau.
    #   non_llm_context_recall -> so với nhãn, nên phải nhận VĂN BẢN THÔ của chunk,
    #                             và phải giữ đủ câu (kể cả câu bị từ chối).
    #   faithfulness           -> hỏi "câu trả lời có căn cứ trong thứ model ĐƯỢC ĐƯA không",
    #                             nên phải nhận đúng khối đã vào prompt (--context-source shown)
    #                             và phải BỎ câu từ chối (không có mệnh đề nào để chấm).
    # Chạy chung một lượt bằng cấu hình mặc định sẽ ra số khác với số trong README.
    (7, "Ragas — context recall (baseline, không tốn request)",
     ["-m", "eval.ragas_eval", "--tag", "baseline",
      "--metrics", "non_llm_context_recall"], True),
    (7, "Ragas — faithfulness (baseline)",
     ["-m", "eval.ragas_eval", "--tag", "baseline", "--metrics", "faithfulness",
      "--context-source", "shown", "--drop-refused", "--max-tokens", "4096"], True),
    (7, "Ragas — context recall (agent, không tốn request)",
     ["-m", "eval.ragas_eval", "--tag", "agent",
      "--metrics", "non_llm_context_recall"], True),
    (7, "Ragas — faithfulness (agent)",
     ["-m", "eval.ragas_eval", "--tag", "agent", "--metrics", "faithfulness",
      "--context-source", "shown", "--drop-refused", "--max-tokens", "4096"], True),
    (8, "Tổng hợp results/ -> results/SUMMARY.md", ["-m", "eval.report"], False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="bỏ mọi giai đoạn cần Gemini")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    print("== Project 5 — chạy toàn bộ ==")
    print(f"   {'BỎ QUA' if args.no_llm else 'BAO GỒM'} các giai đoạn cần Gemini\n")

    failed = []
    for stage, title, cmd, needs_llm in STAGES:
        if stage < args.start:
            continue
        if needs_llm and args.no_llm:
            print(f"-- [{stage}] {title}  (bỏ qua: cần LLM)")
            continue
        print(f"\n{'=' * 70}\n[{stage}] {title}\n{'=' * 70}")
        t0 = time.time()
        rc = subprocess.call([sys.executable, *cmd])
        if rc != 0 and cmd[:2] == ["-m", "pytest"]:
            # Môi trường không có pytest: file test có sẵn runner riêng, không có
            # lý do gì để cả pipeline hỏng chỉ vì thiếu một thư viện test.
            print("   (không chạy được pytest — dùng runner có sẵn trong file test)")
            rc = subprocess.call([sys.executable, "tests/test_metrics_local.py"])
        print(f"   ({time.time() - t0:.0f}s, mã thoát {rc})")
        if rc != 0:
            failed.append(title)
            if args.stop_on_error:
                break

    print("\n" + "=" * 70)
    if failed:
        print("CÁC BƯỚC LỖI:")
        for f in failed:
            print(f"  - {f}")
        print("\nChạy `python -m eval.doctor` để chẩn đoán.")
        sys.exit(1)
    print("Xong. Kết quả nằm trong results/.")


if __name__ == "__main__":
    main()
