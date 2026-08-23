"""Baseline RAG pipeline: truy hồi -> prompt -> Gemini sinh câu trả lời.

Đây là ĐƯỜNG ỐNG TĨNH dùng làm mốc so sánh cho agent ở Khối 5. Cố ý giữ y hệt
hình dạng của Project 4 (một lần truy hồi, một lần gọi LLM) để phép so sánh
"agent vs pipeline" đo đúng thứ cần đo.

Dùng chung với agent: `build_llm()` và `SYSTEM_PROMPT` — nếu hai bên dùng model
hoặc prompt khác nhau thì bảng so sánh vô nghĩa.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from rag.config import (
    GEMINI_CONTEXT_WINDOW,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    LLM_MAX_RETRIES,
    LLM_SLEEP_SECONDS,
    LLM_TEMPERATURE,
    PROJECT5_DIR,
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
)
from rag.retriever import RegulationRetriever, RetrievalConfig, format_blocks


# --------------------------------------------------------------------------
def load_api_key() -> str:
    """Tìm khóa Gemini: biến môi trường trước, rồi .env của project 5 / project 4.

    KHÔNG in khóa ra màn hình, không ghi vào log, không commit.
    """
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if os.getenv(var):
            return os.environ[var]

    for env_path in (PROJECT5_DIR / ".env", PROJECT5_DIR.parent / "project4_rag_assistant" / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                return v.strip().strip("'\"")

    raise SystemExit(
        "Không tìm thấy khóa Gemini.\n"
        "  Đặt biến môi trường GOOGLE_API_KEY, hoặc tạo file .env trong "
        f"{PROJECT5_DIR} với dòng:  GOOGLE_API_KEY=...\n"
        "  (.env đã nằm trong .gitignore)"
    )


def build_llm(model: str | None = None, temperature: float | None = None):
    """Tạo LLM Gemini.

    Truyền cả max_tokens lẫn context_window có chủ đích: nếu thiếu một trong hai,
    GoogleGenAI sẽ gọi mạng ngay trong hàm khởi tạo để hỏi metadata của model.
    """
    from llama_index.llms.google_genai import GoogleGenAI

    return GoogleGenAI(
        model=model or GEMINI_MODEL,
        api_key=load_api_key(),
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=GEMINI_MAX_TOKENS,
        context_window=GEMINI_CONTEXT_WINDOW,
    )


def complete_with_retry(llm, prompt: str, sleep: float = LLM_SLEEP_SECONDS) -> str:
    """Gọi LLM có backoff. Free tier hay trả 429/503 — thất bại tạm thời, không phải lỗi code."""
    delay = sleep
    last: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            out = llm.complete(prompt)
            time.sleep(sleep)  # chủ động giãn nhịp để không đụng rate limit
            return str(out).strip()
        except Exception as e:  # noqa: BLE001 - muốn bắt mọi lỗi tạm thời của API
            last = e
            msg = str(e).lower()
            transient = any(s in msg for s in ("429", "503", "500", "quota", "rate", "timeout", "overload"))
            if not transient or attempt == LLM_MAX_RETRIES - 1:
                break
            print(f"    [retry {attempt + 1}/{LLM_MAX_RETRIES}] {type(e).__name__}: {str(e)[:90]}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"LLM thất bại sau {LLM_MAX_RETRIES} lần: {last}")


# --------------------------------------------------------------------------
@dataclass
class RagAnswer:
    question: str
    answer: str
    contexts: list[str]                     # văn bản THÔ của chunk — dùng cho metric so nhãn
    chunk_ids: list[str]
    origins: list[str]
    scores: list[float]
    n_llm_calls: int
    n_tool_calls: int                       # pipeline tĩnh: luôn = 1 lần truy hồi
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    # ĐÚNG những khối văn bản đã đưa vào prompt: tiêu đề trích dẫn + dòng LƯU Ý + text.
    # Tách khỏi `contexts` vì hai thứ này phục vụ hai loại metric khác nhau:
    #   contexts       -> so với nhãn (non_llm_context_recall) -> phải là văn bản thô
    #   contexts_shown -> chấm faithfulness -> phải đúng thứ model đã nhìn thấy
    # Trộn hai vai này vào một trường là lỗi đã đo được (notes/03 §3.6 f).
    contexts_shown: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return REFUSAL_SENTINEL[:40].lower() in self.answer.lower()


class BaselineRAG:
    """Một lần truy hồi + một lần gọi LLM. Không vòng lặp, không quyết định."""

    def __init__(self, retriever: RegulationRetriever | None = None, llm=None,
                 config: RetrievalConfig | None = None, sleep: float = LLM_SLEEP_SECONDS,
                 amendment_notice: bool = True):
        self.retriever = retriever or RegulationRetriever()
        self.llm = llm or build_llm()
        self.config = config or RetrievalConfig()
        self.sleep = sleep
        # Công tắc ablation cho dòng "LƯU Ý: đã bị sửa bởi..." — xem format_context.
        self.amendment_notice = amendment_notice

    def build_prompt(self, question: str, context: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== CÁC ĐOẠN TRÍCH ===\n{context}\n\n"
            f"=== CÂU HỎI ===\n{question}\n\n"
            f"=== TRẢ LỜI ==="
        )

    def answer(self, question: str) -> RagAnswer:
        hits = self.retriever.search(question, self.config)
        blocks = format_blocks(hits, amendment_notice=self.amendment_notice)
        context = "\n\n".join(blocks)
        prompt = self.build_prompt(question, context)
        try:
            text = complete_with_retry(self.llm, prompt, sleep=self.sleep)
            err = ""
        except Exception as e:  # noqa: BLE001
            text, err = "", str(e)
        return RagAnswer(
            question=question,
            answer=text,
            contexts=[c.text for c in hits],
            chunk_ids=[c.chunk_id for c in hits],
            origins=[c.origin for c in hits],
            scores=[c.score for c in hits],
            n_llm_calls=1,
            n_tool_calls=1,
            trajectory=[{"type": "retrieve", "query": question,
                         "n_chunks": len(hits), "config": self.config.name}],
            error=err,
            contexts_shown=blocks,
        )


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Hồ sơ tự công bố sản phẩm gồm những gì?"
    rag = BaselineRAG()
    r = rag.answer(q)
    print(f"\n=== {q} ===\n{r.answer}\n")
    print(f"chunks: {r.chunk_ids}")
    print(f"từ chối: {r.refused}")
