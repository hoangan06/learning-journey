"""Khối 5 — Agent tool-calling bằng LlamaIndex, có ghi lại quỹ đạo.

Khác baseline ở đúng một chỗ: LLM quyết định gọi tool nào, mấy lần, với truy vấn
nào. Mọi thứ còn lại (model, nhiệt độ, retriever, quy tắc trả lời) giữ y hệt
`rag/pipeline.py` — nếu khác thì bảng so sánh không đo được thứ định đo.

Quỹ đạo được ghi qua `handler.stream_events()`. Đây là điểm mấu chốt của Khối 3:
không có trace thì chỉ đo được KẾT QUẢ, không đo được ĐƯỜNG ĐI, và một agent đi
6 bước lởm rồi may mắn ra đáp án đúng sẽ bị chấm là "tốt".

Chạy thử:
    python -m agent.app "Nhãn phụ hàng bị trả lại có phải ghi 'Được sản xuất tại Việt Nam' không?"
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rag.config import REFUSAL_SENTINEL, SYSTEM_PROMPT
from rag.pipeline import RagAnswer, build_llm
from rag.retriever import format_blocks
from rag.retriever import RegulationRetriever

DEFAULT_MAX_ITERATIONS = 6      # chốt chặn cứng; xem notes/00 §3.2
DEFAULT_TIMEOUT = 180.0         # LlamaIndex mặc định KHÔNG có timeout cho agent

AGENT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

BẠN CÓ CÔNG CỤ. Cách dùng:
- Gọi `search_regulations` để tra nội dung quy định. Câu hỏi cần nhiều văn bản
  hoặc nhiều Điều thì gọi NHIỀU LẦN, mỗi lần một truy vấn hẹp.
- Gọi `check_amendment` trước khi khẳng định nội dung bất kỳ Điều nào của
  Nghị định 43/2017/NĐ-CP.
- Khi đã đủ căn cứ thì dừng gọi tool và trả lời. Đừng gọi lại cùng một truy vấn.
- Đánh số trích dẫn theo thứ tự các đoạn bạn đã nhận được từ tool.
"""
)


class RegulationAgent:
    def __init__(self, retriever: RegulationRetriever | None = None, llm=None,
                 top_k: int = 5, use_parent_child: bool = True,
                 expand_amendments: bool = False,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 use_react: bool = False):
        from llama_index.core.agent.workflow import FunctionAgent, ReActAgent

        from agent.tools import build_tools

        self.retriever = retriever or RegulationRetriever()
        self.llm = llm or build_llm()
        self.max_iterations = max_iterations
        self.use_react = use_react
        self._collector: list = []

        tools = build_tools(
            self.retriever, self._collector,
            top_k=top_k, use_parent_child=use_parent_child,
            expand_amendments=expand_amendments,
        )
        cls = ReActAgent if use_react else FunctionAgent
        self.agent = cls(
            tools=tools,
            llm=self.llm,
            system_prompt=AGENT_SYSTEM_PROMPT,
            timeout=DEFAULT_TIMEOUT,
        )

    # ----------------------------------------------------------------------
    async def _arun(self, question: str) -> RagAnswer:
        from llama_index.core.agent.workflow import (
            AgentOutput,
            ToolCall,
            ToolCallResult,
        )
        from llama_index.core.workflow.errors import WorkflowRuntimeError

        self._collector.clear()
        trajectory: list[dict[str, Any]] = []
        n_tool_calls = 0
        n_llm_calls = 0
        hit_cap = False
        answer = ""
        error = ""

        # max_iterations là THAM SỐ CỦA .run(), không phải của hàm khởi tạo.
        # Truyền vào constructor sẽ bị nuốt im lặng (pydantic extra="ignore").
        handler = self.agent.run(user_msg=question, max_iterations=self.max_iterations)
        try:
            async for ev in handler.stream_events():
                if isinstance(ev, ToolCall) and not isinstance(ev, ToolCallResult):
                    n_tool_calls += 1
                    trajectory.append({"type": "tool_call", "name": ev.tool_name,
                                       "args": dict(ev.tool_kwargs)})
                elif isinstance(ev, ToolCallResult):
                    out = str(ev.tool_output)
                    trajectory.append({
                        "type": "tool_result", "name": ev.tool_name,
                        "args": dict(ev.tool_kwargs),
                        "is_error": bool(getattr(ev.tool_output, "is_error", False)),
                        "output_chars": len(out),
                        "output_head": out[:300],
                    })
                elif isinstance(ev, AgentOutput):
                    n_llm_calls += 1
                    if ev.response.content:
                        trajectory.append({"type": "llm_text",
                                           "text": ev.response.content[:400]})
            resp = await handler
            answer = str(resp)
        except WorkflowRuntimeError as e:
            # Chạm trần số bước. Đây là TRẠNG THÁI LỖI CẦN ĐO, không phải chuyện hiếm.
            hit_cap = True
            error = f"max_iterations({self.max_iterations}): {e}"
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"

        seen: set[str] = set()
        chunks = []
        for c in self._collector:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                chunks.append(c)

        return RagAnswer(
            question=question,
            answer=answer,
            contexts=[c.text for c in chunks],
            chunk_ids=[c.chunk_id for c in chunks],
            origins=[c.origin for c in chunks],
            scores=[c.score for c in chunks],
            n_llm_calls=n_llm_calls,
            n_tool_calls=n_tool_calls,
            trajectory=trajectory,
            error=error,
            # Đúng thứ tool đã trả về cho model (có tiêu đề trích dẫn + dòng LƯU Ý),
            # không phải văn bản thô. Ragas chấm faithfulness trên trường này.
            contexts_shown=format_blocks(chunks),
        )

    def answer(self, question: str) -> RagAnswer:
        return asyncio.run(self._arun(question))


def _fix_windows_loop() -> None:
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    _fix_windows_loop()
    q = sys.argv[1] if len(sys.argv) > 1 else (
        "Hàng hóa bị trả lại khi lưu thông trong nước, nhãn phụ có phải ghi "
        "'Được sản xuất tại Việt Nam' không?"
    )
    a = RegulationAgent()
    r = a.answer(q)
    print(f"\n=== CÂU HỎI ===\n{q}\n")
    print("=== QUỸ ĐẠO ===")
    for i, s in enumerate(r.trajectory, 1):
        if s["type"] == "tool_call":
            print(f"  {i}. gọi {s['name']}({s['args']})")
        elif s["type"] == "tool_result":
            print(f"  {i}. kết quả {s['name']}: {s['output_chars']} ký tự"
                  f"{' [LỖI]' if s['is_error'] else ''}")
        else:
            print(f"  {i}. LLM: {s['text'][:120]}...")
    print(f"\n=== TRẢ LỜI ===\n{r.answer}\n")
    print(f"tool calls={r.n_tool_calls} | llm calls={r.n_llm_calls} | "
          f"chunks={len(r.chunk_ids)} | từ chối={r.refused}")
    if r.error:
        print(f"LỖI: {r.error}")
