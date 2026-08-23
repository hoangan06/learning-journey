"""Công cụ của agent.

Hai tool, mỗi tool giải quyết một kiểu hỏng khác nhau đã ĐO ĐƯỢC ở Khối 1–2:

1. `search_regulations` — bọc retriever. Tool chính.
2. `check_amendment`   — tra quan hệ sửa đổi. Tồn tại vì một lý do cụ thể:
   `--retrieval-check` cho thấy ngưỡng similarity không tách được câu ngoài
   phạm vi, và câu mh03 chứng minh một Điều có thể được truy hồi đúng nhưng
   nội dung ĐÃ BỊ BÃI BỎ. Quan hệ sửa đổi là quan hệ trong ĐỒ THỊ văn bản,
   không phải quan hệ tương đồng ngữ nghĩa — vector search không thể đảm bảo
   tìm ra nó, nên phải cho agent một đường tra trực tiếp.

Docstring của hai hàm dưới đây CHÍNH LÀ prompt: model chọn tool nào và truyền
tham số gì hoàn toàn dựa vào chúng. Sửa docstring = sửa hành vi agent.
"""

from __future__ import annotations

from typing import Callable

from rag.config import DOC_TITLES
from rag.retriever import RegulationRetriever, RetrievalConfig, format_context

SEP = "|"


def build_tools(
    retriever: RegulationRetriever,
    collector: list,
    top_k: int = 5,
    use_parent_child: bool = True,
    expand_amendments: bool = False,
) -> list[Callable]:
    """Tạo hai tool, đóng gói sẵn retriever và một `collector`.

    `collector` là list dùng để GHI LẠI mọi chunk mà agent đã kéo về. Không có
    nó thì không tính được context recall cho agent — quỹ đạo của agent nằm bên
    trong vòng lặp, không lộ ra ở câu trả lời cuối.
    """

    def search_regulations(query: str, top_k_override: int = 0) -> str:
        """Tra cứu điều khoản trong 5 văn bản pháp quy FMCG Việt Nam.

        Dùng tool này cho MỌI câu hỏi về nội dung quy định. Nếu câu hỏi cần thông
        tin từ nhiều văn bản hoặc nhiều Điều khác nhau, hãy gọi tool NHIỀU LẦN,
        mỗi lần một truy vấn cụ thể, thay vì gộp tất cả vào một truy vấn dài.

        Kho tài liệu gồm: Luật An toàn thực phẩm, Luật Hải quan,
        Nghị định 15/2018 (hướng dẫn Luật ATTP), Nghị định 43/2017 (nhãn hàng hóa),
        Nghị định 111/2021 (sửa đổi Nghị định 43/2017).

        Args:
            query: Truy vấn tiếng Việt, nên chứa từ khóa pháp lý cụ thể
                (ví dụ "đăng ký bản công bố sản phẩm trẻ dưới 36 tháng"),
                không nên chép nguyên văn câu hỏi dài của người dùng.
            top_k_override: Số đoạn cần lấy. Để 0 dùng mặc định.

        Returns:
            Các đoạn văn bản kèm số Điều, số hiệu văn bản và ghi chú sửa đổi (nếu có).
        """
        cfg = RetrievalConfig(
            top_k=top_k_override or top_k,
            use_parent_child=use_parent_child,
            expand_amendments=expand_amendments,
        )
        hits = retriever.search(query, cfg)
        collector.extend(hits)
        if not hits:
            return "Không tìm thấy đoạn nào phù hợp. Hãy thử truy vấn khác với từ khóa pháp lý cụ thể hơn."
        return format_context(hits)

    def check_amendment(document: str, article: str) -> str:
        """Kiểm tra một Điều luật đã bị SỬA ĐỔI hoặc BÃI BỎ hay chưa, và trả về nội dung sửa đổi.

        BẮT BUỘC dùng tool này trước khi khẳng định nội dung của bất kỳ Điều nào
        thuộc Nghị định 43/2017/NĐ-CP, vì nhiều Điều của văn bản này đã bị
        Nghị định 111/2021/NĐ-CP sửa đổi hoặc bãi bỏ. Trả lời theo bản gốc mà
        không kiểm tra sẽ dẫn tới câu trả lời sai luật.

        Args:
            document: Số hiệu văn bản, ví dụ "43/2017/NĐ-CP" hoặc "15/2018/NĐ-CP".
            article: Số Điều, chỉ ghi con số, ví dụ "8" hoặc "15".

        Returns:
            Tình trạng sửa đổi của Điều đó, kèm nguyên văn khoản sửa đổi nếu có.
        """
        doc = (document or "").strip()
        art = str(article or "").strip().replace("Điều", "").strip()

        if doc not in DOC_TITLES:
            return (
                f"Không có văn bản '{document}' trong kho. "
                f"Các số hiệu hợp lệ: {', '.join(DOC_TITLES)}."
            )

        got = retriever.col.get(
            where={"$and": [{"so_hieu": {"$eq": doc}}, {"so_dieu": {"$eq": art}}]},
            include=["documents", "metadatas"],
        )
        if not got["ids"]:
            return f"Không tìm thấy Điều {art} trong {doc}."

        lines = [f"Điều {art} của {doc} — {DOC_TITLES[doc]}"]
        refs: list[str] = []
        statuses: list[str] = []
        amends: list[str] = []
        for meta in got["metadatas"]:
            if meta.get("sua_doi"):
                statuses.append(meta["sua_doi"])
                refs += [r for r in (meta.get("amend_refs") or "").split(SEP) if r]
            amends += [a for a in (meta.get("amends") or "").split(SEP) if a]

        # Gộp lại thay vì in mỗi part một dòng — Điều nhiều part sẽ lặp y hệt nhau.
        for s in dict.fromkeys(statuses):
            lines.append(f"  TÌNH TRẠNG: {s}")
        if amends:
            lines.append(f"  Điều này SỬA ĐỔI các điều: {', '.join(dict.fromkeys(amends))}")

        if not refs:
            if len(lines) == 1:
                lines.append("  TÌNH TRẠNG: không ghi nhận sửa đổi. Nội dung bản gốc vẫn hiệu lực.")
            return "\n".join(lines)

        ref_docs = retriever.col.get(ids=sorted(set(refs)), include=["documents", "metadatas"])
        lines.append("\n  NGUYÊN VĂN NỘI DUNG SỬA ĐỔI:")
        for cid, text, meta in zip(ref_docs["ids"], ref_docs["documents"], ref_docs["metadatas"]):
            lines.append(f"  --- {cid} ---\n{text}")
            collector.append(_FakeChunk(cid, text, dict(meta)))
        return "\n".join(lines)

    return [search_regulations, check_amendment]


class _FakeChunk:
    """Đủ giao diện để `collector` và metric ở Khối 3 xử lý được như một RetrievedChunk."""

    __slots__ = ("chunk_id", "text", "meta", "score", "origin", "hit_rank")

    def __init__(self, chunk_id: str, text: str, meta: dict):
        self.chunk_id = chunk_id
        self.text = text
        self.meta = meta
        self.score = 0.0
        self.origin = "amendment_tool"
        self.hit_rank = 0

    @property
    def citation(self) -> str:
        """PHẢI có, và phải khớp chữ ký của `RetrievedChunk.citation`.

        `format_blocks()` đọc `c.citation` cho MỌI chunk trong collector, kể cả
        chunk do `check_amendment` tạo ra. Thiếu thuộc tính này thì lời gọi đó
        ném `AttributeError` — và nó nằm NGOÀI khối try/except trong `_arun`,
        nên hỏng cả câu trả lời chứ không chỉ ghi vào trường `error`.

        Lần chạy đầu thoát nạn do may: cả hai lần `check_amendment` được gọi đều
        trả về "không ghi nhận sửa đổi", nên không `_FakeChunk` nào được tạo.
        Đúng câu có sửa đổi thật mà agent chịu gọi tool là sập.
        """
        m = self.meta
        s = f"Điều {m.get('so_dieu', '?')} {m.get('so_hieu', '?')}"
        if m.get("n_parts", 1) > 1:
            s += f" (phần {m.get('part_index')}/{m.get('n_parts')})"
        return s
