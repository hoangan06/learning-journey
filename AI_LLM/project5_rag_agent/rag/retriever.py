"""Khối 1 — Retriever trên Chroma, thay cho brute-force `emb @ q` của Project 4.

Module này CỐ Ý không biết gì về LLM hay agent: nó chỉ nhận câu hỏi, trả về
các đoạn văn bản. Nhờ vậy Khối 4 quét được hàng chục biến thể retrieval mà
không phải khởi động agent, không tốn quota Gemini, và không lẫn biến giữa
tầng truy hồi với tầng sinh.

Ba công tắc (bật/tắt độc lập -> chính là các biến thí nghiệm của Khối 4):
  top_k              : lấy bao nhiêu hit từ vector search
  use_parent_child   : hit thuộc Điều bị cắt nhiều part -> kéo các part còn lại
  expand_amendments  : hit là Điều đã bị sửa -> kéo theo khoản sửa đổi

Chạy thử nhanh:
    python -m rag.retriever "hồ sơ tự công bố sản phẩm gồm những gì"
    python -m rag.retriever "nhãn hàng hóa nhập khẩu bắt buộc ghi gì" --amendments
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterable

import chromadb

from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    E5_QUERY_PREFIX,
    EMBED_MODEL_NAME,
)

SEP = "|"


# --------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    """Một cấu hình retrieval = một dòng trong bảng so sánh ở Khối 4."""

    top_k: int = DEFAULT_TOP_K
    use_parent_child: bool = True
    expand_amendments: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            bits = [f"k{self.top_k}"]
            bits.append("pc" if self.use_parent_child else "nopc")
            if self.expand_amendments:
                bits.append("amend")
            self.name = "_".join(bits)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float           # cosine similarity với câu hỏi (1.0 = trùng khớp)
    meta: dict[str, Any]
    origin: str            # "hit" | "sibling" | "amendment"
    hit_rank: int          # thứ hạng của hit đã kéo đoạn này vào (0-based)

    @property
    def citation(self) -> str:
        m = self.meta
        s = f"Điều {m.get('so_dieu', '?')} {m.get('so_hieu', '?')}"
        if m.get("n_parts", 1) > 1:
            s += f" (phần {m.get('part_index')}/{m.get('n_parts')})"
        return s


# --------------------------------------------------------------------------
class RegulationRetriever:
    def __init__(
        self,
        persist_dir=CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embed_model_name: str = EMBED_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        if not persist_dir.exists():
            sys.exit(
                f"Chưa có index tại {persist_dir}. Chạy trước: python -m rag.build_index"
            )
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self.col = self._client.get_collection(collection_name)
        self._embed_model_name = embed_model_name
        self._device = device
        self._model = None  # nạp lười: chỉ tải khi thật sự cần embed câu hỏi

    # -- embedding ---------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self._device
            if device is None:
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            print(f"[retriever] nạp {self._embed_model_name} trên {device}...")
            self._model = SentenceTransformer(self._embed_model_name, device=device)
        return self._model

    def embed_query(self, query: str) -> list[float]:
        # Prefix "query: " là bắt buộc với e5 và phải khớp với "passage: "
        # đã dùng lúc embed văn bản ở Project 4.
        vec = self.model.encode(
            [E5_QUERY_PREFIX + query], normalize_embeddings=True
        )[0]
        return vec.astype(float).tolist()

    # -- truy hồi ----------------------------------------------------------
    def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        **overrides,
    ) -> list[RetrievedChunk]:
        cfg = config or RetrievalConfig(**overrides)

        res = self.col.query(
            query_embeddings=[self.embed_query(query)],
            n_results=cfg.top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        out: list[RetrievedChunk] = []
        seen: set[str] = set()
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    score=1.0 - float(dist),  # chroma cosine space: distance = 1 - sim
                    meta=dict(meta),
                    origin="hit",
                    hit_rank=rank,
                )
            )
            seen.add(cid)

        if cfg.use_parent_child:
            out += self._expand_parents(out, seen)
        if cfg.expand_amendments:
            out += self._expand_amendments(out, seen)

        # Xếp lại theo (thứ hạng hit, thứ tự part) để context đọc liền mạch,
        # đúng trình tự văn bản gốc thay vì trình tự điểm số.
        order = {"hit": 0, "sibling": 1, "amendment": 2}
        out.sort(key=lambda c: (c.hit_rank, order[c.origin], c.meta.get("part_index", 0)))
        return out

    def _expand_parents(
        self, hits: list[RetrievedChunk], seen: set[str]
    ) -> list[RetrievedChunk]:
        """Hit nằm trong một Điều bị cắt nhiều part -> kéo nốt các part còn lại.

        Đây chính là cách vá lỗi 'retrieval trượt vì Điều bị cắt' phát hiện ở
        Project 4: chunk con lọt top-k nhưng thiếu phần thân của cùng một Điều.
        """
        wanted = {h.meta["parent_id"]: h for h in hits if h.meta.get("n_parts", 1) > 1}
        if not wanted:
            return []

        got = self.col.get(
            where={"parent_id": {"$in": list(wanted)}},
            include=["documents", "metadatas"],
        )
        extra: list[RetrievedChunk] = []
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            if cid in seen:
                continue
            seen.add(cid)
            src = wanted[meta["parent_id"]]
            extra.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    # Kế thừa điểm của hit đã kéo nó vào: đoạn này không tự
                    # khớp câu hỏi, nó vào context nhờ quan hệ cấu trúc.
                    score=src.score,
                    meta=dict(meta),
                    origin="sibling",
                    hit_rank=src.hit_rank,
                )
            )
        return extra

    def _expand_amendments(
        self, current: list[RetrievedChunk], seen: set[str]
    ) -> list[RetrievedChunk]:
        """Điều đã bị sửa -> kéo theo khoản sửa đổi ở văn bản khác.

        Không có bước này, hỏi về NĐ 43/2017 sẽ nhận đúng bản gốc 2017 và bỏ
        qua NĐ 111/2021 đã sửa nó. Câu trả lời vẫn 'trung thực với context'
        (faithfulness cao) nhưng SAI LUẬT — kiểu hỏng mà groundedness không bắt được.
        """
        targets: list[str] = []
        owner: dict[str, RetrievedChunk] = {}
        for c in current:
            for ref in _split(c.meta.get("amend_refs", "")):
                if ref not in seen:
                    targets.append(ref)
                    owner.setdefault(ref, c)
        if not targets:
            return []

        got = self.col.get(ids=list(dict.fromkeys(targets)),
                           include=["documents", "metadatas"])
        extra: list[RetrievedChunk] = []
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            if cid in seen:
                continue
            seen.add(cid)
            src = owner[cid]
            extra.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    score=src.score,
                    meta=dict(meta),
                    origin="amendment",
                    hit_rank=src.hit_rank,
                )
            )
        return extra

    # -- tiện ích ----------------------------------------------------------
    def count(self) -> int:
        return self.col.count()


# --------------------------------------------------------------------------
def _split(s: str) -> list[str]:
    return [x for x in (s or "").split(SEP) if x]


def format_context(chunks: Iterable[RetrievedChunk], max_chars: int | None = None) -> str:
    """Ghép các đoạn thành context cho LLM, kèm nhãn trích dẫn.

    Nhãn `[origin]` được giữ lại có chủ đích: khi đọc trace sẽ thấy ngay đoạn
    nào do vector search lấy về, đoạn nào do mở rộng cấu trúc kéo theo.
    """
    chunks = list(chunks)
    parts, total = [], 0
    for i, c in enumerate(chunks, 1):
        head = f"[{i}] {c.citation} — {c.meta.get('ten_dieu', '')} ({c.origin}"
        head += f", sim={c.score:.3f})" if c.origin == "hit" else ")"
        if c.meta.get("sua_doi"):
            head += f"\n    LƯU Ý: {c.meta['sua_doi']}"
        block = f"{head}\n{c.text}"
        if max_chars and total + len(block) > max_chars:
            parts.append(f"... (cắt bớt {len(chunks) - i + 1} đoạn do giới hạn độ dài)")
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--no-parent-child", action="store_true")
    ap.add_argument("--amendments", action="store_true")
    ap.add_argument("--full", action="store_true", help="in toàn văn thay vì rút gọn")
    a = ap.parse_args()

    r = RegulationRetriever()
    cfg = RetrievalConfig(
        top_k=a.top_k,
        use_parent_child=not a.no_parent_child,
        expand_amendments=a.amendments,
    )
    hits = r.search(a.query, cfg)

    print(f"\n== {cfg.name} | {len(hits)} đoạn ({r.count()} chunk trong index) ==\n")
    for i, c in enumerate(hits, 1):
        print(f"[{i}] {c.chunk_id}  origin={c.origin}  sim={c.score:.4f}")
        print(f"    {c.meta.get('ten_dieu', '')}")
        if c.meta.get("sua_doi"):
            print(f"    !! {c.meta['sua_doi']}")
        print("   ", c.text if a.full else c.text[:200].replace("\n", " ") + "...")
        print()
