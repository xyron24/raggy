"""Text chunking service built on LangChain splitters."""

from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import Chunk, DocumentIn


class ChunkingService:
    """Splits documents into overlapping passages for embedding.

    Uses character-length splitting (not token counting) to stay fast and
    dependency-light; sizes are tuned through config.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_text(self, text: str) -> list[str]:
        """Split raw text into passages. Blank input yields no passages."""
        if not text or not text.strip():
            return []
        return self._splitter.split_text(text)

    def chunk_document(self, document: DocumentIn) -> list[Chunk]:
        """Split a document, assigning ids and propagating metadata."""
        doc_id = document.doc_id or uuid4().hex
        passages = self.split_text(document.text)
        return [
            Chunk(
                chunk_id=f"{doc_id}:{position}",
                doc_id=doc_id,
                position=position,
                text=passage,
                title=document.title,
                metadata=dict(document.metadata),
            )
            for position, passage in enumerate(passages)
        ]
