import os
import re
import hashlib
from datetime import datetime, timezone
from typing import Any

from database import get_supabase
from sentence_transformers import SentenceTransformer

TABLE = os.getenv("EVIDENCE_CHUNK_TABLE")
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_CHARS = 100

# Model ini menghasilkan embedding 384 dimensi,
# sesuai dengan kolom embedding vector(384) di Supabase.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 50

# ============================================================
# TEXT CHUNKING
# ============================================================

def split_sentences(text: str) -> list[str]:
    """Normalisasi whitespace lalu memisahkan teks berdasarkan tanda baca."""
    text = " ".join((text or "").split())

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;:])\s+", text)
        if sentence.strip()
    ]


def hard_split(
    sentence: str,
    size: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """Memecah kalimat yang terlalu panjang dengan overlap karakter."""
    if size <= 0:
        raise ValueError("size harus lebih besar dari 0")

    if overlap_chars < 0 or overlap_chars >= size:
        raise ValueError("overlap_chars harus >= 0 dan lebih kecil dari size")

    if len(sentence) <= size:
        return [sentence]

    step = size - overlap_chars

    return [
        sentence[i:i + size]
        for i in range(0, len(sentence), step)
    ]


def chunk_by_sentence(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """
    Membuat chunk berdasarkan batas kalimat.
    Kalimat panjang akan dipecah dengan hard_split.
    Chunk berikutnya membawa kalimat sebelumnya selama masih
    berada dalam batas overlap dan max_chars.
    """
    if max_chars <= 0:
        raise ValueError("max_chars harus lebih besar dari 0")

    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars harus >= 0 dan lebih kecil dari max_chars"
        )

    sentences = []

    for sentence in split_sentences(text):
        if len(sentence) > max_chars:
            sentences.extend(
                hard_split(sentence, max_chars, overlap_chars)
            )
        else:
            sentences.append(sentence)

    chunks = []
    current = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])

        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))

            # Ambil kalimat terakhir sebagai overlap jika memungkinkan.
            carry = []
            size = 0

            for previous in reversed(current):
                extra_size = len(previous) + (1 if carry else 0)

                if size + extra_size > overlap_chars:
                    break

                carry.insert(0, previous)
                size += extra_size

            current = (
                carry
                if carry and len(" ".join(carry + [sentence])) <= max_chars
                else []
            )

        current.append(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ============================================================
# METADATA HELPERS
# ============================================================

def get_first_value(data: dict, *keys: str, default: Any = ""):
    """Mengambil nilai pertama yang tersedia dari beberapa nama kolom."""
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def make_checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ============================================================
# BUILD CHUNK RECORDS
# ============================================================

def build_chunk_records(
    web_documents: list[dict] | None = None,
    pdf_documents: list[dict] | None = None,
    peran: dict | None = None
) -> list[dict]:
    """
    Menggabungkan dokumen WEB dan PDF menjadi records untuk evidence_chunks.

    Struktur output setiap record:
    {
        id, source_code, content, page, chunk_index,
        metadata, checksum
    }
    """
    web_documents = web_documents or []
    pdf_documents = pdf_documents or []
    peran = peran or {}

    records = []

    # -------------------------
    # WEB documents
    # -------------------------
    for doc in web_documents:
        source_code = doc["source_code"]
        title = get_first_value(doc, "source_title", "title")
        source_url = get_first_value(doc, "source_url", "pdf_url")

        chunks = chunk_by_sentence(doc.get("text", ""))

        for chunk_index, chunk in enumerate(chunks):
            records.append({
                "id": f"{source_code}-web-c{chunk_index}",
                "source_code": source_code,
                "content": chunk,
                "page": None,
                "chunk_index": chunk_index,
                "metadata": {
                    "source_code": source_code,
                    "title": str(title)[:500],
                    "publisher": str(doc.get("publisher", ""))[:300],
                    "source_url": str(source_url)[:1500],
                    "document_date": doc.get("publication_year"),
                    "verified_by": doc.get("verified_by"),
                    "peran": str(doc.get("peran", peran.get(source_code, ""))),
                    "source_type": "web",
                    "chunk_index": chunk_index,
                    "retrieved_at": doc.get("retrieved_at"),
                },
                "checksum": make_checksum(chunk),
            })

    # -------------------------
    # PDF documents (per page)
    # -------------------------
    for doc in pdf_documents:
        source_code = doc["source_code"]
        title = get_first_value(doc, "source_title", "title")
        source_url = get_first_value(doc, "source_url", "pdf_url")
        page_number = int(doc["page"])

        chunks = chunk_by_sentence(doc.get("text", ""))

        for chunk_index, chunk in enumerate(chunks):
            records.append({
                "id": f"{source_code}-p{page_number}-c{chunk_index}",
                "source_code": source_code,
                "content": chunk,
                "page": page_number,
                "chunk_index": chunk_index,
                "metadata": {
                    "source_code": source_code,
                    "title": str(title)[:500],
                    "publisher": str(doc.get("publisher", ""))[:300],
                    "source_url": str(source_url)[:1500],
                    "document_date": doc.get("publication_year"),
                    "verified_by": doc.get("verified_by"),
                    "peran": str(peran.get(source_code, doc.get("peran", ""))),
                    "source_type": "pdf",
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "retrieved_at": doc.get("retrieved_at"),
                },
                "checksum": make_checksum(chunk),
            })

    # Validasi ID chunk agar tidak ada duplikasi.
    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        duplicates = sorted({
            chunk_id for chunk_id in ids if ids.count(chunk_id) > 1
        })
        raise ValueError(f"Ada ID chunk ganda: {duplicates[:10]}")

    print("Total chunk:", len(records))

    average_length = (
        sum(len(record["content"]) for record in records) // len(records)
        if records else 0
    )

    print("Rata-rata panjang:", average_length, "karakter")

    return records


# ============================================================
# EMBEDDING
# ============================================================

def generate_embeddings(
    records: list[dict],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> list[dict]:
    """Membuat embedding untuk kolom vector(384)."""
    if not records:
        return records

    print(f"Memuat model embedding: {model_name}")
    model = SentenceTransformer(model_name)

    dimension = model.get_sentence_embedding_dimension()

    if dimension != 384:
        raise ValueError(
            f"Dimensi embedding model {dimension}, "
            "sedangkan kolom Supabase menggunakan vector(384)."
        )

    texts = [record["content"] for record in records]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding.tolist()

    print("Embedding berhasil dibuat:", len(records))

    return records


# ============================================================
# SAVE TO SUPABASE
# ============================================================

def save_chunks_to_supabase(
    records: list[dict],
    batch_size: int = UPSERT_BATCH_SIZE
) -> int:
    """
    Menyimpan chunk ke evidence_chunks.
    Upsert berdasarkan primary key id agar proses dapat dijalankan ulang.
    """
    if not records:
        print("Tidak ada chunk untuk disimpan.")
        return 0

    supabase = get_supabase()
    saved_count = 0

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        response = (
            supabase.table(TABLE)
            .upsert(batch, on_conflict="id")
            .execute()
        )

        saved = len(response.data or [])
        saved_count += saved

        print(
            f"Upsert batch {start // batch_size + 1}: "
            f"{saved}/{len(batch)} chunk"
        )

    print("Total chunk tersimpan:", saved_count)

    return saved_count


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_and_save_chunks(
    web_documents: list[dict] | None = None,
    pdf_documents: list[dict] | None = None,
    peran: dict | None = None
) -> list[dict]:
    """
    Pipeline:
    WEB + PDF -> Chunking -> Embedding -> Supabase.
    """
    records = build_chunk_records(
        web_documents=web_documents,
        pdf_documents=pdf_documents,
        peran=peran
    )

    records = generate_embeddings(records)

    save_chunks_to_supabase(records)

    return records


if __name__ == "__main__":
    print(
        "Modul chunk_processor.py siap digunakan. "
        "Panggil process_and_save_chunks() dari main.py "
        "dengan web_documents dan pdf_documents."
    )
