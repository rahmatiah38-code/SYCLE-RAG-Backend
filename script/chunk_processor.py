import os
import re
import hashlib
from datetime import datetime, timezone
from typing import Any

from supabase_configuration.database import get_supabase
from sentence_transformers import SentenceTransformer

TABLE = os.getenv("EVIDENCE_CHUNK_TABLE")
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_NUM"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP"))

# Model ini menghasilkan embedding 384 dimensi,
# sesuai dengan kolom embedding vector(384) di Supabase.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 50

# Potongan yang lebih pendek dari ini biasanya hanya judul/sisa halaman.
MIN_CHUNK_CHARS = 80

# ============================================================
# TEXT CLEANING
# ============================================================

# Tanda pemenggal kata dari PDF, contoh: "In­ flammatory" -> "Inflammatory".
SOFT_HYPHEN = re.compile(r"(\w)­\s*(\w)")

# Judul bagian belakang dokumen yang tidak dipakai sebagai evidence.
# Semua teks sejak judul pertama yang ditemukan dibuang.
BACK_MATTER = re.compile(
    r"\b(References|Daftar Pustaka|Bibliography|Acknowledg(?:e)?ments?|"
    r"Declaration of competing interest|CRediT authorship|Author contributions)\b",
    re.I,
)

# Pola halaman depan jurnal: afiliasi, email, lisensi, info penerbit.
# Potongan dianggap tidak berguna bila cocok dengan 2 pola atau lebih.
NOISE_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.]+",
    r"Contents lists available",
    r"open access article under",
    r"\b(Department|Faculty|School) of\b",
    r"\bUniversity\b",
    r"Writing\s+[–-]\s+(review|original)",
    r"Received .{0,40}accepted",
    r"A R T I C L E I N F O",
]


def clean_text(text: str) -> str:
    """Memperbaiki kata yang terpenggal lalu merapikan spasi."""
    text = SOFT_HYPHEN.sub(r"\1\2", text or "")
    return " ".join(text.split())


def cut_back_matter(text: str) -> tuple[str, bool]:
    """
    Memotong teks pada judul bagian belakang pertama (References, dll).
    Mengembalikan (teks_tersisa, ditemukan).
    """
    match = BACK_MATTER.search(text or "")
    if not match:
        return text or "", False
    return text[:match.start()], True


def is_noise_chunk(chunk: str) -> bool:
    """True bila potongan berisi daftar pustaka, data penulis, atau terlalu pendek."""
    if len(chunk) < MIN_CHUNK_CHARS:
        return True

    citations = len(re.findall(r"\[\d+\]|doi\.org|https?://", chunk))
    if citations >= 3:
        return True

    hits = sum(bool(re.search(p, chunk, re.I)) for p in NOISE_PATTERNS)
    return hits >= 2


def trim_pdf_documents(pdf_documents: list[dict]) -> list[dict]:
    """
    Daftar pustaka biasanya berlanjut ke beberapa halaman.
    Setelah judul bagian belakang ditemukan di satu halaman,
    sisa halaman itu dan halaman-halaman berikutnya dari sumber yang sama dilewati.
    """
    by_source: dict[str, list[dict]] = {}
    for doc in pdf_documents:
        by_source.setdefault(doc["source_code"], []).append(doc)

    kept = []
    for source_code, docs in by_source.items():
        stop_page = None

        for doc in sorted(docs, key=lambda d: int(d["page"])):
            if stop_page is not None:
                continue

            text, found = cut_back_matter(doc.get("text", ""))
            if found:
                stop_page = int(doc["page"])

            kept.append({**doc, "text": text})

        if stop_page is not None:
            print(f"{source_code}: bagian belakang mulai halaman {stop_page}, sisanya dilewati")

    return kept


def useful_chunks(text: str) -> list[str]:
    """Membersihkan teks, memotong per kalimat, lalu membuang potongan tidak berguna."""
    chunks = chunk_by_sentence(clean_text(text))
    return [chunk for chunk in chunks if not is_noise_chunk(chunk)]


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


def build_metadata(doc: dict, source_code: str, title, source_url, source_type: str, chunk_index: int) -> dict:
    """Metadata yang sama untuk WEB dan PDF, termasuk batas interpretasi dari registry."""
    return {
        "source_code": source_code,
        "title": str(title)[:500],
        "publisher": str(doc.get("publisher", ""))[:300],
        "source_url": str(source_url)[:1500],
        "document_date": doc.get("publication_year"),
        "verified_by": doc.get("verified_by"),
        # Nama kolom berbeda antara registry dan tabel; ambil yang tersedia.
        "peran": get_first_value(doc, "sycle_role", "peran", "Peran SYCLE V2.1", default=None),
        "status": get_first_value(doc, "status_v21", "Status V2.1", default=None),
        "topik": get_first_value(doc, "proposed_topic", "Topik usulan", default=None),
        "batas_interpretasi": get_first_value(
            doc, "interpretation_limit", "Batas interpretasi", default=None
        ),
        "source_type": source_type,
        "chunk_index": chunk_index,
        "retrieved_at": doc.get("retrieved_at"),
    }


# ============================================================
# BUILD CHUNK RECORDS
# ============================================================

def build_chunk_records(
    web_documents: list[dict] | None = None,
    pdf_documents: list[dict] | None = None,
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
    pdf_documents = trim_pdf_documents(pdf_documents or [])

    records = []
    skipped = 0

    # -------------------------
    # WEB documents
    # -------------------------
    for doc in web_documents:
        source_code = doc["source_code"]
        title = get_first_value(doc, "source_title", "title")
        source_url = get_first_value(doc, "source_url", "pdf_url")

        text, _ = cut_back_matter(doc.get("text", ""))
        raw_count = len(chunk_by_sentence(clean_text(text)))
        chunks = useful_chunks(text)
        skipped += raw_count - len(chunks)

        for chunk_index, chunk in enumerate(chunks):
            records.append({
                "id": f"{source_code}-web-c{chunk_index}",
                "source_code": source_code,
                "content": chunk,
                "page": None,
                "chunk_index": chunk_index,
                "metadata": build_metadata(doc, source_code, title, source_url, "web", chunk_index),
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

        text = doc.get("text", "")
        raw_count = len(chunk_by_sentence(clean_text(text)))
        chunks = useful_chunks(text)
        skipped += raw_count - len(chunks)

        for chunk_index, chunk in enumerate(chunks):
            metadata = build_metadata(doc, source_code, title, source_url, "pdf", chunk_index)
            metadata["page"] = page_number

            records.append({
                "id": f"{source_code}-p{page_number}-c{chunk_index}",
                "source_code": source_code,
                "content": chunk,
                "page": page_number,
                "chunk_index": chunk_index,
                "metadata": metadata,
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
    print("Chunk dibuang (pustaka/afiliasi/terlalu pendek):", skipped)

    average_length = (
        sum(len(record["content"]) for record in records) // len(records)
        if records else 0
    )

    print("Rata-rata panjang:", average_length, "karakter")

    empty_role = sum(1 for r in records if not r["metadata"].get("peran"))
    if empty_role:
        print(f"Peringatan: {empty_role} chunk tanpa 'peran' (sycle_role). Cek kolom dokumen masukan.")

    return records


# ============================================================
# EMBEDDING
# ============================================================

def embedding_text(record: dict) -> str:
    """Judul sumber ikut di-embed agar potongan membawa konteks dokumennya."""
    title = (record.get("metadata") or {}).get("title") or ""
    return f"{title}. {record['content']}" if title else record["content"]


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

    texts = [embedding_text(record) for record in records]

    # Teks yang melebihi batas token model akan terpotong tanpa pesan error.
    max_tokens = model.max_seq_length
    too_long = sum(
        1 for text in texts
        if len(model.tokenizer.tokenize(text)) > max_tokens
    )
    print(f"Batas token model: {max_tokens}; chunk yang melebihi batas: {too_long}/{len(texts)}")
    if too_long:
        print("Peringatan: sebagian chunk terpotong saat embedding. Pertimbangkan CHUNK_NUM lebih kecil.")

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

def delete_existing_chunks(source_codes: list[str]) -> None:
    """
    Menghapus chunk lama sumber yang akan diproses ulang.
    Upsert hanya menimpa ID yang sama; ID lama yang tidak terbentuk lagi
    (misalnya karena ukuran chunk berubah) akan tertinggal bila tidak dihapus.
    """
    source_codes = sorted(set(source_codes))
    if not source_codes:
        return

    supabase = get_supabase()
    supabase.table(TABLE).delete().in_("source_code", source_codes).execute()
    print("Chunk lama dihapus untuk:", ", ".join(source_codes))


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
    replace_existing: bool = True,
) -> list[dict]:
    """
    Pipeline:
    WEB + PDF -> Pembersihan -> Chunking -> Embedding -> Supabase.

    replace_existing=True menghapus chunk lama dari sumber yang sama
    sebelum menyimpan, agar tidak ada chunk usang yang tertinggal.
    """
    records = build_chunk_records(
        web_documents=web_documents,
        pdf_documents=pdf_documents
    )

    records = generate_embeddings(records)

    if replace_existing:
        source_codes = [
            doc["source_code"]
            for doc in (web_documents or []) + (pdf_documents or [])
        ]
        delete_existing_chunks(source_codes)

    save_chunks_to_supabase(records)

    return records
