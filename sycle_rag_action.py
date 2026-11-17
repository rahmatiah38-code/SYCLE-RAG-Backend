import logging
import os
from dotenv import load_dotenv
import pandas as pd
"""
Langkah:
  1. Autentikasi ke Google Drive memakai service account
  2. Download file Excel (.xlsx, atau Google Sheets yang diekspor ke .xlsx)
  3. Baca Excel dengan pandas
  4. Bersihkan data (sesuaikan nama kolom, baris kosong, NaN)
  5. Kirim ke tabel Supabase `evidence_registry` 
  6. Mengambil URL dari Tabel 'evidence_registry' di Supabase pada kolom 'source_url' untuk sumber website, dan 'pdf_url' untuk sumber PDF
  7. Melakukan scraping halaman WEB Dan File FDF
  8. Chunking + Embedding Kemudian menyimpan hasil chunking dan Embediing ke Supabase di table 'evidence_chunk'
"""

from supabase_configuration.database import load_exel_to_supabase
from script.authenticate_google import download_excel, read_and_clean, list_drive_files
from script.web_scraper import get_web_documents
from script.pdf_processor import get_pdf_documents
from script.chunk_processor import process_and_save_chunks

log = logging.getLogger("RAG-Sycle: sycle_rag_action")
load_dotenv()

# Opsional: ONLY_SOURCES=SRC-P02,SRC-E02 hanya memproses sumber tersebut.
# Excel tidak diunggah ulang dan chunk sumber lain tidak disentuh,
# jadi evidence yang sudah ada (Tanya Sycle, Refleksi, Edukasi) tetap sama.
ONLY_SOURCES = {
    code.strip()
    for code in os.getenv("ONLY_SOURCES", "").split(",")
    if code.strip()
}

if ONLY_SOURCES:
    log.info("Mode ONLY_SOURCES: %s", sorted(ONLY_SOURCES))
    print("Mode ONLY_SOURCES:", ", ".join(sorted(ONLY_SOURCES)))
else:
    # Download File Exel di google drive 'SYCLE_Evidence_Registry_V2.1.xlsx'
    FILE_ID = os.getenv("GDRIVE_FILE_ID")
    content = download_excel(FILE_ID)
    log.info("Berhasil download dari GDrive: %d bytes", len(content))

    # Baca exel dan Bersihkan Data
    # Mengisi kolom 'pdf_url' dengan link-pdf dari folder 'PDF_DOCUMENT'
    result = list_drive_files(os.getenv("GDRIVE_FOLDER_PDF_ID"))
    clean_data = read_and_clean(content, result)
    log.info("Siap dimasukkan supabase: %d record", len(clean_data))

    # Kirim ke Tabel `evidence_registry` di supabase
    insert_data = load_exel_to_supabase(clean_data)

# Mengambil URL dari Tabel 'evidence_registry' di Supabase pada kolom
# 'source_url' untuk sumber website dan 'pdf_url' untuk sumber PDF,
# lalu scraping halaman WEB dan file PDF.
web_documents = get_web_documents(ONLY_SOURCES or None)
pdf_documents = get_pdf_documents(ONLY_SOURCES or None)

if ONLY_SOURCES:
    found = {d.get("source_code") for d in web_documents + pdf_documents}
    missing = sorted(ONLY_SOURCES - found)
    if missing:
        print("Tidak berhasil diambil:", ", ".join(missing))

# DRY_RUN=1: hanya ambil dan potong teks, tanpa embedding dan tanpa menulis
# ke Supabase. Dipakai untuk mengecek hasil sebelum benar-benar disimpan.
if os.getenv("DRY_RUN"):
    from collections import Counter
    from script.chunk_processor import build_chunk_records

    preview = build_chunk_records(
        web_documents=web_documents,
        pdf_documents=pdf_documents
    )
    counts = Counter(r["source_code"] for r in preview)
    print("DRY_RUN - tidak ada yang disimpan ke Supabase.")
    for code, total in sorted(counts.items()):
        print(f"  {code}: {total} chunk")
    for record in preview[:2]:
        print("\nContoh chunk", record["id"], ":\n", record["content"][:400])
else:
    # Chunking + Embedding -> simpan ke table 'evidence_chunk' di Supabase
    records = process_and_save_chunks(
        web_documents=web_documents,
        pdf_documents=pdf_documents
    )
