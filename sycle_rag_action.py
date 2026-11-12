import io
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import pandas as pd
"""
Langkah:
  1. Autentikasi ke Google Drive memakai service account
  2. Download file Excel (.xlsx, atau Google Sheets yang diekspor ke .xlsx)
  3. Baca Excel dengan pandas
  4. Bersihkan data (nama kolom, baris kosong, NaN, tanggal)
  5. Kirim ke tabel Supabase `evidence_registry` 
  6. Mengambil URL dari Tabel 'evidence_registry' di Supabase pada kolom 'source_url' untuk sumber website, dan 'pdf_url' untuk sumber PDF
  7. Melakukan scraping halaman WEB Dan File FDF
  8. Chunking + Embedding Kemudian menyimpan hasil chunking dan Embediing ke Supabase di table 'evidence_chunk'
"""

from supabase_configuration.database import (
    get_evidence_registry_pdf,
    get_evidence_registry_web,
    load_exel_to_supabase
)
from script.authenticate_google import download_excel, read_and_clean
from script.web_scraper import get_web_documents
from script.pdf_processor import get_pdf_documents
# from script.chunk_processor import process_and_save_chunks

log = logging.getLogger("RAG-Sycle-evidence-action-main")
load_dotenv()

# Baca Credencial goole, service-account.json
JSON_PATH = os.getenv("SA_JSON_PATH")

# Download File Exel di google drive
FILE_ID =os.getenv("GDRIVE_FILE_ID")
with open(JSON_PATH, encoding="utf-8") as f:
        sa_info = json.load(f)
content = download_excel(FILE_ID, sa_info)
log.info("Berhasil download dari GDrive: %d bytes", len(content))

# Baca exel dan Bersihkan Data(nama kolom, baris kosong dan NaN)
clean_data = read_and_clean(content)
log.info("Siap dimasukkan supabase: %d record", len(clean_data))

# Kirim ke Tabel `evidence_registry` di supabase
# ==============================================
insert_data = load_exel_to_supabase(clean_data)

# Mengambil URL dari Tabel 'evidence_registry' di Supabase pada kolom 
# 'source_url' untuk sumber website
# 'pdf_url' untuk sumber PDF
# ===================================================================

source_pdf = get_evidence_registry_pdf()
source_web = get_evidence_registry_web()

# print("Jumlah sumber PDF:", len(source_pdf))
print("Jumlah sumber WEB:", len(source_web))


# # Melakukan scraping halaman WEB Dan File FDF
# # ===========================================

web_documents = get_web_documents()
pdf_documents = get_pdf_documents()

# # Chunking + Embedding + Supabase
# records = process_and_save_chunks(
#     web_documents=web_documents,
#     pdf_documents=pdf_documents
# )


