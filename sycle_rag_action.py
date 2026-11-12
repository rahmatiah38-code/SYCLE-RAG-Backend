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

# Download File Exel di google drive 'SYCLE_Evidence_Registry_V2.1.xlsx'
FILE_ID =os.getenv("GDRIVE_FILE_ID")
content = download_excel(FILE_ID)
log.info("Berhasil download dari GDrive: %d bytes", len(content))

# Baca exel dan Bersihkan Data
# Mengisi kolom 'pdf_url' dengan link-pdf dari folder 'PDF_DOCUMENT' 
# Folder ID: "PDF_DOCUMENT"
result = list_drive_files(os.getenv("GDRIVE_FOLDER_PDF_ID"))
clean_data = read_and_clean(content, result)
log.info("Siap dimasukkan supabase: %d record", len(clean_data))

# Kirim ke Tabel `evidence_registry` di supabase
insert_data = load_exel_to_supabase(clean_data)

# Mengambil URL dari Tabel 'evidence_registry' di Supabase pada kolom 
# 'source_url' untuk sumber website
# 'pdf_url' untuk sumber PDF
# Melakukan scraping halaman WEB Dan File FDF

web_documents = get_web_documents()
pdf_documents = get_pdf_documents()

#Chunking + Embedding -> simpan ke table 'evidence_chunk' di Supabase
records = process_and_save_chunks(
    web_documents=web_documents,
    pdf_documents=pdf_documents
)


