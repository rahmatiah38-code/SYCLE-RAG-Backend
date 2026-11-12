import io
import json
import logging
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("RAG-Sycle-evidence-action")

raw_scopes = os.getenv("SCOPES","")
SCOPES = [s.strip() for s in raw_scopes.split(",") if s.strip()]
GSHEET = os.getenv("GSHEET_MIME")
XLSX = os.getenv("XLSX_MIME")
FILE_ID =os.getenv("GDRIVE_FILE_ID")

# Baca Credencial goole, service-account.json
cred_filename = os.getenv("SA_JSON_PATH")
JSON_PATH = BASE_DIR / cred_filename


def download_excel(file_id: str, sa_info: dict) -> bytes:
    creds = service_account.Credentials.from_service_account_info(
        sa_info, scopes=SCOPES
    )
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    meta = (
        drive.files()
        .get(fileId=file_id, fields="id,name,mimeType", supportsAllDrives=True)
        .execute()
    )
    log.info("File Drive: %s (%s)", meta["name"], meta["mimeType"])

    if meta["mimeType"] == GSHEET:
        request = drive.files().export_media(fileId=file_id, mimeType=XLSX)
    else:
        request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()

def read_and_clean(content: bytes) -> list[dict]:
    sheet = os.environ.get("SHEET_NAME") or 0
    raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet, engine="openpyxl")
    log.info("Terbaca %d baris, %d kolom dari Excel", len(raw), len(raw.columns))
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Kode"][0]
    registry = raw.iloc[header_row + 1:].copy()
    registry.columns = raw.iloc[header_row].astype(str).str.strip()
    registry = registry[registry["Kode"].astype(str).str.startswith(("SRC", "EDU"))].reset_index(drop=True)

    FULLTEXT_MAP = {
        "full text": True,
        "parafrase saja": False,
        }
    
    registry = registry.rename(columns={
        "Kode": "source_code",
        "Judul sumber": "title",
        "Organisasi / penulis": "publisher",
        "Tahun": "publication_year",
        "Pemakaian usulan": "proposed_use",
        "Cakupan verifikasi": "verification_scope",
        "Link pdf": "pdf_url",
        "Akses / lisensi": "access_license",
        "Topik usulan": "proposed_topic",
        "URL sumber": "source_url",
        "DOI": "doi",
        "Batas interpretasi": "interpretation_limit",
        "Catatan verifikasi": "verification_notes",
        "Review tim": "team_review_status",
        "Reviewer UUID": "reviewer_uuid",
        "Waktu review": "reviewed_at",
        "UUID sumber": "source_uuid",
        "Peran SYCLE V2.1": "sycle_role",
        "Status V2.1": "status_v21",
        "Full Text Terindex": "index_full_text",
    })

    # Buat adjustment beberapa isi kolom (pdf_url dan index_full_text)
    col = registry["index_full_text"].astype("string").str.strip().str.lower()
    col = col.replace("", pd.NA)
    unknown_vals = set(col.dropna().unique()) - set(FULLTEXT_MAP)
    if unknown_vals:
        log.warning("Nilai index_full_text tidak dikenali: %s", unknown_vals)
    registry["index_full_text"] = col.map(FULLTEXT_MAP).astype("object")
    
    # to_json mengubah NaN -> null dan Timestamp -> string ISO,
    # sehingga aman dikirim sebagai JSON ke Supabase.
    records = json.loads(registry.to_json(orient="records", date_format="iso"))
    return records
