import os
import logging
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

TABLE = os.getenv("EVIDENCE_TABLE")
CONFLIC = os.getenv("ON_CONFLIC") or None
PROJECT_NAME = os.getenv("PROJECT_NAME")
log = logging.getLogger("database.py")


def get_supabase():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise ValueError(
            "SUPABASE_URL / SUPABASE_SECRET_KEY belum ada di .env"
        )

    return create_client(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY
    )

#Insert dari file exel ke Supabase 'evidence_registry'
def load_exel_to_supabase(records: list[dict]) -> int:
    if not records:
        log.warning("Tidak ada data untuk dimasukkan")
        return 0

    supabase = get_supabase()
    batch_size = 20
    saved_count = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        response = (
                    supabase.table(TABLE)
                    .upsert(batch, on_conflict="source_code")
                    .execute()
                )
        saved = len(response.data or [])
        saved_count += saved
        log.info("Batch %d terkirim (%d/%d)", i // batch_size + 1, saved, len(records))
 
    log.info("Selesai: %d record masuk ke %s.%s", saved_count, PROJECT_NAME, TABLE)
    
    return saved_count


def get_evidence_registry_pdf():
    """
    Mengambil evidence_registry yang verification_scope
    mengandung kata PDF.
    """

    supabase = get_supabase()

    response = (
        supabase.table(TABLE)
        .select("*")
        .ilike("verification_scope", "%PDF%")
        .ilike("status_v21", "%Dipakai%")
        .execute()
    )

    return response.data


def get_evidence_registry_web():
    """
    Mengambil evidence_registry yang verification_scope
    tidak mengandung kata PDF.
    """

    supabase = get_supabase()

    response = (
        supabase.table(TABLE)
        .select("*")
        .not_.ilike("verification_scope", "%PDF%")
        .ilike("status_v21", "%Dipakai%")
        .execute()
    )

    return response.data