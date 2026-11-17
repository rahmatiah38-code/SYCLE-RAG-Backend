import requests
import fitz
import re

import os
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

from supabase_configuration.database import get_evidence_registry_pdf


HEADERS = {
    "User-Agent": "SYCLE-RAG-Prototype/0.1 (educational, non-commercial)"
}


# ==========================================
# 1. Mengambil Google Drive File ID
# ==========================================

def get_google_drive_file_id(pdf_url):

    parsed_url = urlparse(pdf_url)

    # Format:
    # https://drive.google.com/file/d/FILE_ID/view

    match = re.search(
        r"/file/d/([a-zA-Z0-9_-]+)",
        parsed_url.path
    )

    if match:
        return match.group(1)

    # Format:
    # https://drive.google.com/open?id=FILE_ID
    # https://drive.google.com/uc?id=FILE_ID

    query_params = parse_qs(parsed_url.query)

    file_id = query_params.get("id", [None])[0]

    if file_id:
        return file_id

    raise ValueError(
        f"Google Drive File ID tidak ditemukan: {pdf_url}"
    )


# ==========================================
# 2. Mengunduh PDF dari Google Drive
# ==========================================

def download_google_drive_pdf(pdf_url, timeout=60):

    file_id = get_google_drive_file_id(pdf_url)

    download_url = "https://drive.google.com/uc"

    session = requests.Session()

    response = session.get(
        download_url,
        params={
            "export": "download",
            "id": file_id
        },
        headers=HEADERS,
        timeout=timeout
    )

    response.raise_for_status()

    # Menangani konfirmasi download file berukuran besar
    confirm_token = next(
        (
            value
            for key, value in response.cookies.items()
            if key.startswith("download_warning")
        ),
        None
    )

    if confirm_token:

        response = session.get(
            download_url,
            params={
                "export": "download",
                "id": file_id,
                "confirm": confirm_token
            },
            headers=HEADERS,
            timeout=timeout
        )

        response.raise_for_status()

    # Memastikan file benar-benar PDF
    pdf_bytes = response.content

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(
            "File yang diunduh bukan PDF. "
            "Periksa URL dan izin akses Google Drive."
        )

    return pdf_bytes


# ==========================================
# 3. Mengekstrak teks PDF per halaman
# ==========================================

def extract_pdf_pages(source):

    pdf_url = source.get("pdf_url")

    if not pdf_url:
        raise ValueError("Kolom pdf_url kosong")

    pdf_bytes = download_google_drive_pdf(pdf_url)

    pdf_file = fitz.open(
        stream=BytesIO(pdf_bytes),
        filetype="pdf"
    )

    pdf_documents = []

    try:

        for page_number, page in enumerate(pdf_file, start=1):

            text = page.get_text("text").strip()

            # Lewati halaman tanpa teks
            if not text:
                print(
                    f"SKIP halaman kosong: "
                    f"{source.get('source_code')} — "
                    f"Halaman {page_number}"
                )
                continue

            document = {
                **source,
                "text": text,
                "page": page_number,
                "retrieved_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            pdf_documents.append(document)

    finally:
        pdf_file.close()

    return pdf_documents


# ==========================================
# 4. Mengambil seluruh PDF dari Supabase
# ==========================================


from datetime import datetime, timezone
import fitz


# PDF cadangan di backend/manual_sources/{kode}.pdf dipakai jika file belum ada
# di folder Drive. FORCE_FULL_TEXT=SRC-E04,... memperlakukan sumber tersebut
# sebagai "Full Text" walau kolom registry masih kosong (atas keputusan tim).
MANUAL_DIR = Path(__file__).resolve().parent.parent / "manual_sources"
FORCE_FULL_TEXT = {
    code.strip() for code in os.getenv("FORCE_FULL_TEXT", "").split(",") if code.strip()
}


def load_pdf_bytes(doc):
    local = MANUAL_DIR / f"{doc.get('source_code')}.pdf"
    if local.exists():
        print(f"PDF lokal: {local.name}")
        return local.read_bytes()
    if not doc.get("pdf_url"):
        raise FileNotFoundError("PDF tidak ada di folder Drive maupun manual_sources")
    return download_google_drive_pdf(doc["pdf_url"])


def get_pdf_documents(only_sources=None):

    source_pdf = get_evidence_registry_pdf()

    # Opsional: hanya proses kode sumber tertentu (ONLY_SOURCES).
    if only_sources:
        source_pdf = [
            s for s in source_pdf
            if s.get("source_code") in only_sources
        ]

    pdf_documents = []

    print("Jumlah sumber PDF:", len(source_pdf))

    for doc in source_pdf:

        # ==================================
        # 1. Pemeriksaan izin indexing
        # ==================================

        if doc.get("source_code") in FORCE_FULL_TEXT:
            doc["index_full_text"] = True

        if not doc.get("index_full_text", False):

            print(
                f"LEWATI (lisensi, parafrase saja): "
                f"{doc.get('source_code')} — "
                f"{doc.get('title')}"
            )

            continue

        # ==================================
        # 1b. Cadangan: artikel tersimpan sebagai HTML (tanpa PDF)
        # ==================================

        html_path = MANUAL_DIR / f"{doc.get('source_code')}.html"
        if html_path.exists() and not (MANUAL_DIR / f"{doc.get('source_code')}.pdf").exists():
            from script.web_scraper import html_to_text

            text = html_to_text(html_path.read_text(encoding="utf-8", errors="ignore"))
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            pdf_documents.append({
                **doc,
                "page": 1,  # satu dokumen utuh; chunk_processor membutuhkan nomor halaman
                "text": text,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"OK (HTML lokal {html_path.name}): {doc.get('source_code')} — {len(text)} karakter")
            continue

        # ==================================
        # 2. Download dan membuka PDF
        # ==================================

        try:

            pdf_bytes = load_pdf_bytes(doc)

            pdf = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

        except Exception as exc:

            print(
                f"SKIP: {doc.get('source_code')} — "
                f"{type(exc).__name__}: {exc}"
            )

            continue

        # ==================================
        # 3. Ekstraksi teks setiap halaman
        # ==================================

        count = 0
        skipped_pages = 0

        try:

            for page_num in range(len(pdf)):

                text = pdf[page_num].get_text("text").strip()

                # Halaman kosong / hasil scan
                if len(text) < 50:

                    skipped_pages += 1

                    continue

                pdf_documents.append({

                    **doc,

                    "page": page_num + 1,

                    "text": text,

                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat()

                })

                count += 1

        finally:

            pdf.close()

        # ==================================
        # 4. Informasi hasil ekstraksi
        # ==================================

        print(
            f"OK: {doc.get('source_code')} — "
            f"{doc.get('publisher')} — "
            f"{count} halaman berhasil — "
            f"{skipped_pages} halaman dilewati"
        )

    # ==================================
    # 5. Ringkasan
    # ==================================

    print("\n========== HASIL EKSTRAKSI PDF ==========")

    print("Jumlah sumber PDF :", len(source_pdf))

    print("Total halaman berhasil:", len(pdf_documents))

    print("Total dokumen hasil ekstraksi:", len(pdf_documents))

    return pdf_documents
