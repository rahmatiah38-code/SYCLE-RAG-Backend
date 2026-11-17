import requests

from bs4 import BeautifulSoup
from datetime import datetime, timezone

from supabase_configuration.database import get_evidence_registry_web


import time
from pathlib import Path

# Header seperti browser biasa. Beberapa situs (mis. CDC, NCBI, Mayo Clinic)
# menolak permintaan dengan identitas yang bukan browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
}

MAX_ATTEMPTS = 2


def fetch_webpage(source, timeout=25):

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                source["source_url"],
                headers=HEADERS,
                timeout=timeout
            )
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(3)

    return {
        **source,
        "text": html_to_text(response.text),
        "retrieved_at": datetime.now(timezone.utc).isoformat()
    }


def html_to_text(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Menghapus elemen yang tidak diperlukan
    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "svg",
        "form",
        "aside"
    ]):
        tag.decompose()

    # Mengambil konten utama
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    return "\n".join(
        line.strip()
        for line in main.get_text("\n").splitlines()
        if line.strip()
    )


# Beberapa situs (CDC, Mayo Clinic, NCBI) menolak scraper (403) walaupun
# memakai header browser. Halaman seperti itu bisa disimpan manual dari
# browser ke folder backend/manual_sources/ dengan nama kode sumber,
# mis. SRC-E02.html atau SRC-E02.pdf. File ini hanya dipakai bila
# pengambilan online gagal; registry tidak perlu diubah.
MANUAL_DIR = Path(__file__).resolve().parent.parent / "manual_sources"


def load_manual_source(source):

    code = source.get("source_code") or ""

    for ext in (".html", ".htm", ".pdf", ".txt"):
        path = MANUAL_DIR / f"{code}{ext}"

        if not path.exists():
            continue

        if ext == ".pdf":
            import fitz
            with fitz.open(path) as pdf:
                text = "\n".join(page.get_text("text") for page in pdf)
        elif ext == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            text = html_to_text(
                path.read_text(encoding="utf-8", errors="ignore")
            )

        text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )

        return {
            **source,
            "text": text,
            "retrieved_at": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "manual_file": path.name,
        }

    return None


def get_web_documents(only_sources=None):

    # Mengambil data registry kategori WEB
    source_web = get_evidence_registry_web()

    # Opsional: hanya proses kode sumber tertentu (ONLY_SOURCES).
    if only_sources:
        source_web = [
            s for s in source_web
            if s.get("source_code") in only_sources
        ]
    print("Jumlah sumber WEB:", len(source_web))

    web_documents = []

    for source in source_web:

        try:
            try:
                doc = fetch_webpage(source)
                error = None
            except Exception as exc:
                doc, error = None, exc

            # Gagal online atau teks terlalu sedikit -> coba file manual.
            if doc is None or len(doc["text"]) < 300:
                manual = load_manual_source(source)

                if manual and len(manual["text"]) >= 300:
                    web_documents.append(manual)
                    print(
                        f"OK (file manual {manual['manual_file']}): "
                        f"{source.get('source_code')} — "
                        f"{len(manual['text'])} karakter"
                    )
                    continue

            if doc is None:
                raise error

            if len(doc["text"]) < 300:

                print(
                    f"SKIP (teks terlalu sedikit): "
                    f"{source.get('source_code')} — "
                    f"{source.get('source_url')}"
                )

                continue

            web_documents.append(doc)

            print(
                f"OK: {source.get('source_code')} — "
                f"{source.get('publisher')} — "
                f"{len(doc['text'])} karakter"
            )

        except Exception as exc:

            print(
                f"SKIP: {source.get('source_code')} — "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        "Halaman web berhasil diambil:",
        len(web_documents)
    )

    return web_documents