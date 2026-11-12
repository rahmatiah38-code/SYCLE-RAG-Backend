import requests

from bs4 import BeautifulSoup
from datetime import datetime, timezone

from supabase_configuration.database import get_evidence_registry_web


HEADERS = {
    "User-Agent": "SYCLE-RAG-Prototype/0.1 (educational, non-commercial)"
}


def fetch_webpage(source, timeout=25):

    response = requests.get(
        source["source_url"],
        headers=HEADERS,
        timeout=timeout
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
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

    text = "\n".join(
        line.strip()
        for line in main.get_text("\n").splitlines()
        if line.strip()
    )

    return {
        **source,
        "text": text,
        "retrieved_at": datetime.now(timezone.utc).isoformat()
    }


def get_web_documents():

    # Mengambil data registry kategori WEB
    source_web = get_evidence_registry_web()
    print("Jumlah sumber WEB:", len(source_web))

    web_documents = []

    for source in source_web:

        try:
            doc = fetch_webpage(source)

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