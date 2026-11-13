import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.generation import generate_answer
from src.rag.refleksi_refactor import refleksi_tracker

app = FastAPI(title="Climate Change RAG API")
retriever = HybridRetriever()


# Jika hanya dipanggil melalui Next.js proxy, CORS browser langsung tidak wajib.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest):
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong")
    try:
        contexts = retriever.hybrid_retrieve(question,top_k=4)
        answer = generate_answer(question, contexts)
        return {
            "answer": answer,
            "sources": sorted({item["source"] for item in contexts}),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class Iklim(BaseModel):
    label: str
    value: float | str
    unit: str | None = None


class Catatan(BaseModel):
    kondisi: str = Field(min_length=1, max_length=100)
    gejala: list[str] = Field(default_factory=list, max_length=30)
    iklim: list[Iklim] = Field(default_factory=list, max_length=10)
    cerita: str = Field(default="", max_length=2000)


class RefleksiRequest(BaseModel):
    catatan: Catatan

@app.post("/refleksi")
def chat(payload: RefleksiRequest):
    catatan = payload.catatan
    if not catatan.kondisi.strip():
        raise HTTPException(status_code=400, detail="kondisi tidak boleh kosong")
    try:
        contexts = retriever.hybrid_retrieve(
            query_catatan(catatan),
            top_k=4,
            where="Tracker dan Personalisasi"
        )

        refleksi = refleksi_tracker(catatan.model_dump(), contexts)
        
        return {
            **refleksi,
            "relatedArticles": artikel_terkait(contexts)
        }

    except Exception as exc:
        logger.exception("Gagal memproses /refleksi")
        raise HTTPException(status_code=500, detail=str(exc))

def query_catatan(c: Catatan) -> str:
    bagian = [c.kondisi, *c.gejala]
    if c.cerita.strip():
        bagian.append(c.cerita.strip())
    label = " ".join(i.label.lower() for i in c.iklim)
    for kunci, topik in TOPIK_IKLIM.items():
        if kunci in label:
            bagian.append(topik)
    return " ".join(bagian)

TOPIK_IKLIM = {
    "panas": "paparan panas",
    "suhu": "paparan panas suhu tinggi",
    "kelembapan": "kelembapan panas",
    "udara": "polusi udara kualitas udara",
    "uv": "radiasi UV", 
}

def artikel_terkait(contexts: list[dict]) -> list[dict]:
    hasil, terlihat = [], set()
    for item in contexts:
        meta = item.get("metadata") or {}
        url = meta.get("source_url")
        kunci = url or item.get("source")
        if not kunci or kunci in terlihat:
            continue
        terlihat.add(kunci)
        hasil.append({
            "title": meta.get("title") or item.get("source"),
            "url": url,
            "publisher": meta.get("publisher"),
        })
    return hasil
