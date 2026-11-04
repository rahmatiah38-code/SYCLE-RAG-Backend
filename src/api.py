from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.hybrid_retriever import HybridRetriever
from rag.generation import generate_answer


app = FastAPI(title="Sycle Climate Change RAG API")
retriever = HybridRetriever()


# Jika hanya dipanggil melalui Next.js proxy, CORS browser langsung tidak wajib.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "https://vercel.app"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


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
