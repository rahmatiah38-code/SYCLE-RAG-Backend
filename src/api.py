from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.hybrid_retriever import HybridRetriever
from rag.generation import generate_answer
import time


app = FastAPI(title="Sycle Climate Change RAG API")
retriever = HybridRetriever()


# Jika hanya dipanggil melalui Next.js proxy, CORS browser langsung tidak wajib.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "https://chatbot-rag-henna.vercel.app/"],
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

    start = time.time()
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong")
    try:
        t1 = time.time()
        contexts = retriever.hybrid_retrieve(question,top_k=4)
        print(f"Retrieval: {time.time() - t1:.2f}s")

        t2 = time.time()
        answer = generate_answer(question, contexts)
        print(f"Generation: {time.time() - t2:.2f}s")

        print(f"Total: {time.time() - start:.2f}s")
        return {
            "answer": answer,
            "sources": sorted({item["source"] for item in contexts}),
        }
    except Exception as exc:
        print(f"RAG Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
