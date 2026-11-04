import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from database import get_supabase


STOPWORDS = set("""
yang dan di ke dari untuk pada dengan ini itu atau apa apakah bagaimana
adalah saya aku kamu ada tidak dalam akan juga oleh sebagai
the of and to in a is for on what how
""".split())


def tokenize(text):
    return [
        t for t in re.findall(r"\b\w+\b", text.lower())
        if t not in STOPWORDS
    ]


class HybridRetriever:

    def __init__(self):

        self.supabase = get_supabase()

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.top_k = 4
        self.candidate_k = 12
        self.max_distance = 0.65
        self.rrf_k = 60

        self.refresh_index()

    def refresh_index(self):
        """Mengambil data evidence_chunks dan membangun indeks BM25."""

        records = []
        start = 0
        batch_size = 500

        while True:
            response = (
                self.supabase
                .table("evidence_chunks")
                .select(
                    "id,source_code,content,page,chunk_index,metadata"
                )
                .order("id")
                .range(start, start + batch_size - 1)
                .execute()
            )

            batch = response.data or []
            records.extend(batch)

            if len(batch) < batch_size:
                break

            start += batch_size

        self.records = [
            r for r in records
            if r.get("id") and r.get("content")
        ]

        self.record_by_id = {
            r["id"]: r for r in self.records
        }

        corpus = [
            tokenize(r["content"])
            for r in self.records
        ]

        self.bm25 = (
            BM25Okapi(corpus)
            if corpus and any(corpus)
            else None
        )

        print(f"BM25 index: {len(self.records)} chunks")

    def vector_search(self, query, where=None):

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        response = self.supabase.rpc(
            "match_evidence_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": self.candidate_k,
                "filter": where or {}
            }
        ).execute()

        rows = response.data or []

        return [
            r for r in rows
            if r.get("distance") is not None
            and float(r["distance"]) <= self.max_distance
        ]

    def hybrid_retrieve(self, query, top_k=None, where=None):

        if not query or not query.strip():
            return []

        top_k = top_k or self.top_k

        if not self.records:
            return []

        # 1. BM25
        bm25_ids = []

        if self.bm25 is not None:

            scores = self.bm25.get_scores(tokenize(query))

            ranked = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            for i in ranked:

                if scores[i] <= 0:
                    continue

                bm25_ids.append(self.records[i]["id"])

                if len(bm25_ids) >= self.candidate_k:
                    break

        # 2. Vector search
        vector_rows = self.vector_search(query, where)

        vector_ids = [r["id"] for r in vector_rows]

        distances = {
            r["id"]: float(r["distance"])
            for r in vector_rows
        }

        # 3. RRF
        rrf = {}

        for rank, doc_id in enumerate(bm25_ids, 1):
            rrf[doc_id] = (
                rrf.get(doc_id, 0)
                + 1 / (self.rrf_k + rank)
            )

        for rank, doc_id in enumerate(vector_ids, 1):
            rrf[doc_id] = (
                rrf.get(doc_id, 0)
                + 1 / (self.rrf_k + rank)
            )

        # 4. Gabungkan hasil
        results = []

        for doc_id in sorted(
            rrf,
            key=rrf.get,
            reverse=True
        )[:top_k]:

            row = self.record_by_id.get(doc_id)

            if not row:
                continue

            metadata = row.get("metadata") or {}

            metadata = dict(metadata)
            metadata.setdefault(
                "source_code",
                row.get("source_code")
            )

            source = (
                metadata.get("title")
                or metadata.get("source_title")
                or row.get("source_code")
                or "Sumber tidak diketahui"
            )

            results.append({
                "id": doc_id,
                "text": row["content"],
                "source": source,
                "metadata": metadata,
                "page": row.get("page"),
                "source_code": row.get("source_code"),
                "rrf_score": rrf[doc_id],
                "rank_bm25": (
                    bm25_ids.index(doc_id) + 1
                    if doc_id in bm25_ids else None
                ),
                "rank_vector": (
                    vector_ids.index(doc_id) + 1
                    if doc_id in vector_ids else None
                ),
                "distance": distances.get(doc_id)
            })

        return results