import os
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from supabase_configuration.database import get_supabase

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

STOPWORDS = set("""
yang dan di ke dari untuk pada dengan ini itu atau apa apakah bagaimana
adalah saya aku kamu ada tidak dalam akan juga oleh sebagai
the of and to in a is for on what how
""".split())


# Istilah dua kata yang harus dibaca sebagai satu istilah, supaya
# "heat stroke" tidak tercampur dengan "stroke" (stroke otak) di sumber lain.
PHRASES = [
    (re.compile(r"\bheat[\s-]?stroke\b"), " heatstroke "),
    (re.compile(r"\bsengatan panas\b"), " heatstroke "),
    (re.compile(r"\bserangan panas\b"), " heatstroke "),
]


def tokenize(text):
    text = text.lower()
    for pattern, replacement in PHRASES:
        text = pattern.sub(replacement, text)
    return [
        t for t in re.findall(r"\b\w+\b", text)
        if t not in STOPWORDS
    ]

def normalize_peran(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip().lower().replace(" dan ", " & ")
    )

def matches_peran(metadata, peran):
    if not peran:
        return True

    metadata = metadata or {}

    return (
        normalize_peran(metadata.get("peran"))
        == normalize_peran(peran)
    )

def matches_topik(metadata, topics=None):
    if not topics:
        return True

    metadata = metadata or {}
    return metadata.get("topik") in topics

class HybridRetriever:

    def __init__(self):

        self.supabase = get_supabase()
        model_name: str = EMBEDDING_MODEL_NAME

        # Di server bersama (Railway), torch mengira ada banyak CPU dan membuat
        # terlalu banyak thread sehingga pencarian jadi sangat lambat. Batasi.
        import torch
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))

        self.embedding_model = SentenceTransformer(model_name)

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

    def vector_search(self, query, where=None, topics=None):

        # Awalan "query: " (format E5) dipakai untuk semua pencarian,
        # karena chunk disimpan dengan awalan "passage: ".
        query_text = f"query: {query}"

        query_embedding = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True
        ).tolist()

        metadata_filter = {}

        if topics:
            metadata_filter["topik_any"] = topics

        response = self.supabase.rpc(
            "match_evidence_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": self.candidate_k,
                "peran": where or None,
                "filter": metadata_filter
            }
        ).execute()

        rows = response.data or []

        return [
            r for r in rows
            if r.get("distance") is not None
            and float(r["distance"]) <= self.max_distance
        ]

    def hybrid_retrieve(self, query, top_k=None, where=None, topics=None, extra_queries=None):
        import time
        started = time.perf_counter()
        try:
            return self._hybrid_retrieve(query, top_k, where, topics, extra_queries)
        finally:
            print(f"[waktu] pencarian sumber {time.perf_counter() - started:.1f} detik", flush=True)

    def _hybrid_retrieve(self, query, top_k=None, where=None, topics=None, extra_queries=None):

        if not query or not query.strip():
            return []

        top_k = top_k or self.top_k

        if not self.records:
            return []

        # Pertanyaan asli + versi lain (misalnya terjemahan bahasa Inggris).
        # Setiap versi dicari sendiri, lalu semua peringkat digabung dengan RRF.
        queries = [query] + [q for q in (extra_queries or []) if q and q.strip() and q.strip() != query.strip()]

        bm25_lists, vector_lists, distances = [], [], {}
        for q in queries:
            ids, rows = self._rank_one(q, where, topics)
            bm25_lists.append(ids)
            vector_lists.append([r["id"] for r in rows])
            for r in rows:
                d = float(r["distance"])
                distances[r["id"]] = min(d, distances.get(r["id"], d))

        bm25_ids = bm25_lists[0]
        vector_ids = vector_lists[0]

        # 3. RRF
        rrf = {}

        for ranked_ids in bm25_lists + vector_lists:
            for rank, doc_id in enumerate(ranked_ids, 1):
                rrf[doc_id] = (
                    rrf.get(doc_id, 0)
                    + 1 / (self.rrf_k + rank)
                )

        return self._build_results(rrf, top_k, bm25_ids, vector_ids, distances)

    def _rank_one(self, query, where=None, topics=None):
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

                record = self.records[i]

                # Samakan filter kategori dan peran dengan vector search.
                if not matches_topik(record.get("metadata"), topics):
                    continue

                if not matches_peran(record.get("metadata"), where):
                    continue

                bm25_ids.append(record["id"])

                if len(bm25_ids) >= self.candidate_k:
                    break

        # 2. Vector search
        vector_rows = self.vector_search(query, where, topics)

        return bm25_ids, vector_rows

    def _build_results(self, rrf, top_k, bm25_ids, vector_ids, distances):
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