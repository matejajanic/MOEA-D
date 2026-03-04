from __future__ import annotations

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from problems.text_summarization import TextSummarization, split_sentences


DEFAULT_TEXT = (
    """
    In the last quarter, the search team rolled out a new retrieval pipeline aimed at reducing latency without hurting answer quality. 
The previous stack used a single dense retriever followed by a heavy cross-encoder reranker. While accuracy was strong on head queries, 
tail queries suffered because candidates were often missing from the first-stage pool. The new system introduces a hybrid retriever that 
combines sparse BM25 with a lightweight embedding model. Candidates from both sources are merged and deduplicated before reranking.

To keep compute costs bounded, the reranker was replaced with a smaller model and a two-stage scoring strategy. The first stage uses a 
fast bi-encoder similarity score and filters down to a short list. The second stage applies a cross-attention model only to the top items. 
In offline evaluation, this reduced average reranking time by 38% while maintaining nearly the same NDCG@10. However, on a subset of 
long-form queries, the new reranker was less consistent, especially when the query included multiple constraints.

During the first production week, customer reports suggested that answers felt “faster but occasionally less grounded.” Telemetry confirmed 
a shift: average latency dropped from 820ms to 520ms, but citation click-through decreased slightly. Investigation showed the hybrid 
retriever sometimes promoted near-duplicate documents. This increased redundancy in the context window and caused the generator to repeat 
similar facts, reducing perceived usefulness.

A mitigation was deployed: deduplication was tightened using cosine similarity thresholds and paragraph-level hashing. Additionally, the 
candidate merge step started enforcing diversity by source domain. After these changes, citation click-through recovered, and hallucination 
flags decreased modestly. Still, an open question remains: whether we should optimize for speed first and rely on post-filters, or allocate 
more budget to retrieval quality to reduce downstream risk.

Next quarter, the team plans to test a query router that predicts whether a query needs deep retrieval and heavy reranking. The router will 
use features such as query length, entity density, and user intent signals. If successful, easy queries will take the fast path, while 
complex queries will use the expensive path. The goal is to keep the median latency low while improving worst-case groundedness."""
)


def get_textsum_problem(
    text_path: str | None = None,
    seed: int = 42,
    max_sentences: int = 40,
    tfidf_max_features: int = 5000,
    max_comp: float | None = None,
) -> dict:
    if text_path is None:
        text = DEFAULT_TEXT
    else:
        text = Path(text_path).read_text(encoding="utf-8")

    sents = split_sentences(text)
    if len(sents) == 0:
        raise ValueError("Text contains no sentences after splitting.")

    sents = sents[:max_sentences]

    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        lowercase=True,
        stop_words="english",
    )

    tfidf = vectorizer.fit_transform(sents + [text])
    tfidf = tfidf.toarray().astype(float)

    sent_vecs = tfidf[:-1]
    doc_vec = tfidf[-1]

    problem = TextSummarization(
        sentences=sents,
        sent_vecs=sent_vecs,
        doc_vec=doc_vec,
        max_comp=max_comp,
        penalty_scale=5.0,
    )

    return {
        "n_obj": 3,
        "n_var": problem.n_sent,
        "evaluate_fn": problem.evaluate,
        "analyze_fn": problem.analyze,
        "build_summary_fn": problem.build_summary,
        "n_sent": problem.n_sent,
        "text_path": text_path,
        "max_comp": max_comp,
        "max_k": problem.max_k,
    }