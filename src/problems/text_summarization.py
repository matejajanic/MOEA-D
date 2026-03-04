from __future__ import annotations
import re
import numpy as np


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))


class TextSummarization:
    """
    Binary sentence selection: x in {0,1}^n_sent

    Objectives (min):
      f1 = 1 - cosine_sim(v_sum, v_doc)          (maximize coverage)
      f2 = k / n_sent                             (minimize length)
      f3 = redundancy = avg cosine_sim(si, sj)    (minimize redundancy)

    Optional constraint:
      max_comp in (0,1] => max_k = floor(max_comp * n_sent), at least 1.
      If k > max_k -> apply soft penalty to all objectives.
    """

    def __init__(
        self,
        sentences: list[str],
        sent_vecs: np.ndarray,
        doc_vec: np.ndarray,
        max_comp: float | None = None,
        penalty_scale: float = 5.0,
    ):
        if len(sentences) == 0:
            raise ValueError("No sentences provided.")
        sent_vecs = np.asarray(sent_vecs, dtype=float)
        doc_vec = np.asarray(doc_vec, dtype=float)

        if sent_vecs.ndim != 2:
            raise ValueError("sent_vecs must be 2D (n_sent, d).")
        if doc_vec.ndim != 1:
            raise ValueError("doc_vec must be 1D (d,).")
        if sent_vecs.shape[0] != len(sentences):
            raise ValueError("sent_vecs rows must match number of sentences.")
        if sent_vecs.shape[1] != doc_vec.shape[0]:
            raise ValueError("Vector dimension mismatch.")

        if max_comp is not None and not (0.0 < max_comp <= 1.0):
            raise ValueError("max_comp must be in (0,1].")

        self.sentences = sentences
        self.sent_vecs = sent_vecs
        self.doc_vec = doc_vec
        self.n_sent = len(sentences)

        self.max_comp = max_comp
        self.max_k = None
        if max_comp is not None:
            self.max_k = max(1, int(np.floor(max_comp * self.n_sent)))

        self.penalty_scale = float(penalty_scale)

    def _redundancy_avg_pairwise(self, idx: np.ndarray) -> float:
        if idx.size < 2:
            return 0.0

        V = self.sent_vecs[idx]  # (k, d)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        U = V / norms
        S = U @ U.T  # cosine similarity matrix
        k = idx.size
        triu = np.triu_indices(k, k=1)
        return float(np.mean(S[triu]))

    def _length_penalty(self, k: int) -> float:
        """
        0 if within constraint, else proportional to overflow.
        """
        if self.max_k is None:
            return 0.0
        if k <= self.max_k:
            return 0.0
        overflow = (k - self.max_k) / float(self.n_sent)  # typically in (0,1]
        return overflow

    def evaluate(self, X_bin: np.ndarray) -> np.ndarray:
        X_bin = np.asarray(X_bin)
        if X_bin.ndim != 2 or X_bin.shape[1] != self.n_sent:
            raise ValueError(f"X_bin must be (N, {self.n_sent}).")

        F = np.empty((X_bin.shape[0], 3), dtype=float)

        for i, x in enumerate(X_bin):
            k = int(np.sum(x))
            if k == 0:
                F[i, 0] = 1.0
                F[i, 1] = 1.0
                F[i, 2] = 1.0
                continue

            idx = np.where(x == 1)[0]
            v_sum = np.mean(self.sent_vecs[idx], axis=0)
            sim = cosine_sim(v_sum, self.doc_vec)

            f1 = 1.0 - sim
            f2 = k / self.n_sent
            f3 = self._redundancy_avg_pairwise(idx)

            pen = self._length_penalty(k)
            if pen > 0.0:
                bump = self.penalty_scale * pen
                f1 += bump
                f2 += bump
                f3 += bump

            F[i, 0] = f1
            F[i, 1] = f2
            F[i, 2] = f3

        return F

    def analyze(self, X_bin: np.ndarray) -> dict[str, np.ndarray]:
        X_bin = np.asarray(X_bin)
        k = np.sum(X_bin, axis=1).astype(int)

        sim = np.zeros(X_bin.shape[0], dtype=float)
        red = np.zeros(X_bin.shape[0], dtype=float)

        for i, x in enumerate(X_bin):
            if k[i] == 0:
                sim[i] = 0.0
                red[i] = 1.0
                continue
            idx = np.where(x == 1)[0]
            v_sum = np.mean(self.sent_vecs[idx], axis=0)
            sim[i] = cosine_sim(v_sum, self.doc_vec)
            red[i] = self._redundancy_avg_pairwise(idx)

        compression = k / float(self.n_sent)
        return {"k": k, "sim": sim, "compression": compression, "redundancy": red}

    def build_summary(self, x: np.ndarray, max_sentences: int | None = None) -> str:
        x = np.asarray(x).astype(int)
        idx = np.where(x == 1)[0]
        if idx.size == 0:
            return ""
        if max_sentences is not None:
            idx = idx[:max_sentences]
        return " ".join(self.sentences[i] for i in idx)