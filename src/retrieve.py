"""Lexical + TF-IDF + hybrid (reciprocal-rank-fusion) retrieval. No model downloads required."""
import json, math, re
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parent.parent / "data"
TOK = re.compile(r"[a-z0-9_.\-]+")
def tokens(t):
    out = []
    for w in TOK.findall(t.lower()):
        out.append(w)
        if re.search(r"[_.\-]", w): out += [p for p in re.split(r"[_.\-]", w) if p]
    return out

def load_docs(): return [json.loads(l) for l in open(DATA / "corpus.jsonl")]

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [Counter(tokens(d["text"])) for d in docs]; self.len = np.array([sum(c.values()) for c in self.toks])
        self.avg = self.len.mean(); self.k1, self.b = k1, b; N = len(docs)
        df = Counter(w for c in self.toks for w in c); self.idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
    def scores(self, q):
        s = np.zeros(len(self.toks))
        for w in set(tokens(q)):
            if w not in self.idf: continue
            for i, c in enumerate(self.toks):
                f = c.get(w, 0)
                if f: s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

class Retriever:
    def __init__(self, docs=None):
        self.docs = docs or load_docs(); texts = [d["text"] for d in self.docs]
        self.bm25 = BM25(self.docs)
        self.tfidf = TfidfVectorizer(tokenizer=tokens, lowercase=False, ngram_range=(1, 2), sublinear_tf=True, token_pattern=None)
        self.M = self.tfidf.fit_transform(texts)
        self.vocab = set(self.bm25.idf)
        self.dense = None
        try:  # optional dense embeddings (pip install sentence-transformers)
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.E = self.st.encode(texts, normalize_embeddings=True, show_progress_bar=False); self.dense = True
        except Exception:
            self.dense = None
    def _rank(self, s): return list(np.argsort(-s))
    def search(self, q, k=5, mode="hybrid"):
        sb = self.bm25.scores(q); st = (self.M @ self.tfidf.transform([q]).T).toarray().ravel()
        sd = (self.E @ self.st.encode([q], normalize_embeddings=True)[0]) if self.dense else None
        if mode == "dense": s = sd
        elif mode == "hybrid_dense":
            s = np.zeros(len(self.docs))
            for sc in (sb, st, sd):
                for r, i in enumerate(self._rank(sc)): s[i] += 1 / (60 + r)
        elif mode == "bm25": s = sb
        elif mode == "tfidf": s = st
        else:
            s = np.zeros(len(self.docs))
            for sc in (sb, st):
                for r, i in enumerate(self._rank(sc)): s[i] += 1 / (60 + r)
        idx = self._rank(s)[:k]
        return [(self.docs[i], float(s[i]), float(sb[i])) for i in idx]
