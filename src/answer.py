"""Grounded answerer: guardrails -> tool routing -> retrieval -> cited extractive answer.
Optional LLM rewrite (--llm openai|ollama) only rephrases the retrieved evidence; default is extractive (no API needed)."""
import re
import pandas as pd
from retrieve import Retriever
import tools

NA = "Not available in current project outputs."
ID_RE = re.compile(r"\b[A-Z]{8}_\d{5}\b")
UNSUPPORTED = re.compile(r"experimentally|validated|confirmed|proven|kinetic|activity assay|temperature optim|ph optim|crystal structure|expression level", re.I)
COUNT_RE = re.compile(r"how many|number of|count of|total (?:number )?of", re.I)
LIST_RE = re.compile(r"\b(?:which|list|show|what)\b.*\b(?:proteins?|candidates?|hits?)\b", re.I)
BY_ID = tools.T.set_index("id")

def card(i):
    r = BY_ID.loc[i]
    s = (f"[{i}] ({r.mag}) putative {r.enzyme} homolog (polymer: {r.polymer}), {r.pident:.1f}% identity / "
         f"{r.qcov:.0f}% coverage to {r.organism}{"" if str(r.accession) == "nan" else " " + str(r.accession)}; {r.tier}.")
    if isinstance(r.pfam, str) and r.pfam: s += f" Pfam: {r.pfam}."
    return s

def answer(q, R: Retriever, k=3, min_bm25=2.0):
    """Returns dict(route, text, ids)"""
    if UNSUPPORTED.search(q):
        return dict(route="refuse", ids=[], text=f"{NA} This corpus is annotation- and homology-based; it contains no experimental validation, kinetics or expression data.")
    known = {d["id"] for d in R.docs}
    unknown = [x for x in ID_RE.findall(q) if x not in known]
    if unknown:
        return dict(route="refuse", ids=[], text=f"{NA} {unknown[0]} has no PlasticDB hit in the indexed outputs.")
    f = tools.parse_filters(q)
    if COUNT_RE.search(q) and f:
        n, _ = tools.count(f)
        return dict(route="tool_count", ids=[], text=f"{n} proteins match {f} (PlasticDB best-hit table; putative, annotation-based).", n=n)
    if LIST_RE.search(q) and f:
        n, d = tools.count(f)
        ids = list(d.sort_values("evalue").id)
        return dict(route="tool_list", ids=ids, text=f"{n} proteins match {f}: " + "; ".join(card(i) for i in ids[:5]) + (" ..." if n > 5 else ""))
    hits = R.search(q, k=k)
    if not hits or hits[0][2] < min_bm25:
        return dict(route="refuse", ids=[], text=f"{NA} No indexed document matches this question well.")
    ids = [d["id"] for d, _, _ in hits if d["kind"] == "protein"]
    ctx = [d["text"] if d["kind"] != "protein" else card(d["id"]) for d, _, _ in hits]
    return dict(route="retrieve", ids=ids, text=" ".join(ctx))
