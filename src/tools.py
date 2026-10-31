"""Structured tool: exact counts/filters over the hits table. RAG alone cannot count reliably,
so aggregate questions are routed here (agentic tool-use)."""
import re
import pandas as pd
from pathlib import Path
T = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "hits_table.csv")

def parse_filters(q):
    ql = q.lower(); f = {}
    m = re.search(r"tier\s*([123])", ql)
    if m: f["tier"] = f"Tier {m.group(1)}"
    for mag in ("maxbin.002", "bin.3"):
        if mag in ql: f["mag"] = mag
    for pol in sorted(T.polymer.unique(), key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(pol.lower())}(?![a-z0-9])", ql): f["polymer"] = pol; break
    return f

def count(f):
    d = T
    for k, v in f.items(): d = d[d[k] == v]
    return len(d), d
