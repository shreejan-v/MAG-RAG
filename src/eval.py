"""Deterministic evaluation set built from the tables (gold answers computed, not hand-typed).
Retrieval tasks compare BM25 / TF-IDF / hybrid; tool tasks check exact answers; unanswerable tasks check abstention."""
import random, itertools, re
import numpy as np, pandas as pd
from pathlib import Path
from retrieve import Retriever
from answer import answer, NA
import tools

random.seed(7); R = Retriever(); T = tools.T; OUT = Path(__file__).resolve().parent.parent / "outputs"; OUT.mkdir(exist_ok=True)
egg = {d["id"]: d["text"] for d in R.docs if d["kind"] == "protein"}
def field(i, key):
    m = re.search(key + r" ([^;]+);", egg[i]); return m.group(1).strip() if m else ""

Q = []  # (task, query, gold_ids/answer)
ids = list(T.id); random.shuffle(ids)
for i in ids[:30]:
    Q.append(("id_lookup", random.choice([f"What is {i} annotated as?", f"Tell me about protein {i}", f"{i} PlasticDB hit and polymer"]), {i}))
descs = {}
for i in T.id:
    m = re.search(r"eggNOG annotation: ([^;]+);", egg[i]); d = m.group(1).strip() if m else ""
    if d and d != "not annotated": descs.setdefault(d, set()).add(i)
for d, g in random.sample(sorted(descs.items()), min(20, len(descs))):
    Q.append(("description_search", random.choice([f"Which protein is annotated as {d}?", f"candidate with eggNOG description {d}"]), g))
pf = {}
for r in T.itertuples():
    for m in re.findall(r"PF\d{5}", (r.pfam if isinstance(r.pfam, str) else "")): pf.setdefault(m, set()).add(r.id)
for p, g in random.sample(sorted(pf.items()), min(15, len(pf))):
    Q.append(("domain_search", random.choice([f"Which candidates contain Pfam domain {p}?", f"proteins with {p}"]), g))
tool_gold = []
for mag, tier in itertools.product(["bin.3", "maxbin.002"], ["Tier 1", "Tier 2", "Tier 3"]):
    Q.append(("count_tool", f"How many {tier} PlasticDB hits does {mag} have?", len(T[(T.mag == mag) & (T.tier == tier)])))
for pol in [p for p in T.polymer.unique() if (T.polymer == p).sum() >= 2][:8]:
    for mag in ["bin.3", "maxbin.002"]:
        g = set(T[(T.mag == mag) & (T.polymer == pol)].id)
        if g: Q.append(("list_tool", f"Which proteins in {mag} are {pol} candidates?", g))
for q in ["Is the polyamidase in maxbin.002 experimentally validated?", "What is the temperature optimum of the PETase candidate?",
          "Was PET degradation confirmed for any bin.3 protein?", "What is the kinetic constant of ADACJJMH_01622?",
          "Show expression level of the copper oxidase", "Does ZZZZZZZZ_99999 have a PlasticDB hit?", "Tell me about protein QQQQQQQQ_00001",
          "What is the crystal structure of the maxbin.002 polyamidase?", "Is any hit proven to degrade polyethylene?", "What is the pH optimum of the cutinase hit?"]:
    Q.append(("unanswerable", q, None))

rows = []
for task, q, gold in Q:
    if task in ("id_lookup", "description_search", "domain_search"):
        for mode in ("bm25", "tfidf", "hybrid", "dense", "hybrid_dense"):
            res = [d["id"] for d, _, _ in R.search(q, k=5, mode=mode)]
            rank = next((r + 1 for r, x in enumerate(res) if x in gold), None)
            rows.append(dict(task=task, system=mode, hit1=rank == 1, hit3=bool(rank and rank <= 3), rr=1 / rank if rank else 0))
        a = answer(q, R)  # end-to-end with abstention guard
        rows.append(dict(task=task, system="full_pipeline", hit1=bool(a["ids"]) and a["ids"][0] in gold, hit3=any(x in gold for x in a["ids"][:3]),
                         rr=0))
    elif task == "count_tool":
        a = answer(q, R); rows.append(dict(task=task, system="full_pipeline", hit1=a.get("n") == gold, hit3=a.get("n") == gold, rr=0))
    elif task == "list_tool":
        a = answer(q, R); rows.append(dict(task=task, system="full_pipeline", hit1=set(a["ids"]) == gold, hit3=set(a["ids"]) == gold, rr=0))
    else:
        a = answer(q, R); rows.append(dict(task=task, system="full_pipeline", hit1=a["route"] == "refuse", hit3=a["route"] == "refuse", rr=0))
df = pd.DataFrame(rows)
res = df.groupby(["task", "system"]).agg(n=("hit1", "size"), hit1=("hit1", "mean"), hit3=("hit3", "mean"), mrr=("rr", "mean")).round(3).reset_index()
res.to_csv(OUT / "eval_results.csv", index=False)
print(res.to_string(index=False))
