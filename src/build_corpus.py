"""Build the retrieval corpus from REAL project tables (no invented values).
One 'card' per protein with a PlasticDB homology hit (annotation-based, not experimental), plus
a few MAG-level / method cards. Usage: python src/build_corpus.py [PROJECT_ROOT]"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "metagenomics_project/mag_reconstruction"
PUB = ROOT / "publication/data"
HERE = Path(__file__).resolve().parent.parent
D = HERE / "data"; D.mkdir(exist_ok=True)

hits = pd.read_csv(PUB / "plasticdb/plasticdb_best_hits.tsv", sep="\t")
ipr = pd.read_csv(PUB / "plasticdb/interpro_summary.tsv", sep="\t").set_index("protein")
ann = pd.read_csv(PUB / "annotation_summary.tsv", sep="\t")

def eggnog(path):
    rows = {}
    for line in open(path):
        if line.startswith("#"): continue
        f = line.rstrip("\n").split("\t")
        rows[f[0]] = dict(description=f[7], name=f[8], ec=f[10], ko=f[11], pathway=f[12], cog=f[6], pfams=f[20])
    return rows
EGG = {"bin.3": eggnog(ROOT / "bin3_eggnog.emapper.annotations"),
       "maxbin.002": eggnog(ROOT / "maxbin002_eggnog.emapper.annotations")}
clean = lambda v: "not annotated" if v in ("", "-", None) else v

docs, table = [], []
for r in hits.itertuples():
    e = EGG[r.MAG].get(r.qseqid, {})
    tier = r.Tier.split(" - ")[0]
    rec = dict(id=r.qseqid, mag=r.MAG, tier=tier, enzyme=r.db_enzyme, polymer=r.db_polymer,
               organism=r.db_organism, accession=r.db_accession, pident=float(r.pident),
               qcov=float(r.qcovhsp), evalue=float(r.evalue), pfam="", interpro="")
    if r.qseqid in ipr.index:
        rec["pfam"] = str(ipr.loc[r.qseqid, "pfam"]); rec["interpro"] = str(ipr.loc[r.qseqid, "interpro"])
    table.append(rec)
    text = (f"Protein {r.qseqid} from MAG {r.MAG}. PlasticDB best homology hit: {r.db_enzyme} "
            f"(polymer: {r.db_polymer}) from {r.db_organism}, accession {r.db_accession}; "
            f"{r.pident:.1f}% identity, {r.qcovhsp:.1f}% query coverage, e-value {r.evalue:.2g}; "
            f"confidence {r.Tier}. eggNOG annotation: {clean(e.get('description'))}; "
            f"preferred name {clean(e.get('name'))}; KEGG KO {clean(e.get('ko'))}; COG category {clean(e.get('cog'))}; "
            f"EC {clean(e.get('ec'))}; Pfam {clean(e.get('pfams'))}.")
    if rec["pfam"]:
        text += f" InterProScan: Pfam {rec['pfam']}; InterPro {rec['interpro']}."
    docs.append(dict(id=r.qseqid, kind="protein", mag=r.MAG, text=text))

# MAG- and method-level cards (values taken from the project handoff/manuscript tables)
a = ann.set_index("MAG")
docs += [
 dict(id="MAG:bin.3", kind="mag", mag="bin.3", text=(
  "MAG bin.3: Desulfobacterota, GTDB-Tk family-level placement JAENJC01 (genus and species not resolved). "
  "100% complete, 1.01% contamination, 3.36 Mb, 111.83x coverage. "
  f"{int(a.loc['bin.3','Proteins_eggNOG_annotated'])} eggNOG-annotated proteins, {int(a.loc['bin.3','Unique_KOs'])} unique KOs, "
  f"{int(a.loc['bin.3','KEGG_pathways_hit'])} KEGG pathways.")),
 dict(id="MAG:maxbin.002", kind="mag", mag="maxbin.002", text=(
  "MAG maxbin.002: Chloroflexota, class Anaerolineae, GTDB-Tk placement UBA4142 (genus and species not resolved). "
  "86.75% complete, 7.64% contamination, 5.25 Mb, 36.65x coverage. "
  f"{int(a.loc['maxbin.002','Proteins_eggNOG_annotated'])} eggNOG-annotated proteins, {int(a.loc['maxbin.002','Unique_KOs'])} unique KOs, "
  f"{int(a.loc['maxbin.002','KEGG_pathways_hit'])} KEGG pathways.")),
 dict(id="METHOD:plasticdb_tiers", kind="method", mag="", text=(
  "PlasticDB screening method: MAG Prokka proteins were searched against PlasticDB (329 curated proteins) with DIAMOND blastp "
  "(very-sensitive, e-value 1e-5). Tier 1 strong: at least 40% identity, 70% coverage, e-value 1e-20 or lower. "
  "Tier 2 moderate: at least 30% identity, 50% coverage, e-value 1e-10 or lower. Tier 3 remote: weaker but e-value 1e-5 or lower. "
  "All hits are annotation-based putative candidates; none are experimentally validated.")),
]
(D / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
pd.DataFrame(table).to_csv(D / "hits_table.csv", index=False)
print(f"{len(docs)} documents ({len(table)} protein cards) -> data/")
