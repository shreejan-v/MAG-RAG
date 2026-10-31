# MAG-RAG: grounded question answering over metagenome annotation results

A small retrieval-augmented question-answering system over real results from a genome-resolved metagenomics study (two MAGs, bin.3 and maxbin.002; PlasticDB homology screening, eggNOG-mapper and InterProScan annotations). Answers are cited to protein IDs, aggregate questions are routed to an exact table tool instead of being guessed, and anything the data cannot support is refused rather than invented.

Built as a bioinformatics + applied-AI portfolio project. The RAG pattern (index, retrieve, answer with sources) follows the common learning path in [techwithprateek/agenticai-projects](https://github.com/techwithprateek/agenticai-projects); the code here is original and domain-specific.

## What it does
```
question -> guardrails (unsupported claims, unknown IDs) -> router
          -> count / list questions: exact pandas tool
          -> everything else: hybrid retrieval (BM25 + TF-IDF, rank fusion) -> cited answer
          -> weak match: "Not available in current project outputs."
```
- **Corpus (79 documents):** one card per PlasticDB-hit protein (76), two MAG cards, one method card. Every value comes from the project's own tables; nothing is generated.
- **Wording is deliberately cautious:** all hits are "putative", annotation-based candidates, never experimentally validated.
- **Runs without an API key.** Dense embeddings (sentence-transformers) are optional; without them it falls back to lexical retrieval. An optional LLM rewrite step could be added on top of the retrieved evidence, but it is not implemented or evaluated here.

## Usage
```bash
pip install -r requirements.txt      # add sentence-transformers for dense retrieval
python src/build_corpus.py /path/to/mag_reconstruction   # builds data/corpus.jsonl from the project tables
python src/ask.py "What is ADACJJMH_01622 annotated as?"
python src/ask.py "How many Tier 1 PlasticDB hits does maxbin.002 have?"
python src/eval.py                                       # writes outputs/eval_results.csv
```

## Evaluation
Questions are generated from the tables with computed gold answers (seeded, `src/eval.py`); 91 questions in total. Retrieval systems compared on the three search tasks (Hit@1 / Hit@3):

| System | ID lookup (n=30) | Description search (n=20) | Pfam domain search (n=11) |
|---|---|---|---|
| BM25 | 1.00 / 1.00 | 0.75 / 0.95 | 0.27 / 0.55 |
| TF-IDF | 1.00 / 1.00 | 0.80 / 1.00 | 0.27 / 0.73 |
| Hybrid (BM25+TF-IDF) | 1.00 / 1.00 | 0.80 / 0.95 | 0.27 / 0.55 |
| Dense only (MiniLM-L6-v2) | 0.03 / 0.07 | 0.45 / 0.70 | 0.18 / 0.27 |
| Hybrid + dense | 0.23 / 0.63 | 0.65 / 1.00 | 0.64 / 0.73 |
| **Full pipeline** (dense only for non-ID queries) | 1.00 / 1.00 | 0.65 / 1.00 | 0.55 / 0.55 |

Other tasks (full pipeline): count questions 6/6, list questions 14/14, unanswerable questions refused 10/10.

**Takeaways:** dense embeddings alone fail on exact identifiers (3% hit@1 on IDs) because MiniLM does not understand locus tags; fusing them with lexical search helps the domain-search task (0.27 to 0.64 hit@1) but hurts ID lookup, so the pipeline routes ID questions to lexical-only retrieval. On description search the lexical hybrid has the best hit@1 (0.80 vs 0.65), so the routed pipeline trades top-1 precision for recall at 3 there.

## Limitations (please read)
- **Small, templated evaluation.** Questions are generated from templates over one dataset; sample sizes are small (domain search n = 11). ID lookup is easy for lexical retrieval by construction.
- **Tool and refusal scores are not retrieval quality.** Count and list results test routing and parsing against exact tables; the refusal set uses rules I wrote alongside the test questions, so 10/10 is optimistic.
- **Pfam domain search is still the weak spot** (best hit@3 is 0.73). Domain search has only 11 questions, so differences between systems are not statistically reliable.
- Corpus covers only proteins with a PlasticDB hit and the two MAGs; it cannot answer questions about the rest of the genome.
- Underlying biology is annotation-based: homology hits are putative candidates, not evidence of enzymatic activity.
