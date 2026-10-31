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
- **No API key or model download needed.** An optional LLM rewrite step could be added on top of the retrieved evidence, but it is not implemented or evaluated here.

## Usage
```bash
pip install -r requirements.txt
python src/build_corpus.py /path/to/mag_reconstruction   # builds data/corpus.jsonl from the project tables
python src/ask.py "What is ADACJJMH_01622 annotated as?"
python src/ask.py "How many Tier 1 PlasticDB hits does maxbin.002 have?"
python src/eval.py                                       # writes outputs/eval_results.csv
```

## Evaluation
Questions are generated from the tables with computed gold answers (seeded, `src/eval.py`). 91 questions in total.

| Task | n | System | Hit@1 | Hit@3 |
|---|---|---|---|---|
| Protein ID lookup | 30 | BM25 / TF-IDF / hybrid | 1.00 | 1.00 |
| Description search | 20 | BM25 | 0.75 | 0.95 |
| Description search | 20 | TF-IDF | 0.80 | 1.00 |
| Description search | 20 | hybrid | 0.80 | 0.95 |
| Pfam domain search | 11 | BM25 / hybrid | 0.27 | 0.55 |
| Pfam domain search | 11 | TF-IDF | 0.27 | 0.73 |
| Count questions (tool) | 6 | full pipeline | 1.00 | |
| List questions (tool) | 14 | full pipeline | 1.00 | |
| Unanswerable (refuse) | 10 | full pipeline | 1.00 | |

## Limitations (please read)
- **Small, templated evaluation.** Questions are generated from templates over one dataset; sample sizes are small (domain search n = 11). ID lookup is easy for lexical retrieval by construction.
- **Tool and refusal scores are not retrieval quality.** Count and list results test routing and parsing against exact tables; the refusal set uses rules I wrote alongside the test questions, so 10/10 is optimistic.
- **Pfam domain search is the weak spot**: lexical retrieval ranks the right cards in the top 3 only about half to three quarters of the time. Dense embeddings would be the obvious next step; they were not run here.
- Corpus covers only proteins with a PlasticDB hit and the two MAGs; it cannot answer questions about the rest of the genome.
- Underlying biology is annotation-based: homology hits are putative candidates, not evidence of enzymatic activity.
