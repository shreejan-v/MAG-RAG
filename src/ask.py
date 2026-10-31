"""Interactive CLI:  python src/ask.py "What is ADACJJMH_01622 annotated as?"  (omit the question for a prompt loop)"""
import sys
from retrieve import Retriever
from answer import answer
R = Retriever()
qs = sys.argv[1:] or iter(lambda: input("question> "), "")
for q in ([" ".join(sys.argv[1:])] if sys.argv[1:] else qs):
    a = answer(q, R); print(f"[{a['route']}] {a['text']}\n")
