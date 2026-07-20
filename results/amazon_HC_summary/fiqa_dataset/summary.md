FIQA:

======================================================================

END-TO-END RAG RESULTS

======================================================================

Method Avg K | Token F1 ROUGE-L | P@1 NDCG Ret Pre Ret Rec | Ctx Eff F1/doc

---

Baseline k=5 5.0 | 0.265 0.175 | 0.387 0.362 0.171 0.373 | 0.409 0.0531

HC γ=0.1 3.7 | 0.250 0.173 | 0.387 0.400 0.319 0.327 | 1.341 0.0671

HC γ=0.1 min_k=2 4.1 | 0.256 0.173 | 0.387 0.375 0.257 0.350 | 0.777 0.0621

HC γ=0.1 min_k=3 4.6 | 0.263 0.175 | 0.387 0.371 0.219 0.373 | 0.563 0.0568

HC γ=0.2 6.9 | 0.261 0.175 | 0.387 0.412 0.292 0.375 | 1.104 0.0380

HC γ=0.2 min_k=2 7.2 | 0.264 0.175 | 0.387 0.388 0.237 0.395 | 0.661 0.0368

HC γ=0.2 min_k=3 7.6 | 0.268 0.177 | 0.387 0.385 0.204 0.417 | 0.484 0.0355

Failure explanation:

HC is designed for the case where there's a detectable boundary in the similarity distribution between relevant and irrelevant documents. On FiQA, that boundary doesn't exist cleanly — relevant docs are scattered and interleaved with irrelevant ones at similar similarity scores. A fixed k=5 works better simply because it brute-forces past the overlap zone and picks up more relevant docs by accident.

Num queries- 648

Num docs- 57638
