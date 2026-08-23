# Agent vs baseline RAG — evaluation harness results

All metrics on this page are **deterministic**: they come from set comparison against hand-labelled gold chunks, from a prefix match on a fixed refusal sentence, and from counters recorded in the agent trajectory. No LLM judge is involved, so re-running produces identical numbers. LLM-judged metrics (Ragas) are reported separately.

> **`Citation indices valid` is only defined for single-retrieval answers.** A citation `[n]` can be checked against the retrieved set only when there was one retrieval to number against. When the agent calls a tool more than once each result block restarts at `[1]`, while the scored set is the deduplicated union of all calls, so the check would pass vacuously. Those questions are scored `None` and excluded from this row — meaning the row has a smaller denominator for the agent column than for the baseline, and the two are not directly comparable.

## Overall

| metric | `baseline` | `agent` |
|---|---|---|
| Context recall (labelled chunks) | 93.8% (16/20) | 100.0% (16/20) |
| Full-recall rate | 93.8% (16/20) | 100.0% (16/20) |
| Context precision (labelled) | 23.9% (16/20) | 23.7% (16/20) |
| MRR | 0.77 (16/20) | 0.76 (16/20) |
| Refusal accuracy (all 20) | 95.0% | 100.0% |
| False-refusal rate (in-scope) | 6.2% | 0.0% |
| Citation present | 100.0% (15/20) | 100.0% (16/20) |
| Citation indices valid | 100.0% (15/20) | 100.0% (9/20) |
| Mean tool calls / question | 1.00 | 1.35 |
| Mean LLM calls / question | 1.00 | 2.35 |
| Mean context chars | 7,752 | 6,641 |
| Hit step cap | 0.0% | 0.0% |
| Generation errors | 0.0% | 0.0% |

## Single-hop questions (n=8)

| metric | `baseline` | `agent` |
|---|---|---|
| Context recall (labelled chunks) | 100.0% | 100.0% |
| Full-recall rate | 100.0% | 100.0% |
| Context precision (labelled) | 16.6% | 17.0% |
| MRR | 0.88 | 0.83 |
| Refusal accuracy (all 20) | 100.0% | 100.0% |
| False-refusal rate (in-scope) | 0.0% | 0.0% |
| Citation present | 100.0% | 100.0% |
| Citation indices valid | 100.0% | 100.0% (6/8) |
| Mean tool calls / question | 1.00 | 1.25 |
| Mean LLM calls / question | 1.00 | 2.25 |
| Mean context chars | 7,342 | 6,158 |
| Hit step cap | 0.0% | 0.0% |
| Generation errors | 0.0% | 0.0% |

## Multi-hop questions (n=8)

| metric | `baseline` | `agent` |
|---|---|---|
| Context recall (labelled chunks) | 87.5% | 100.0% |
| Full-recall rate | 87.5% | 100.0% |
| Context precision (labelled) | 31.2% | 30.3% |
| MRR | 0.67 | 0.68 |
| Refusal accuracy (all 20) | 87.5% | 100.0% |
| False-refusal rate (in-scope) | 12.5% | 0.0% |
| Citation present | 100.0% (7/8) | 100.0% |
| Citation indices valid | 100.0% (7/8) | 100.0% (3/8) |
| Mean tool calls / question | 1.00 | 1.75 |
| Mean LLM calls / question | 1.00 | 2.75 |
| Mean context chars | 7,316 | 8,334 |
| Hit step cap | 0.0% | 0.0% |
| Generation errors | 0.0% | 0.0% |

## Out-of-scope questions (n=4)

| metric | `baseline` | `agent` |
|---|---|---|
| Refusal accuracy (all 20) | 100.0% | 100.0% |
| Mean tool calls / question | 1.00 | 0.75 |
| Mean LLM calls / question | 1.00 | 1.75 |
| Mean context chars | 9,444 | 4,219 |
| Hit step cap | 0.0% | 0.0% |
| Generation errors | 0.0% | 0.0% |

## Questions on amended articles (n=3)

| metric | `baseline` | `agent` |
|---|---|---|
| Context recall (labelled chunks) | 66.7% | 100.0% |
| Full-recall rate | 66.7% | 100.0% |
| Context precision (labelled) | 29.5% | 38.9% |
| MRR | 0.50 | 0.48 |
| Refusal accuracy (all 20) | 66.7% | 100.0% |
| False-refusal rate (in-scope) | 33.3% | 0.0% |
| Citation present | 100.0% (2/3) | 100.0% |
| Citation indices valid | 100.0% (2/3) | 100.0% (2/3) |
| Mean tool calls / question | 1.00 | 1.33 |
| Mean LLM calls / question | 1.00 | 2.33 |
| Mean context chars | 6,214 | 5,996 |
| Hit step cap | 0.0% | 0.0% |
| Generation errors | 0.0% | 0.0% |

## How to read these numbers

The in-scope benchmark has **16 questions**, so one question is worth **6.2%**. Any gap smaller than that is a single question changing its mind, not a real difference — and even a one-question gap is within the noise of a non-deterministic agent. Differences are only worth discussing when they are several questions wide, and the honest framing for anything smaller is "indistinguishable on this benchmark".
