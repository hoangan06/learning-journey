# Project 5 — RAG Agent + Evaluation Harness (Vietnamese FMCG regulations)

A tool-calling RAG agent (LlamaIndex) over five Vietnamese food-safety, customs and
product-labelling instruments, with an **evaluation harness** that measures answer
groundedness, hallucination/refusal behaviour and **retrieval-tuning experiments**
benchmarked against a hand-labelled question set.

The point of this repository is not the agent. It is the **measurement**: what is
actually being measured, what the numbers can and cannot support, and which results
came out negative.

---

## Key results

| | baseline | agent |
|---|---|---|
| Context recall, 16 in-scope questions | 93.8% | **100%** |
| Refusal accuracy, 20 questions (4 out-of-scope) | 95.0% | **100%** |
| False-refusal rate | 6.2% | **0%** |
| Mean tool calls per question | 1.00 | **1.35** |

- Chroma migration verified against the previous project's brute-force retrieval:
  **100% top-5 overlap (25/25), max score delta 7.15 × 10⁻⁷** — run before trusting any table.
- **Negative results, kept and reported:** parent-child expansion buys exactly one question at
  +37% context (§4.2); and **all three amendment-handling mechanisms measured as non-load-bearing**
  — graph traversal changed recall by 0.0 at every `k`, the agent tool was skipped on both
  questions that hinged on an amendment, and removing the in-context notice left the answer
  unchanged (§4.3). Plain vector retrieval was doing the work.
- **A similarity threshold cannot separate out-of-scope questions** — the highest out-of-scope
  score (0.8641) exceeds the lowest in-scope one (0.8569). Refusal is prompt-driven and measured
  on a dedicated out-of-scope set: 4/4 for both systems (§4.4).
- **A rule written as mandatory in the prompt was followed 2 times out of 4**, and only on the
  questions where following it was pointless (§4.6). Prompt rules have compliance rates.
- **Six measurement defects were found and fixed** — none of which raised an exception; each
  returned a plausible number (§4.7).

---

## 1. Problem

A supply-chain team needs to answer questions such as *"what must appear on the label of
an imported infant formula, and does it need self-declaration or registered declaration?"*
against five instruments:

| Reference | Instrument |
|---|---|
| `61/VBHN-VPQH` | Law on Food Safety (consolidated 2025) |
| `54/VBHN-VPQH` | Customs Law (consolidated 2026) |
| `15/2018/NĐ-CP` | Decree implementing the Law on Food Safety |
| `43/2017/NĐ-CP` | Decree on goods labelling |
| `111/2021/NĐ-CP` | Decree **amending** 43/2017 |

That last row is what makes the corpus interesting. Twelve articles of Decree 43/2017 have
been amended or repealed by Decree 111/2021, and the amending text lives in a **separate
document**. A retriever that returns the original article is returning text that is no
longer the law — while looking perfectly grounded to any groundedness metric.

## 2. What was built

```
        Project 4 assets (reused, read-only)
        291 structure-aware chunks  +  precomputed e5 embeddings
                          │
                  rag/build_index.py
                          │  validates invariants, resolves amendment links
                          ▼
                  Chroma (persistent, cosine)
                          │
   ┌──────────────────────┴────────────────────────┐
   │                                                │
rag/pipeline.py                             agent/app.py
baseline RAG                                LlamaIndex FunctionAgent
1 retrieval + 1 LLM call                    tools: search_regulations,
(the control condition)                            check_amendment
   │                                                │
   └──────────────────────┬─────────────────────────┘
                          ▼
                    eval/ harness
   deterministic metrics · Ragas (LLM-judge) · retrieval experiments
```

**Two answering systems, one harness.** The baseline is a fixed pipeline — one retrieval,
one generation — deliberately shaped like the previous project so that the agent comparison
isolates a single variable: *who decides the control flow*.

**Two tools.** `search_regulations` wraps the retriever. `check_amendment` follows the
amendment edges parsed at index time and reports whether a given article is still in force.
It exists for a reason that measurement has since sharpened: retrieval turns out to surface
the amending text perfectly well on its own (§4.3), so the tool is not there to *find* text —
it is there to establish which of two retrieved provisions **supersedes** the other, a
relation that similarity ranking says nothing about. Whether the model actually chose to
use it for that is a separate question, and §4.6 answers it — unfavourably.

## 3. How the evaluation is built

The benchmark is 20 hand-labelled questions: 8 single-hop, 8 multi-hop, 4 out-of-scope.
Each carries a reference answer, the **gold chunk ids** required to answer it, a minimum
step count, and whether the system should refuse.

Metrics are split into two tiers that are **never mixed in a table**, because they do not
deserve equal trust:

**Tier 1 — deterministic.** Context recall/precision and MRR by set comparison against the
gold chunk ids; refusal accuracy by matching the first 40 characters of a fixed refusal sentence the
prompt mandates (a prefix match, so trailing punctuation does not flip the result); citation validity by parsing `[n]` markers; tool-call and step counts from the
agent trajectory. No LLM involved, so re-running gives identical numbers.

**Tier 2 — LLM-as-judge (Ragas 0.4.3).** `eval/ragas_eval.py` wires up faithfulness, answer
relevancy, and LLM/non-LLM context precision and recall. Only **two** were run for this
report — `faithfulness` and `non_llm_context_recall` — because the LLM-judged context
metrics cost one request per context chunk and the free-tier quota is a daily one; the tool
prints a request estimate and refuses to exceed a `--budget`. Reported separately from
Tier 1, always with coverage, and never mixed into the same table.

Two further design points:

- **Outcome vs trajectory.** The harness records not only *what* the agent answered but
  *how it got there* — every tool call, its arguments, and whether it errored. Without the
  trajectory, an agent that makes six poor queries and stumbles onto the right chunk scores
  the same as one that makes a single precise query.
- **Refusal is made machine-checkable by contract.** The system prompt requires a fixed
  sentence when the context is insufficient, so "did it refuse?" is a substring check
  (`REFUSAL_SENTINEL[:40]`, case-folded) rather than another LLM judgement.

### 3.1 The prompt, and why it is not the one from Project 4

Both systems share a single `SYSTEM_PROMPT` in `rag/config.py`; the agent appends tool
instructions to it and changes nothing else. That sharing is a precondition for §4.6 — if the
two systems were prompted differently, the comparison would be measuring the prompt.

It is a rewrite of the Project 4 prompt, not a copy, and the changes are all in one direction:
**making the output checkable by code rather than by reading.**

| | Project 4 | Project 5 |
|---|---|---|
| Grounding constraint | context only, no outside knowledge | same |
| Citation format | *"cite down to the clause, using the `[Law]`/`[Article]` tags"* — free-form prose | exactly `[1]`, `[2]` — **parseable with a regex** |
| Refusal | *"Không tìm thấy thông tin trong tài liệu."* | one fixed sentence, reproduced verbatim, **matched on its first 40 characters** |
| Amendments | prefer the amending text | same, **plus** must state that the provision was amended |
| Partial answers | explicit rule: answer what is supported, name what is missing | **removed** |

The first three rows are why the deterministic tier exists at all: *"did it cite?"* and *"did it
refuse?"* became string operations instead of a second LLM judgement.

The last row is a deliberate trade, and it costs something. Dropping partial answers makes
refusal binary — every question is either answered or refused, which is what makes refusal
accuracy a clean metric — but a question that is answerable *in part* now gets refused whole.
That did not occur in this benchmark (`mh02`, the only false refusal, had **neither** of its gold
chunks retrieved, so refusing was the correct behaviour given the evidence). It is a known cost
of the design, not a measured problem, and it is stated here rather than left for a reader to
find.

## 4. Results

### 4.1 Migration parity — the precondition for trusting anything else

The previous project retrieved by brute-force dot product over a normalised matrix. This
one uses Chroma. Before any comparison is meaningful, the two must agree:

| Check | Result |
|---|---|
| Stored documents match the text that was embedded | **pass, all 291** |
| Top-5 neighbours, Chroma vs brute force (25 probe vectors) | **100% overlap, 25/25 exact** |
| Max score difference | **7.15 × 10⁻⁷** (float32 noise) |
| Amendment links resolved | 12 of 12 |

Without this step, every table below could be measuring a migration bug and attributing it
to an experimental variable.

### 4.2 Retrieval tuning

Recall against gold chunks, 16 in-scope questions. `ctx chars` is the mean context length
handed to the LLM — the **price** of the recall.

| variant | recall | full-recall | precision | MRR | ctx chars |
|---|---|---|---|---|---|
| `k3_nopc` | 78.1% | 10/16 | 37.5% | 0.771 | 2,977 |
| `k3_pc` | 80.2% | 11/16 | 34.1% | 0.771 | 4,060 |
| `k5_nopc` | 91.7% | 14/16 | 28.8% | 0.771 | 5,335 |
| `k5_pc` | 93.8% | 15/16 | 23.9% | 0.771 | 7,329 |
| `k10_nopc` | **100%** | 16/16 | 16.2% | 0.781 | 11,652 |
| `k10_pc` | 100% | 16/16 | 13.7% | 0.781 | 14,217 |

The six amendment-aware variants are omitted here because they change recall by exactly
zero at every `k` — see §4.3. The full 12-variant table is in
[`results/retrieval_experiments.md`](results/retrieval_experiments.md).

Parent-child expansion — the feature carried over from the previous project to fix a known
retrieval failure — recovers **exactly one additional question** at `k=3` and at `k=5`, and
**nothing** at `k=10`, while costing **+36%, +37% and +22% context** respectively.

One question out of sixteen is the smallest difference this benchmark can resolve (§5 item 1).
So the honest reading is: parent-child is worth roughly one question at small `k`, becomes
irrelevant once `k` is large enough, and is never free. Whether one question justifies a
third more context is a judgement call about the operating point, not something these
numbers settle.

**This table is also a correction of an earlier version of itself, and the correction is
the more instructive result.** An earlier pass reported parent-child buying *nothing* at
`k=5`. That was measured against an incomplete label: two multi-hop questions had a gold
chunk missing from `reference_chunk_ids` — specifically a `_part_2` chunk that only arrives
*because* parent-child pulls it in. The label had been written on the assumption that
parent-child would fetch it anyway, which quietly made the yardstick depend on the very
setting under test, and hid the feature's only real benefit. Fixing the labels moved `k5_nopc`
from 93.8% to 91.7% and restored a measurable gap.

The lesson is not about parent-child. It is that **a benchmark label must be independent of
the system configuration being measured**, and that vague reference answers are what let
incomplete labels survive: the missing chunk was found only when the reference answer was
rewritten to state the obligation concretely instead of deferring to "as provided in this
Decree". `eval/benchmark.py` now warns on deferring phrases for that reason.

Two further readings of the same table. **Precision falls from 37.5% at `k=3` to 16.2% at
`k=10`** — the cost of buying recall by widening the window rather than by ranking better.
And **`k10_nopc` strictly dominates `k10_pc`**: identical 100% recall and full-recall, 18%
less context, higher precision. Once `k` is large enough to reach every gold chunk on its
own, parent-child has nothing left to contribute and is pure overhead.

An internal consistency check worth recording: MRR is identical across every variant at the
same `k` (0.771 at `k=3` and `k=5`) and rises to 0.781 at `k=10`. The 0.010 difference over
16 questions is 0.16 ≈ 1/6.25 — exactly one question whose first gold chunk sits at about
rank 6. That is `mh02`, missed at `k=5` and found at `k=10`. The numbers agree with each
other in a way they only can if the harness is computing what it claims to.

### 4.3 Amendment handling: three mechanisms, and none of them was load-bearing

The corpus links twelve articles of Decree 43/2017 to the clauses of Decree 111/2021 that
amend them. Those links are parsed at index time, and `expand_amendments` follows them: when
a retrieved article has been amended, the amending clause is pulled into context.

The prediction before running the sweep was that this would rescue `mh02` — the one question
still missed at `k=5`, whose two gold chunks are related by exactly that amendment link.

It did not. **Recall is identical to three significant figures at every `k`, with and without
amendment expansion**: 78.1/78.1, 80.2/80.2, 91.7/91.7, 93.8/93.8, 100/100, 100/100. Not one
question gained a single gold chunk. Precision drops slightly (23.9% → 23.2% at `k5_pc`) and
context grows 4%, so on this benchmark the feature is a small pure cost.

Two reasons, and the second is the more interesting:

1. **Structural expansion amplifies retrieval; it cannot substitute for it.** `mh02` was
   missing *both* of its gold chunks, so there was no anchor to expand from. A graph edge is
   useless without an entry point into the graph.
2. **The amending clauses already rank on their own.** Decree 111/2021 quotes the article it
   replaces verbatim — `"Điều 10. Nội dung bắt buộc thể hiện trên nhãn hàng hóa …"` — so it is
   semantically as close to a query as the original is. Vector search finds it without help.
   In an earlier manual probe the two Decree 111 clauses came back at ranks 3 and 5 as plain
   *hits*, not as expansions; the signal was there before the sweep confirmed it.

**The third mechanism — the in-context notice — was then ablated, and it is not the cause
either.** `format_context()` prepends `LƯU Ý: <amended by …>` to any chunk carrying an
amendment in its metadata: a string already on disk, costing zero LLM calls and zero extra
retrieval. It was the remaining candidate explanation for why `mh03` — the question built
around a repealed provision — is answered correctly. Re-running that one question with
`--no-amendment-notice` holds retrieval fixed (identical 7 chunks, identical order, identical
scores) and removes only the notice:

| | context includes the notice | answer |
|---|---|---|
| baseline | yes | correct — "No", cites the repealing clause |
| `--no-amendment-notice` | no | **correct — same conclusion, same clause cited** |

The answer survives without it. The reason is visible in the retrieval log: chunk `[1]` is
Article 2 of Decree 111/2021, whose heading is literally *"Repeal and replacement of certain
provisions of Decree 43/2017"*, retrieved at **rank 1, similarity 0.885**. The model is reading
the repeal from the primary text, not from a metadata hint.

So all three amendment mechanisms measure as non-load-bearing on this corpus:

| mechanism | cost | measured effect |
|---|---|---|
| `expand_amendments` (graph traversal) | +4% context, −0.7 pt precision | **0.0 recall at every `k`** |
| `check_amendment` (agent tool) | 1 extra LLM round-trip | called 2/4 times, **never on the two questions that hinged on an amendment** (§4.6) |
| `LƯU Ý` notice in context | free | **answer unchanged when removed** |

What did the work was ordinary vector retrieval, for a reason specific to how Vietnamese
amending decrees are written: they restate and name the provisions they replace, so the
repealing text is lexically and semantically close to any query about the original.

Two honest limits on that conclusion. The ablation is **one question, one run** — it rules the
notice out as *necessary* for `mh03`, not as useless in general. And the whole finding is a
property of this corpus: it would not hold for an amendment expressed as *"in Article 8, replace
'30 days' with '45 days'"*, which names no subject matter and would not rank for a topical query.
The benchmark contains no such case, which is itself a gap (§5 item 8).

This **narrows** the case for a graph representation (§7) considerably rather than supporting it.
Retrieving both the original and the amending text is easy here — three separate mechanisms built
to help with it all measured at zero. The case that survives is about *relation*: knowing which
provision supersedes which, which recall does not measure at all.

### 4.4 Out-of-scope questions cannot be filtered by a similarity threshold

| question | top-1 similarity |
|---|---|
| `oos03` trademark registration procedure (IP law — high lexical overlap) | **0.8641** |
| `oos02` administrative fine for mislabelled origin | 0.8436 |
| `oos01` import tariff rate on milk powder | 0.8177 |
| `oos04` maternity leave entitlement (unrelated domain) | 0.8151 |
| **lowest in-scope question** | **0.8569** |

The classes **overlap** (gap = −0.0072). The worst offender is the lexical trap: a question
about *trademark* registration, which shares almost every keyword with the corpus but
belongs to intellectual-property law, scores **higher than a genuine in-scope question**.

Consequence: a `if similarity < threshold: refuse` rule would either never fire or would
reject real questions. Refusal has to be driven by the prompt and **measured on the
out-of-scope set**. Anyone who ships a threshold here without measuring it has a filter
that silently never fires.

This also invalidates using mean retrieval score as a confidence signal — doubly so because
structurally-expanded chunks inherit their parent hit's score, so "confidence" would rise
with the amount of expansion rather than the amount of evidence.

### 4.5 Answer-level results (baseline)

Configuration `k5_pc` (top_k=5, parent-child on), generator and judge both
`gemini-3.5-flash-lite`, 20 questions (16 in-scope, 4 out-of-scope). Every figure below is
reproduced by the JSON files in `results/`.

**Deterministic tier** — no LLM involved in scoring:

| Metric | Value | Denominator |
|---|---|---|
| Context recall | 93.8% | 16 in-scope |
| Full recall (all gold chunks present) | 93.8% | 16 in-scope |
| Context precision | 23.9% | 16 in-scope |
| MRR | 0.77 | 16 in-scope |
| Citation present | 100% | 15 answered (mh02 refused, so undefined) |
| Refusal correct | 95.0% | 20 |
| — of which out-of-scope refused | 100% (4/4) | 4 |
| False refusal | 6.2% (1/16) | 16 in-scope |
| Tool calls per question | 1.00 | 20 |

By question type: `single_hop` recall 100%, `multi_hop` 87.5%, tag `amended` 66.7%. The false
refusal and the whole `amended` shortfall are the same question, `mh02` — one question, three
bad-looking cells. See §5 items 1 and 8.

**This table is unchanged from an earlier run on `gemini-3.6-flash` — every cell, to the
decimal.** Recall, precision and MRR are model-independent by construction, but refusal
accuracy and citation rate are not, and two models of visibly different capability produced
identical numbers. At this scale the deterministic tier is measuring the *retrieval-and-prompt
contract* rather than model capability: stable, and therefore also unable to distinguish a
strong generator from a weak one. That distinction has to come from elsewhere.

**LLM-as-judge tier** — reported with coverage and the configuration that produced it:

| Metric | Value | Coverage | Context fed to judge | Refusals |
|---|---|---|---|---|
| `non_llm_context_recall` | 0.9375 | 16/16 | raw chunk text | kept |
| `faithfulness` | 1.0 | 15/15 | as shown to the model | dropped |

`non_llm_context_recall = 0.9375` is a **cross-check that agreed**: the deterministic tier
compares `chunk_id` sets, Ragas compares context strings with fuzzy matching — two different
matching mechanisms over the same labels, landing on the same 15/16. That agreement is what
licenses trusting the fuzzy matcher elsewhere.

`faithfulness = 1.0` is **not** a strong result. Read together with §4.6, this judge returned
1.0 on 30 of 31 gradings across both systems. A metric with almost no variance cannot rank
anything, and it is the same model that wrote the answers (§5 item 6). The defensible claim is
that faithfulness **detected no grounding failure**, not that grounding is perfect.

The four measurement defects that had to be fixed before either of these numbers
could be trusted are documented in §4.7.

### 4.6 Agent vs baseline

Both run on `gemini-3.5-flash-lite`, same system prompt, same retriever, same 20 questions.
Holding those fixed is what makes the columns comparable at all.

| metric | baseline | agent | Δ |
|---|---|---|---|
| Context recall (16 in-scope) | 93.8% | **100.0%** | +6.2 pt |
| Full-recall rate | 93.8% | **100.0%** | +6.2 pt |
| Refusal accuracy (20) | 95.0% | **100.0%** | +5.0 pt |
| False-refusal rate | 6.2% | **0.0%** | −6.2 pt |
| Context precision | 23.9% | 23.7% | ≈ 0 |
| MRR | 0.77 | 0.76 | ≈ 0 |
| Citation present | 100% (15/20) | 100% (16/20) | 0 |
| **Mean tool calls / question** | **1.00** | **1.35** | **+35%** |
| Mean context chars | 7,752 | 6,641 | −14% |

The whole gap is one question. `mh02` is the single question the static pipeline missed at
`k=5`, and it is the same question responsible for its false refusal and for the `amended`
tag sitting at 66.7%. The agent retrieved both missing chunks by issuing a second, narrower
query, so `amended` goes to 100% and the false refusal disappears. Everything else — precision,
MRR, citation rate — is flat.

Three things must be said next to that table, or it over-claims:

1. **A cheaper fix reaches the same place.** §4.2 shows plain `k=10` retrieval also hits 100%
   recall, with no agent and no extra LLM call. The agent's win here is *not* evidence that
   agents retrieve what static pipelines cannot; it is evidence that an agent can recover from
   a `k=5` budget by spending a second query. Which of the two is cheaper depends on whether
   context tokens or LLM calls are the binding cost.
2. **+6.2 points is exactly one question out of sixteen** — the smallest quantity this
   benchmark can express (§5 item 1). It is a directionally consistent result, not a measured
   effect size.
3. **The trajectory column is where the cost shows up.** 1.35 tool calls per question means
   the agent issued extra retrievals on questions it did not need them for. On the eight
   single-hop questions it averaged 1.25 — a 25% overhead that bought nothing, since the
   baseline already scored 100% recall there. This is precisely the failure mode the eight
   single-hop questions were included to catch, and it did occur.

**Faithfulness does not separate the two systems, and the raw table exaggerates that it
tries to.** `SUMMARY.md` shows `agent 0.9875` against `baseline 1.0`, but those are not the
same question set: refusals are excluded from faithfulness (§4.7), the baseline
refused `mh02`, and the agent answered it — so the agent was graded on 16 questions and the
baseline on 15, with the extra one being the hardest in the benchmark. On the shared 15:

| | baseline | agent |
|---|---|---|
| `faithfulness`, shared 15 questions | 1.0000 | 0.9867 |

The entire difference is one question (`sh03`, scored 0.8). Every other answer from both
systems scored exactly 1.0 — including `mh02`, the question the agent alone answered. A judge
that returns 1.0 on 30 of 31 gradings has almost no variance to rank systems with, and it is
the same model that generated the answers (§5 item 6). The honest statement is that
faithfulness **detected no grounding failure in either system**, not that either system is
better. The deterministic tier is what separates them.

**A second cross-check that agreed.** `non_llm_context_recall` for the agent is **1.0 (16/16)**,
matching its deterministic full-recall of 100% exactly — as the baseline's 0.9375 matched its
15/16. Two metrics computed by different mechanisms (chunk-id set comparison vs fuzzy string
matching) over the same labels agree on both systems.

**What the trajectory records actually contain.** Across the 20 questions:

| | count |
|---|---|
| `search_regulations` calls | 25 |
| `check_amendment` calls | 2 |
| Tool calls returning an error | 0 |
| Runs that hit the step cap | 0 |
| Questions by tool-call count | 0 calls: 1 · 1 call: 12 · 2 calls: 6 · 3 calls: 1 |

`mh02` — the question the baseline missed — shows the mechanism plainly: a broad first query
(*"ghi xuất xứ hàng hóa Nghị định 43 2017"*), then a narrower second one naming the article
(*"Điều 15 Nghị định 43 2017 xuất xứ hàng hóa gốc"*). That second query is what recovered both
gold chunks.

**A mandatory prompt rule was obeyed 2 times out of 4.** The agent's system prompt contains, in
imperative form: *"call `check_amendment` before asserting the content of any article of Decree
43/2017"*. Four benchmark questions have a Decree 43/2017 article among their gold chunks:

| question | gold chunks include Decree 43/2017 | called `check_amendment` |
|---|---|---|
| `sh06` | `43/2017_Điều_14` | **yes** — returned *"no amendment recorded"* |
| `sh08` | `43/2017_Điều_4` | **yes** — returned *"no amendment recorded"* |
| `mh02` | `43/2017_Điều_15` (+ the Decree 111 clause amending it) | no |
| `mh03` | `43/2017_Điều_8` (+ the Decree 111 clause repealing it) | no |

The compliance rate is 50%, and *which* half it complied on is the finding: the tool fired on the
two questions whose articles turned out **not** to be amended, and was skipped on both questions
that actually hinge on an amendment — including `mh03`, the one question in the benchmark built
around a repealed provision. The rule was followed exactly where it did not matter.

The likely reason is visible in the data rather than guessed: for `mh02` and `mh03` the search
results already carried the `LƯU Ý: <amended by …>` line, so the model had its answer and no
reason to ask again.

Two things follow, and the second is the more useful one:

1. It is §4.3 reached from another direction — the expensive amendment mechanism is redundant
   with a free one. The tool is not worthless (it is the only component that can rule on an
   article that was *never retrieved*), but on this benchmark it was never load-bearing, and
   presenting it as a headline feature would misrepresent the trajectories.
2. **A rule written as mandatory in a prompt is not a constraint — it is a request with a
   compliance rate, and that rate is measurable.** Here it was 50%, concentrated in the cases
   where compliance was pointless. Nothing failed, nothing errored, and the answers were correct
   anyway; an evaluation that looked only at outputs would have reported a system that follows
   its instructions. If the amendment check has to be guaranteed, it belongs in code — a forced
   call before generation — not in a prompt.

**One trajectory finding the outcome metrics hide.** On `oos04` the agent produced the correct
refusal with **zero tool calls** — it never searched before declaring the question out of scope.
Outcome scoring records that as a success; the trajectory record shows it never checked. In a
regulatory setting, refusing without looking is right by accident, and a benchmark that only
scored outcomes would have reported it as a clean 4/4. This is the concrete case for evaluating
agents on two tiers rather than one.

`results/agent_vs_baseline.md` and `results/SUMMARY.md` are generated by
`python -m eval.run_eval` and `python -m eval.report`. See §6.

### 4.7 Measurement defects found and fixed

Six defects were found in the harness and in this report during the work. They are collected here
rather than scattered through the results, because they share one property that matters
more than any of them individually: **not one produced an exception, a NaN, or a warning.**
Each returned a plausible number.

Four of them surfaced together. The first complete judged run (`gemini-3.6-flash`) returned
`faithfulness 0.8862` over 17/20 samples and `non_llm_context_recall 0.75` over 20/20. Both
looked reportable; neither was valid. Reading the **per-sample** scores rather than the means
is what exposed them:

| question | score (superseded run) | what it actually was |
|---|---|---|
| `mh02` | 0.0 | a refusal asserts no claims, so the ratio is undefined; Ragas returns 0.0, penalising it as if it had fabricated |
| `mh03` | 0.5 | the answer was correct — verified by reading — but quoted the in-context amendment notice, which was missing from the contexts handed to the judge |

1. **The evaluation context was not the context the model saw.** `RagAnswer.contexts` held raw
   chunk text; the prompt was built by `format_context()`, which prepends a citation header and
   a `LƯU Ý: <amended by …>` line. Anything the model derived from that notice scored as
   unsupported — so the metric penalised precisely the amendment-awareness feature this project
   exists to test. Fixed by splitting `format_blocks()` out of `format_context()`, recording a
   separate `contexts_shown` field on both systems, and adding `--context-source raw|shown` so
   each metric receives the context it is actually asking about. Older records make
   `--context-source shown` abort rather than silently fall back.
2. **Out-of-scope questions were dragging a retrieval metric.** They have no gold chunks, so
   their `reference_contexts` held a placeholder that fuzzy matching scores at 0 — four
   questions counted as retrieval failures for correctly having no answer. The distortion is
   exactly `0.9375 × 16/20 = 0.75`: the metric was reporting retrieval quality *times the
   in-scope fraction of the benchmark*, so adding out-of-scope questions would lower the score
   without changing the system. They are now excluded by default (`--include-oos` opts back in)
   and scored by the deterministic refusal metric instead.
3. **The missing 3/20 was not a quota failure.** Three samples died with
   `IncompleteOutputException: output is incomplete due to a max_tokens length limit` —
   faithfulness has the judge emit the answer decomposed into a JSON claim list, and the longest
   answers (enumerating 5–9 clauses) overflow the model default. Retrying could never fix it.
   Worse, the loss was *correlated with answer length*, so the surviving 17 were a biased
   sample, not merely a smaller one. Fixed with `--max-tokens` on the judge.
4. **A refusal has no defined faithfulness.** `--drop-refused` excludes those rows; refusals are
   scored by refusal accuracy, which is the metric that applies to them.
5. **A retried run left two rows for the same question, and scoring both corrupted every rate.**
   `eval/generate.py` appends and treats a row carrying an `error` as not-yet-done — which is what
   makes retrying after a quota failure work. But a question that failed on a 429 and succeeded on
   retry then appears twice. The first agent scoring pass ran over 27 rows for a 20-question
   benchmark, counting seven never-answered rows as failures: it reported the agent at 89.5%
   recall, 85.2% refusal accuracy and 69.6% citation rate, against true figures of 100%, 100% and
   100%. `eval/run_eval.py` now keeps exactly one row per question — preferring a successful row
   over an errored one — prints how many lines it dropped, and names any question still without an
   answer. **An append-only result log is not a result set until it is deduplicated.**

6. **A claim in this report was itself measured the wrong way.** §4.6's tool-compliance figure
   was first computed by searching for the string `"43/2017"` in the **answer text**. That
   misclassified in both directions: `sh06` and `sh08` do cite Decree 43/2017 articles but write
   only `[1]`, so they were dropped from the denominator — even though they are precisely the two
   questions that *did* call the tool; and `mh08`, whose gold chunks are entirely Decree 111/2021,
   was counted in. The figure printed was "0 of 3". Recomputed from `reference_chunk_ids` — the
   benchmark label rather than the model's prose — it is **2 of 4**, and the corrected version is
   the more interesting result. It is §4.2's lesson again in a new place: **classify by the label,
   never by parsing the system's own output**, because the output is the thing under test. This
   one was caught by an external code review rather than by the harness, which is why the
   deterministic layer now carries unit tests (`tests/test_metrics_local.py`, §6).

Together these are the reason for the defensive choices visible throughout the harness: every
score printed with `n_ok/n`, the deterministic and LLM-judged tiers cross-checked against each
other, `n` printed inside comparison tables rather than beside them, and `eval/inspect_run.py`
kept as a required step rather than an optional one. Each defect above was caught by one of
those or, in the case of defect 6, by an external code review — **none of them by a test**,
because until then there were none. That is why the deterministic layer now carries
`tests/test_metrics_local.py`, which pins the exact behaviours defects 2, 4, 5 and 6 turned on.

## 5. Limitations

Stated plainly, because a benchmark this size invites over-claiming.

1. **n = 20 (16 in-scope).** One question is worth 6.25% of the full-recall rate. Any gap
   narrower than that is a single question changing its mind, not a finding — which is
   exactly the size of the parent-child effect in §4.2. No confidence intervals are reported
   because at this n they would be wider than every effect measured.
2. **Single annotator, no cross-labelling.** Reference answers were extracted from the
   source text and then verified by hand; there is no second annotator and no inter-annotator
   agreement statistic.
3. **Reference answers were drafted with LLM assistance from the source articles and then
   human-verified**, not written from scratch. Said explicitly because the provenance of a
   benchmark is part of its credibility. The `verified_by_human` flag in
   `data/benchmark.jsonl` records the current state per question and the harness prints it.
4. **Questions were authored, not sampled from real user logs.** Real questions are shorter,
   less well-formed, and use less precise legal vocabulary.
5. **`reference_chunk_ids` marks chunks *sufficient* to answer, not every relevant chunk.**
   Precision computed against that label therefore penalises useful-but-unlabelled context.
   Recall is the more trustworthy of the two.
6. **Ragas is LLM-as-judge, and here the judge is the *same model* as the generator.** Both
   are `gemini-3.5-flash-lite`. This is the strongest form of self-preference bias — not merely
   the same family, the same model grading its own output — on top of the usual run-to-run
   variance and false precision. It shows: this judge returned 1.0 on 30 of 31 gradings, so it
   has essentially no power to separate the two systems (§4.5, §4.6), and the separation in
   this report comes from the deterministic tier instead. `eval/inspect_run.py` exists to
   compare the judge against manual reading. Re-running the judge on a different model family
   would quantify the bias; it has not been done, and the free-tier daily quota is the reason.
7. **The agent is non-deterministic.** Temperature is 0, which reduces but does not remove
   run-to-run variation in tool choice.
8. **"Multi-hop" here means reasoning, not retrieval.** The multi-hop questions require
   composing two provisions, and in several cases require establishing which regime applies
   before knowing which article to consult. But the provisions involved often sit next to
   each other in the same instrument, so at `k=10` a single query already returns every gold
   chunk. The benchmark therefore does **not** show agent superiority on retrieval reach; if
   the agent wins, it has to win on answer composition and on citing the amended text.
   Building genuinely retrieval-multi-hop questions — spanning instruments that a single
   query cannot cover — is future work.
9. **`min_steps` is a floor, not an expectation.** It records the minimum number of
   *semantically independent* lookups, so `steps_over_min` is a comparison against a lower
   bound rather than an efficiency verdict. An agent that spends an extra call on
   `check_amendment` to confirm a provision is still in force is taking the *correct* path
   while scoring `+1`. Read that column alongside the trajectory, not on its own.
10. **Amendment expansion is one-directional, and its measured value is zero.** Original
    article → amending clause is wired; the reverse is stored in metadata but not used. §4.3
    shows the forward direction changed recall by 0.0 at every `k`, so there is no reason to
    expect the reverse direction to help either — and it could actively harm by pulling
    superseded text back into context. The feature is kept in the code and in the report
    because a measured null result is worth more than an unmeasured feature.
11. **A negative retrieval result does not clear the underlying risk.** Recall says both the
    original and the amending text are retrievable. It says nothing about whether the model
    then cites the right one. That failure mode is measured at the answer level (`mh03`),
    not here, and the two must not be conflated.
12. **`faithfulness` cannot distinguish a legally correct answer from a legally superseded
    one.** On `mh03` the system answered correctly and cited the repealing clause — verified
    by reading the output, not by the score. Had it answered from the repealed article
    instead, that answer would also have been fully grounded in retrieved text, and
    `faithfulness` would also have scored it high. The metric measures *did it invent
    anything*, not *is the provision still in force*. This is why manual cross-check
    (`eval/inspect_run.py`) is a required step of the harness rather than an optional one,
    and why `faithfulness` is never reported alone.
13. **Tests cover the deterministic metric layer only.** `tests/test_metrics_local.py` pins the
    behaviour that the measurement defects in §4.7 turned on — None vs 0.0 for undefined metrics,
    prefix matching on the refusal sentence, citation-index validity, aggregation that excludes
    out-of-scope rows from recall, and record deduplication. Retrieval, the agent loop and the
    Ragas wiring have no automated tests: they need a vector store, an API key and a paid quota,
    so they are exercised by `run_all.py` and by reading `eval/inspect_run.py` output instead.
14. **The amendment ablation is a single question, run once.** §4.3 removes the in-context
    notice for `mh03` and the answer is unchanged, which rules the notice out as *necessary* on
    that question. It does not establish that the notice is useless: `n = 1`, the generator is
    non-deterministic even at temperature 0, and the question happens to retrieve the repealing
    article at rank 1. A corpus where the amending text does not rank would likely reverse the
    result, and this benchmark contains no such case.

## 6. Reproducing

```bash
conda activate ai
cd AI_LLM/project5_rag_agent

python -m pytest tests/ -q       # 11 unit tests, no API key, ~1 s
python -m eval.doctor            # environment + version check, prints fixes

python run_all.py --no-llm       # everything that needs no API key (~1 min)
python run_all.py                # + generation, agent, Ragas (needs GOOGLE_API_KEY)
python -m eval.report            # consolidate results/ into results/SUMMARY.md
```

Individual stages:

```bash
python -m rag.build_index --force        # load Project 4 chunks into Chroma
python -m rag.verify_migration           # parity vs brute-force retrieval
python -m eval.benchmark                 # validate benchmark labels
python -m eval.experiments               # retrieval variants (no LLM)
python -m eval.generate --mode baseline  # needs API key
python -m eval.generate --mode agent     # needs API key
python -m eval.run_eval --tags baseline,agent --md agent_vs_baseline

# Ragas runs twice per tag on purpose — the two metrics need mutually exclusive
# configurations, and running them together with the defaults reproduces neither
# of the numbers in §4.5. See §4.5 "How this tier was validated".
python -m eval.ragas_eval --tag baseline --metrics non_llm_context_recall   # no API calls
python -m eval.ragas_eval --tag baseline --metrics faithfulness \
        --context-source shown --drop-refused --max-tokens 4096             # needs API key

python -m eval.ragas_eval --tag agent --metrics non_llm_context_recall       # no API calls
python -m eval.ragas_eval --tag agent --metrics faithfulness \
        --context-source shown --drop-refused --max-tokens 4096

# §4.3 ablation — a separate --tag so it cannot overwrite baseline.jsonl
python -m eval.generate --mode baseline --only mh03 --tag mh03_noNotice \
        --no-amendment-notice
python -m eval.inspect_run --tag mh03_noNotice --id mh03

python -m eval.inspect_run --tag baseline --id mh03   # manual cross-check
python -m eval.report                                 # -> results/SUMMARY.md
```

**Dependency note.** `pip install ragas==0.4.3` on a clean environment resolves
langchain 1.x, and `import ragas` then fails with
`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`.
The working pin set is in `requirements.txt`; `eval/doctor.py` detects the condition and
prints the fix.

## 7. Future work

**GraphRAG — with the case narrowed by §4.3.** The naive argument for a graph index was that
vector similarity approximates *amends* / *repeals* badly, so a traversal would retrieve the
amending clause that similarity misses. **That argument does not survive measurement on this
corpus**: the amending clauses rank on their own, because Decree 111/2021 quotes the article
it replaces, and following the edges added exactly zero recall.

The case that does survive is about **relation, not retrieval**. Getting both provisions into
context is easy; establishing that one supersedes the other, that a clause was repealed
rather than rewritten, or that a repeal is partial (Decree 111 repeals *one sentence* of
Article 8 clause 4, leaving the rest in force) is a question about typed edges between
provisions. Similarity ranking cannot express it at all, and the current workaround — a text
notice in the context plus a tool the model must remember to call — is an approximation of a
graph, and §4.3 now shows that on this corpus neither part of that approximation was doing any
work: the tool was skipped on both questions where an amendment was in play, and removing the
notice changed nothing. That is an argument for building the graph on a corpus where amending
text does *not* rank on its own, not on this one. The next honest step is to build the typed edge set from the
cross-reference text already parsed into the metadata, and to evaluate it **at the answer
level** where the failure actually shows, rather than at the recall level where §4.3 shows
there is nothing left to win.

**Conversational memory.** Multi-turn follow-ups ("and for imported goods?") currently
re-retrieve from scratch. `Context` is already threaded through the agent; what is missing
is an evaluation for it — multi-turn benchmarks need per-turn labels, which the current
schema does not carry.

**Larger benchmark with cross-annotation**, which is the only way to make any of the
narrow gaps above interpretable.

## 8. Repository layout

```
rag/
  config.py            single source of truth: paths, model, prompt, refusal sentence
  build_index.py       Project 4 chunks -> Chroma; resolves amendment links
  retriever.py         retrieval + parent-child + amendment expansion (LLM-free)
  verify_migration.py  parity check vs Project 4 brute force
  pipeline.py          baseline RAG (control condition)
  show_chunk.py        corpus inspection, used to verify benchmark labels
agent/
  tools.py             search_regulations, check_amendment
  app.py               FunctionAgent + trajectory capture
tests/
  test_metrics_local.py  11 unit tests over the deterministic metric layer
eval/
  benchmark.py         schema, validation, recall ceiling
  metrics_local.py     deterministic metrics
  generate.py          answer generation (cached to disk)
  run_eval.py          deterministic scoring + comparison tables
  ragas_eval.py        Ragas scoring of an existing run
  experiments.py       retrieval variant sweep
  inspect_run.py       side-by-side manual cross-check
  report.py            consolidate results/
  doctor.py            environment diagnosis
run_all.py             every stage in order; --no-llm runs the API-free half
data/     benchmark.jsonl, chroma_db/ (gitignored — rebuilt by rag/build_index.py)
results/  generated tables and raw scores; runs/*.jsonl are the raw model outputs
```

Built on Project 4 (`../project4_rag_assistant`), which produced the chunking and the
embeddings this project reuses unchanged.
