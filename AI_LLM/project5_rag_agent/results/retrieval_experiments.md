# Retrieval tuning experiments

Benchmark: **20 questions** (16 in-scope, 4 out-of-scope). 
All numbers below are computed by **set comparison against hand-labelled `reference_chunk_ids`** — no LLM is involved, so they are fully deterministic and reproducible.

`recall` = fraction of labelled gold chunks retrieved. `precision` = fraction of retrieved chunks that are gold — see *Limitations*, this metric penalises useful-but-unlabelled chunks. `MRR` = mean reciprocal rank of the first gold chunk. `ctx chars` = mean characters of context handed to the LLM, i.e. **the cost of the recall**.

## Main table

| variant | top-k | parent-child | amendment | recall | full-recall | precision | MRR | chunks | ctx chars |
|---|---|---|---|---|---|---|---|---|---|
| `k3_nopc` | 3 | no | no | 78.1% | 62.5% | 37.5% | 0.771 | 3.0 | 2,977 |
| `k3_nopc_amend` | 3 | no | yes | 78.1% | 62.5% | 36.5% | 0.771 | 3.1 | 2,996 |
| `k3_pc` | 3 | yes | no | 80.2% | 68.8% | 34.1% | 0.771 | 3.7 | 4,060 |
| `k3_pc_amend` | 3 | yes | yes | 80.2% | 68.8% | 33.0% | 0.771 | 3.8 | 4,079 |
| `k5_nopc` | 5 | no | no | 91.7% | 87.5% | 28.8% | 0.771 | 5.0 | 5,335 |
| `k5_nopc_amend` | 5 | no | yes | 91.7% | 87.5% | 27.6% | 0.771 | 5.3 | 5,636 |
| `k5_pc` | 5 | yes | no | 93.8% | 93.8% | 23.9% | 0.771 | 6.4 | 7,329 |
| `k5_pc_amend` | 5 | yes | yes | 93.8% | 93.8% | 23.2% | 0.771 | 6.7 | 7,629 |
| `k10_nopc` | 10 | no | no | 100.0% | 100.0% | 16.2% | 0.781 | 10.0 | 11,652 |
| `k10_nopc_amend` | 10 | no | yes | 100.0% | 100.0% | 15.5% | 0.781 | 10.5 | 11,915 |
| `k10_pc` | 10 | yes | no | 100.0% | 100.0% | 13.7% | 0.781 | 12.1 | 14,217 |
| `k10_pc_amend` | 10 | yes | yes | 100.0% | 100.0% | 13.0% | 0.781 | 12.6 | 14,480 |

## Recall broken down by question type

| variant | single-hop | multi-hop | amended-article questions |
|---|---|---|---|
| `k3_nopc` | 100.0% | 56.2% | 55.6% |
| `k3_nopc_amend` | 100.0% | 56.2% | 55.6% |
| `k3_pc` | 100.0% | 60.4% | 66.7% |
| `k3_pc_amend` | 100.0% | 60.4% | 66.7% |
| `k5_nopc` | 100.0% | 83.3% | 66.7% |
| `k5_nopc_amend` | 100.0% | 83.3% | 66.7% |
| `k5_pc` | 100.0% | 87.5% | 66.7% |
| `k5_pc_amend` | 100.0% | 87.5% | 66.7% |
| `k10_nopc` | 100.0% | 100.0% | 100.0% |
| `k10_nopc_amend` | 100.0% | 100.0% | 100.0% |
| `k10_pc` | 100.0% | 100.0% | 100.0% |
| `k10_pc_amend` | 100.0% | 100.0% | 100.0% |

The breakdown matters more than the headline number: a variant can lift the average while making one class of question worse.

## Out-of-scope separability

Config used: `k5_pc`

| question | type | top-1 similarity |
|---|---|---|
| oos01 | out-of-scope | 0.8177 |
| oos02 | out-of-scope | 0.8436 |
| oos03 | out-of-scope | 0.8641 |
| oos04 | out-of-scope | 0.8151 |
| **lowest in-scope** | in-scope | **0.8569** |

Lowest in-scope top-1 = **0.8569**, highest out-of-scope top-1 = **0.8641** (gap = -0.0072). **The two classes overlap, so a similarity threshold cannot be used to decide when to refuse.** Refusal therefore has to be driven by the prompt and measured on the out-of-scope set, not by a score cutoff.

## Questions still missing gold chunks

- `k3_nopc`: mh01:15/2018/NĐ-CP_Điều_7_part_1,15/2018/NĐ-CP_Điều_7_part_2; mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15; mh04:61/VBHN-VPQH_Điều_38; mh05:15/2018/NĐ-CP_Điều_6; mh06:15/2018/NĐ-CP_Điều_6; mh08:111/2021/NĐ-CP_Điều_1_Khoản_5_part_1
- `k3_nopc_amend`: mh01:15/2018/NĐ-CP_Điều_7_part_1,15/2018/NĐ-CP_Điều_7_part_2; mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15; mh04:61/VBHN-VPQH_Điều_38; mh05:15/2018/NĐ-CP_Điều_6; mh06:15/2018/NĐ-CP_Điều_6; mh08:111/2021/NĐ-CP_Điều_1_Khoản_5_part_1
- `k3_pc`: mh01:15/2018/NĐ-CP_Điều_7_part_1,15/2018/NĐ-CP_Điều_7_part_2; mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15; mh04:61/VBHN-VPQH_Điều_38; mh05:15/2018/NĐ-CP_Điều_6; mh06:15/2018/NĐ-CP_Điều_6
- `k3_pc_amend`: mh01:15/2018/NĐ-CP_Điều_7_part_1,15/2018/NĐ-CP_Điều_7_part_2; mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15; mh04:61/VBHN-VPQH_Điều_38; mh05:15/2018/NĐ-CP_Điều_6; mh06:15/2018/NĐ-CP_Điều_6
- `k5_nopc`: mh01:15/2018/NĐ-CP_Điều_7_part_2; mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15
- `k5_nopc_amend`: mh01:15/2018/NĐ-CP_Điều_7_part_2; mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15
- `k5_pc`: mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15
- `k5_pc_amend`: mh02:111/2021/NĐ-CP_Điều_1_Khoản_7,43/2017/NĐ-CP_Điều_15
