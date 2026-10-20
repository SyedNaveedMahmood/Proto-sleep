# Phase VIII offline research dashboard

Primary **B1** robustness gate: **FAIL**. Additive confirmation: **NOT TESTED**. CUDA: **BLOCKED**. Provenance: **PASS**.

| Task | Seed | Method | Clean F1 | Competitive F1 | Added F1 | Competitive ECE |
|---|---:|---|---:|---:|---:|---:|
| single | 123 | P0 | 1.000000 | 0.433349 | 0.999000 | 0.426510 |
| single | 123 | R1 | 0.869854 | 0.731286 | 0.833128 | 0.036408 |
| single | 123 | R2 | 0.715794 | 0.638450 | 0.706528 | 0.050511 |
| single | 123 | B1 | 0.753671 | 0.678202 | 0.749168 | 0.063558 |
| single | 123 | A1 | 0.550341 | 0.492418 | 0.542574 | 0.083355 |
| single | 123 | T1 | 0.856215 | 0.811722 | 0.779391 | 0.085028 |
| single | 456 | P0 | 0.981976 | 0.722616 | 0.915851 | 0.215003 |
| single | 456 | R1 | 0.747855 | 0.627467 | 0.774183 | 0.064855 |
| single | 456 | R2 | 0.840142 | 0.716827 | 0.842483 | 0.053055 |
| single | 456 | B1 | 0.926616 | 0.816445 | 0.951842 | 0.057150 |
| single | 456 | A1 | 0.740544 | 0.547910 | 0.735622 | 0.174165 |
| single | 456 | T1 | 0.624494 | 0.544566 | 0.586672 | 0.054799 |
| single | 789 | P0 | 0.998000 | 0.632271 | 0.997000 | 0.255974 |
| single | 789 | R1 | 1.000000 | 0.982263 | 1.000000 | 0.033596 |
| single | 789 | R2 | 1.000000 | 0.983236 | 1.000000 | 0.034780 |
| single | 789 | B1 | 1.000000 | 0.975467 | 0.999000 | 0.041509 |
| single | 789 | A1 | 0.999000 | 0.873658 | 0.995000 | 0.053744 |
| single | 789 | T1 | 0.934256 | 0.941097 | 0.914024 | 0.063615 |
| multiple | 123 | P0 | 0.931719 | 0.348685 | 0.893949 | 0.408918 |
| multiple | 123 | R1 | 0.733430 | 0.670399 | 0.733281 | 0.053991 |
| multiple | 123 | R2 | 0.711395 | 0.671585 | 0.719058 | 0.039440 |
| multiple | 123 | B1 | 0.754244 | 0.659384 | 0.745788 | 0.047458 |
| multiple | 123 | A1 | 0.724277 | 0.639221 | 0.728318 | 0.040757 |
| multiple | 123 | T1 | 0.865222 | 0.755810 | 0.811805 | 0.037073 |
| multiple | 456 | P0 | 0.981974 | 0.662492 | 0.930007 | 0.133722 |
| multiple | 456 | R1 | 0.833696 | 0.702889 | 0.810444 | 0.065526 |
| multiple | 456 | R2 | 0.838422 | 0.735889 | 0.814804 | 0.073643 |
| multiple | 456 | B1 | 0.787818 | 0.636023 | 0.763135 | 0.076387 |
| multiple | 456 | A1 | 0.844077 | 0.621033 | 0.802593 | 0.130637 |
| multiple | 456 | T1 | 0.921108 | 0.737555 | 0.894800 | 0.037247 |
| multiple | 789 | P0 | 0.994000 | 0.531956 | 0.983045 | 0.274126 |
| multiple | 789 | R1 | 0.973969 | 0.834179 | 0.959185 | 0.024881 |
| multiple | 789 | R2 | 0.984979 | 0.853056 | 0.971000 | 0.026410 |
| multiple | 789 | B1 | 0.990024 | 0.900650 | 0.969091 | 0.053729 |
| multiple | 789 | A1 | 0.991029 | 0.827479 | 0.968015 | 0.033471 |
| multiple | 789 | T1 | 0.936434 | 0.904591 | 0.924265 | 0.052355 |

## Failed primary clauses

- single/123: clean_floor, clean_penalty, competitive, diagnostic_pairs
- single/456: clean_penalty, diagnostic_pairs, nuisance_pairs
- single/789: 
- multiple/123: clean_floor, clean_penalty, competitive, diagnostic_pairs
- multiple/456: clean_floor, clean_penalty, competitive, diagnostic_pairs, nuisance_pairs
- multiple/789: 

## Authorization and next blockers

SYNTHETIC_ONLY; no real EEG/RESERVED/ST/SHHS/legacy NPZ

- Complete robust primary prerequisite before additive confirmation
- Numerically acceptable and diagnostically neutral multi-donor support
- Independent scientific justification and human authorization before real source work

CPU correctness: 825 passed / 11 CUDA skipped. Source and test hashes are in correctness_receipt.json.

Descriptive completion: descriptive_diagnostics_summary.json. Aggregate figures: figures/ (16 charts). No new fits or gate changes in the appendix.
