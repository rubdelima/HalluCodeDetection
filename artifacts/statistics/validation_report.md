# Statistical pipeline validation

- RQ1: 13 models × 563 unique, identically paired tasks validated.
- RQ2/RQ3: 3729 test rows clustered within 113 tasks.
- Duplicate evaluation attempts are resolved by retaining the last persisted model/sample row.
- Bootstrap and permutation resampling use the programming task as the independent unit.
- Seed: `20260920`; resamples/permutations: `10000`.
- BLOCKED local tool-assisted raw inputs not mounted: `/mnt/ssd/HalluCode/results/phase6_local_tools/generation_with_tools.jsonl, /mnt/ssd/HalluCode/results/phase7_local_tools/classification_with_tools.jsonl`.
