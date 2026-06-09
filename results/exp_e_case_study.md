# Exp-E — Interpretability case study

Seed = 1. Layer 4 α = 0.30, β = 0.70. Four cases, one per Lemma-1 tier quadrant.

## Case A_critical_attack

- **tx**: `0x94359bd3f0fc4cd53b96054a23b05f531e127fa6cca644dfc870851af5e0e3dc` — ground truth = **malicious**.
- **dst-side observation**: chain=6648936, token=`0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48`, beneficiary=`0xa8ecaf8745c56d5935c232d2c5b83b9cd3de1f6a` (in PeckShield: **False**), amount=202440725413, value≈$202427, origin_observed=False.
- **Layer 2 (Graph2Vec)**: anomaly_score = 0.9991 → λ_τ = `high_risk`.
- **Layer 3 (Datalog)**: σ_τ = 3; matched origin? **False**.
    - Rule trace:
        - forged_withdrawal (σ=3): no origin-side burn observation matching (withdrawal_id=4922, beneficiary=0xa8ecaf8745c56d5935c232d2c5b83b9cd3de1f6a, amount=202440725413) was found before dst_ts=1659394314.
- **Layer 4**: R_τ = α·s + β·σ/3 = 0.30·0.999 + 0.70·3/3 = 0.9997 → tier = **critical**.
- **Black-box reference**: GCN score = 1.0000 (`high_risk`), GraphSAGE score = 1.0000 (`high_risk`). These outputs lack any per-alert structural explanation analogous to the Layer 3 rule trace above.

## Case C_ml_only_false_positive

- **tx**: `0xa975da348f6ce920892b14a50c08a70dfd0970ae063909f58500418dec0ee24c` — ground truth = **normal**.
- **dst-side observation**: chain=6648936, token=`0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2`, beneficiary=`0xa8047cc795c10c64e71235b07c41f663e51c7196` (in PeckShield: **False**), amount=122030000000000000, value≈$345, origin_observed=True.
- **Layer 2 (Graph2Vec)**: anomaly_score = 0.6188 → λ_τ = `high_risk`.
- **Layer 3 (Datalog)**: σ_τ = 0; matched origin? **True**.
    - Rule trace:
        - (no Layer 3 rule fired — Datalog signal silent.)
- **Layer 4**: R_τ = α·s + β·σ/3 = 0.30·0.619 + 0.70·0/3 = 0.1856 → tier = **suspicious**.
- **Black-box reference**: GCN score = 0.0000 (`normal`), GraphSAGE score = 0.0000 (`normal`). These outputs lack any per-alert structural explanation analogous to the Layer 3 rule trace above.

## Case D_clean_normal

- **tx**: `0x13dbd2fa803bdc7545518a3e4ef9b5770e859b821da31fd7a4beed4d78e2ed8a` — ground truth = **normal**.
- **dst-side observation**: chain=6648936, token=`0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48`, beneficiary=`0x05f51de33bd21abd4172634eddc78c8884212033` (in PeckShield: **False**), amount=92000000000, value≈$91942, origin_observed=True.
- **Layer 2 (Graph2Vec)**: anomaly_score = 0.0052 → λ_τ = `normal`.
- **Layer 3 (Datalog)**: σ_τ = 0; matched origin? **True**.
    - Rule trace:
        - (no Layer 3 rule fired — Datalog signal silent.)
- **Layer 4**: R_τ = α·s + β·σ/3 = 0.30·0.005 + 0.70·0/3 = 0.0016 → tier = **normal**.
- **Black-box reference**: GCN score = 0.0000 (`normal`), GraphSAGE score = 0.0000 (`normal`). These outputs lack any per-alert structural explanation analogous to the Layer 3 rule trace above.

