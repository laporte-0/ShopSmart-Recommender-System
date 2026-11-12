# Figure captions

All V2 scores are development validation scores after model selection, conditional
on users with ≥1 positive unseen item in the training catalog. Fractions range
from 0 to 1. Coverage is sample-dependent. Exact results and bootstrap intervals
are in `reports/results/validation_selected.csv`.

- `validation_hit_rate.png`: Hit Rate@10 for the selected configuration in each family on identical users.
- `validation_ndcg.png`: NDCG@10, the primary validation selection criterion. Family maxima have selection optimism.
- `catalog_coverage.png`: Distinct recommended products / 999 training-catalog items.
- `accuracy_coverage.png`: NDCG@10 versus catalog coverage; neither axis alone measures overall usefulness.
- `user_knn_sensitivity.png`: Neighbor count sensitivity with other displayed hyperparameters held fixed. This is not an exhaustive search.
- `v1_v2_protocol_context.png`: Separate V1 examined test and V2 validation panels. Different dates, catalog construction, targets and populations preclude direct claims of improvement.
