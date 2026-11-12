# V2 development evaluation protocol

This is a new raw-source, globally chronological **development benchmark**, not
an independent final test. Dates and bounded search grid were specified before
running the V2 models. Historical V1 test interactions are not reshuffled into a
purported untouched test. Source data and past human inspection are not fully
provenanced; metadata is a later snapshot. Independent generalization would
require a separately protected future dataset or prospective collection.

## Training and candidate selection

1. Read the entire rating CSV in chunks; retain valid IDs, timestamps and ratings
   in [1,5]. Count only events strictly before 2020-01-01 UTC. Choose the 1,000
   most frequent product IDs, breaking ties lexicographically. This counts events,
   not unique users; repeated pre-cutoff events can affect catalog selection.
2. Within those products, keep the earliest observed rating per user/product;
   users need ≥8 unique pre-cutoff pairs. Keep the first 5,000 users under SHA256
   ID ordering. The cap uses neither validation activity nor relevance.
3. Use all these pre-cutoff pairs as training. Remove catalog products with no
   observations after the user cap. Final candidate/coverage denominator: 999.
4. Later ratings in [2020-01-01, 2022-01-01) from these users form development
   validation. Keep the earliest later rating per pair, remove all training
   pairs, and retain an additional unfiltered file for out-of-catalog analysis.
   Distinct later items are all eligible targets, not just the last positive.
5. Metadata are loaded from raw source for training-catalog products, not chosen
   using V1 IDs. Static text is fit on this catalog only. No validation ratings
   enter profiles, popularity, similarities or factor training gradients.

## User sample and metrics

Eligible ranking users have at least one unseen validation item rated ≥4 in the
training catalog. Draw up to 1,000 distinct users from sorted eligible IDs with
NumPy seed 42; save *all* in-catalog positive targets for each selected user.
All algorithms use this exact sample, the complete 999-item candidate universe,
and remove every product the user rated during training (including low ratings).
This is a warm-user/warm-catalog conditional scenario. Later out-of-catalog
positives cannot be recalled and are excluded only from this clearly labelled
conditional metric; their proportion and full-target recall are recorded in
`reports/results/full_target_recall.csv` and the README. Users without a later positive are excluded, not counted as successes.

For each user: Precision@K = number of retrieved relevant items / requested K,
including empty-list positions as misses; Recall@K divides by all relevant
held-out items; HR@K = indicator of ≥1 hit. NDCG uses binary gains and log2(rank+1)
discounts with ideal length min(K, relevant count). MRR@10 is reciprocal rank of
the first relevant item in the top 10. Macro-average over users, not interactions.
With a singleton target, Recall = HR and Precision = HR/K. With multiple targets,
these equalities generally do not hold. Unknown interactions are not observed
negative feedback; these metrics measure recovery of known positives, not
true user utility or the precision of all recommended products.

Coverage = distinct recommended training-catalog items / 999, dependent on sample
size. Average list length counts unique returned items, capped at 10. Diversity
is mean pairwise cosine distance in the same baseline TF-IDF representation
across every model; lists shorter than two have undefined diversity and are
excluded from that mean. This representation-dependent measure is not novelty
or human judgement of meaningful product differences.

Only MF receives RMSE/MAE: clipped [1,5] predictions versus **all** warm-catalog
validation ratings, not only positives and not only sampled ranking users.
MF uses validation RMSE for early stopping (patience 5) and NDCG for selecting
hyperparameters; training gradients still use only observed training ratings.

## Selection, fusion and uncertainty

Primary selection criterion: maximum macro validation NDCG@10; tie-break HR@10,
then model name. Search is a bounded exploratory grid, not exhaustive or nested
cross-validation. User k: 10/20/30/50/100, overlap 1/2/3, shrinkage 0/10/20,
minimum neighbor support 1/2, regularization 0/2, optional mean-centering.
Item k: 10/30/50/100, overlap 1/2/3, shrinkage 0/10/20, score normalization vs sum.
Exact joint settings are stored for every run. Item positive rating strength is
rating−3; V1 and V2 both use this weighting. Its independent ablation is future
work. Content compares equal positive averaging, separate field weights,
bigrams, signed preferences and MMR. Near-duplicate suppression uses content
similarity rather than assuming distinct IDs imply different products.

MF compares 8/16/32 factors, learning rates .01/.02, regularization .05/.1,
40/60 maximum epochs. Minibatch SGD sums observed gradients for repeated IDs;
regularization is per observation. RRF combines the best content, item and MF
models and optional popularity; a component contributes weight/(60+rank) for
its first 100 finite-scored unseen items. Absent component scores contribute
zero. Items missed by every component receive a tiny popularity fallback below
any nonzero fusion contribution. Weights are selected on validation only.

Store one per-user row per model; percentile bootstrap 1,000 resamples of users,
seed 42, provides descriptive 95% intervals and paired NDCG differences. These
intervals do not adjust for model selection or multiple comparisons and cannot
establish a significant improvement on unseen data.


## Interpretation and historical workflow

Notebooks 01–05 preserve exploration, original full-period selection,
per-user chronological leave-two-out, content/popularity, user/item KNN and the
historical comparison. V1 selected 2,000 popular products and 7,368 active users
from the full timeline before splitting. Training contained 1,998 items; content
could search all 2,000 metadata items. The globally chronological benchmark
fixes selection leakage and unifies the candidate universe. It cannot fix later
metadata snapshots or establish an untouched independent final test.

Item_V1 has the best observed validation NDCG@10 (0.03092), HR@10 9.7% and
coverage 79.28%. Its descriptive user-bootstrap NDCG interval is
[0.02417, 0.03777], and HR interval [7.9%, 11.7%]. Paired differences versus
several alternatives cross zero; these do not demonstrate independent superiority.
The best hybrid reaches NDCG 0.02907. Weighted text and MMR do not beat the
original text profile; MMR trades accuracy for coverage/diversity. Sparse overlap
constraints can remove useful evidence. MF uses an observed-rating objective;
selected MF_1 reaches RMSE 1.12649 and MAE 0.73177 but only HR@10 1.5%, showing
that rating-prediction quality does not imply positive-item ranking quality.

89.8% of later unseen pairs (21,327 / 23,663) and 89.8% of later positives
(17,215 / 19,167) lie outside the frozen catalog. Only 1,139 of 3,095 users with
later positives qualify for warm-catalog ranking. Winner Recall@10 is 0.06036
against in-catalog targets, falling to 0.01926 against all later positive targets
of the same sampled users. The latter still conditions on sample eligibility.
Outside-catalog items are not necessarily newly introduced; many were excluded
by popularity/resource limits. Unknown users receive training popularity;
unsupported KNN scores are excluded, and content without a positive profile
falls back to popularity. No personalized cold-item performance is claimed.

All 34 configurations, per-user metrics, MF learning curves, timings, sample and
data hashes, paired comparisons, and the second-run reproducibility check are
in `reports/results/`. Historical results replay exactly in the recorded
environment (`v1_reproduction.csv`) but are never compared directly as gains
over the newer protocol. Critical tests cover sparse missing-value handling,
known metric examples, chronology, observed-rating optimization and exclusion.
One split/sample, bounded tuning, metadata time ambiguity, parent-ASIN changes,
exposure bias and TF-IDF diversity as a proxy remain threats to validity.
The item binary-weight ablation is unrun. These limitations should accompany
any university report or oral presentation.
