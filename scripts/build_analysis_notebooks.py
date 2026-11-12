"""Create compact analysis notebooks reusing measured outputs and saved models."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
setup="""from pathlib import Path
import sys
import pandas as pd
import joblib
ROOT = Path.cwd() if (Path.cwd() / 'reports').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
results = pd.read_csv(ROOT / 'reports/results/validation_all_runs.csv')
bundle = joblib.load(ROOT / 'artifacts/recommenders.joblib')
"""
notebooks=[
('06_Model_Optimization',"""# Model optimization: V2 development validation

See `docs/evaluation_protocol.md` for chronology, eligibility and caveats.
The existing V1 notebooks are unchanged. This notebook analyzes executed runs;
rerun `python scripts/run_experiments.py` to refit models. NDCG@10 selects family
winners; lower accuracy variants are retained as negative findings.
""",[
"display(results[results.family.isin(['Content_V1','Content_V2','User_V1','User_V2','User_V3','Item_V1','Item_V2'])][['model','hyperparameters','Hit Rate@10','NDCG@10','Coverage','Diversity']])",
"from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / 'reports/figures/user_knn_sensitivity.png')))"
]),
('07_Matrix_Factorization',"""# Observed-rating matrix factorization

Biased latent factors use minibatch SGD only on observed training ratings.
Missing entries never become zero targets. Validation RMSE early stopping uses
patience 5; maximum 40/60 epochs. NDCG@10 selects the model within the MF family.
Raw ranking scores and clipped rating estimates have different purposes.
""",[
"display(results[results.family.eq('MF')][['model','hyperparameters','best_epoch','RMSE','MAE','Hit Rate@10','NDCG@10']])",
"for path in sorted((ROOT / 'reports/results').glob('MF_*_learning_curve.csv')):\n    print(path.name)\n    display(pd.read_csv(path))"
]),
('08_Hybrid_Recommender',"""# Reciprocal rank fusion and recommendation demonstration

Each component contributes weight/(60+rank) over its top 100 unseen candidates.
Absent scores contribute zero; popularity fills candidates with no contribution.
Weights use validation only. The hybrid is compared against each component and
is not assumed to win. All results are development estimates, not independent tests.
""",[
"display(results[results.family.eq('Hybrid')][['model','hyperparameters','Hit Rate@10','NDCG@10','Coverage','Diversity']])\ndisplay(pd.read_csv(ROOT / 'reports/results/validation_selected.csv'))",
"data=bundle['data']\nuser=data.users[0]\nprint('User:', user)\ndisplay(data.train[data.train.user_id.eq(user)].merge(data.products,on='parent_asin')[['title','rating']].head())\nfor name in [bundle['winner'], 'Popularity']:\n    recs=bundle['models'][name].recommend(user)\n    frame=pd.DataFrame(recs,columns=['parent_asin','ranking_score']).merge(data.products,on='parent_asin')\n    print(name, 'ranking scores, not calibrated ratings')\n    display(frame[['title','categories','ranking_score']])"
])]
for name,intro,cells in notebooks:
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(setup)]+[nbf.v4.new_code_cell(c) for c in cells])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb')
