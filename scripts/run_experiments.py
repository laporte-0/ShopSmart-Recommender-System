"""Run the shared V2 development benchmark. No test data are read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from src.data.prepare import OUT
from src.models import Dataset,Popularity,Random,Content,KNN,MatrixFactorization,Hybrid
from src.evaluation import evaluate,bootstrap


def main(sample_size=1000):
    results_dir=ROOT/'reports/results';figures=ROOT/'reports/figures'
    results_dir.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(OUT/'train.csv');validation=pd.read_csv(OUT/'validation.csv')
    products=pd.read_json(OUT/'products.jsonl',lines=True)
    data=Dataset(train,products)
    positives=validation[validation.rating>=4]
    eligible=sorted(positives.user_id.unique())
    rng=np.random.default_rng(42)
    selected=sorted(rng.choice(eligible,min(sample_size,len(eligible)),replace=False))
    sample=positives[positives.user_id.isin(selected)].sort_values(['user_id','parent_asin'])
    sample.to_csv(results_dir/'evaluation_sample.csv',index=False)
    truth=sample.groupby('user_id').parent_asin.agg(set).to_dict()
    if not truth:
        raise ValueError('No eligible validation users')
    print('Training',len(train),'ratings;',len(data.users),'users;',len(data.items),'products; evaluation',len(truth),'users',flush=True)
    rows=[];per_users={};models={};params={}
    diversity=Content(data,weighted=False)
    def run(name,constructor,config,family):
        start=time.perf_counter();model=constructor();fit=time.perf_counter()-start
        scores,per=evaluate(model,truth,diversity.X)
        row=dict(model=name,family=family,hyperparameters=json.dumps(config,sort_keys=True),fit_seconds=fit,protocol='raw_global_time_v2_development',**scores)
        if isinstance(model,MatrixFactorization):
            preds=np.array([model.predict_rating(r.user_id,r.parent_asin) for r in validation.itertuples()])
            row['RMSE']=float(np.sqrt(np.mean((preds-validation.rating.to_numpy())**2)))
            row['MAE']=float(np.mean(abs(preds-validation.rating.to_numpy())))
            row['best_epoch']=model.best_epoch
            pd.DataFrame(model.history).to_csv(results_dir/f'{name}_learning_curve.csv',index=False)
        rows.append(row);models[name]=model;per_users[name]=per;params[name]=config
        pd.DataFrame(rows).to_csv(results_dir/'validation_all_runs.csv',index=False)
        print(name, 'HR10',round(scores['Hit Rate@10'],4),'NDCG10',round(scores['NDCG@10'],4),'coverage',round(scores['Coverage'],4),'fit s',round(fit,2),flush=True)
        return model
    run('Popularity',lambda:Popularity(data),{},'Popularity')
    run('Random',lambda:Random(data),{'seed':42},'Random')
    run('Content_V1',lambda:Content(data),{'weighted':False},'Content_V1')
    for name,cfg in [('Content_weighted',dict(weighted=True)),('Content_bigrams',dict(weighted=True,ngrams=2)),('Content_negative',dict(weighted=True,negative=True)),('Content_MMR',dict(weighted=True,rerank=True))]:
        run(name,lambda cfg=cfg:Content(data,**cfg),cfg,'Content_V2')
    for name,cfg,family in [('User_V1',dict(axis='user',k=30), 'User_V1'),('User_V2',dict(axis='user',k=30,support=2,reg=2),'User_V2'),('Item_V1',dict(axis='item',k=30,reg=2),'Item_V1')]:
        run(name,lambda cfg=cfg:KNN(data,**cfg),cfg,family)
    # One-variable sensitivity and a few prespecified joint settings: bounded laptop grid.
    configs=[dict(axis='user',k=k,min_common=2,shrink=10,support=2,reg=2) for k in [10,20,30,50,100]]
    configs += [dict(axis='user',k=30,min_common=m,shrink=s,support=sp,reg=r,center=c) for m,s,sp,r,c in [(1,0,1,2,False),(3,20,2,2,False),(2,10,2,2,True),(2,10,1,0,False)]]
    for j,cfg in enumerate(configs):
        run(f'User_V3_{j}',lambda cfg=cfg:KNN(data,**cfg),cfg,'User_V3')
    configs=[dict(axis='item',k=k,min_common=2,shrink=10,reg=2) for k in [10,30,50,100]]
    configs += [dict(axis='item',k=30,min_common=3,shrink=20,reg=1),dict(axis='item',k=30,min_common=2,shrink=10,normalize_scores=False)]
    for j,cfg in enumerate(configs):
        run(f'Item_V2_{j}',lambda cfg=cfg:KNN(data,**cfg),cfg,'Item_V2')
    for j,cfg in enumerate([dict(factors=8,lr=.01,reg=.05,epochs=40),dict(factors=16,lr=.02,reg=.05,epochs=60),dict(factors=32,lr=.01,reg=.1,epochs=60),dict(factors=16,lr=.01,reg=.1,epochs=40)]):
        run(f'MF_{j}',lambda cfg=cfg:MatrixFactorization(data,validation,**cfg),cfg,'MF')
    def best(family):
        candidates=[r for r in rows if r['family']==family]
        return sorted(candidates,key=lambda r:(-r['NDCG@10'],-r['Hit Rate@10'],r['model']))[0]['model']
    components=[best('Content_V2'),best('Item_V2'),best('MF'),'Popularity']
    for j,weights in enumerate([(1,1,1,0),(2,1,1,0),(1,2,1,0),(1,1,2,0),(1,1,1,1)]):
        cfg=dict(components=components,weights=weights,depth=100,constant=60)
        run(f'Hybrid_{j}',lambda weights=weights:Hybrid(data,[models[x] for x in components],weights),cfg,'Hybrid')
    all_results=pd.DataFrame(rows)
    selected_names=[best(f) for f in all_results.family.unique()]
    selected_results=all_results[all_results.model.isin(selected_names)].sort_values(['NDCG@10','Hit Rate@10'],ascending=False)
    selected_results.to_csv(results_dir/'validation_selected.csv',index=False)
    all_per=pd.concat([p.assign(model=name) for name,p in per_users.items()],ignore_index=True)
    all_per.to_csv(results_dir/'per_user_metrics.csv',index=False)
    winner=selected_results.iloc[0].model
    differences=[]
    for name in selected_results.model:
        a=per_users[winner].set_index('user_id')['NDCG@10']
        b=per_users[name].set_index('user_id')['NDCG@10']
        delta=a-b
        low,high=bootstrap(delta)
        differences.append(dict(winner=winner,comparison=name,mean_ndcg_difference=delta.mean(),ci_low=low,ci_high=high))
    pd.DataFrame(differences).to_csv(results_dir/'paired_comparisons.csv',index=False)
    artifact_names=sorted(set(selected_names+['Popularity']))
    (ROOT/'artifacts').mkdir(exist_ok=True)
    joblib.dump(dict(data=data,models={k:models[k] for k in artifact_names},winner=winner,protocol='raw_global_time_v2_development'),ROOT/'artifacts/recommenders.joblib',compress=3)
    metadata=dict(seed=42,sample_size=len(truth),selection_criterion='Maximum macro validation NDCG@10; tie: Hit Rate@10 then name',winner=winner,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        sample_sha256=hashlib.sha256((results_dir/'evaluation_sample.csv').read_bytes()).hexdigest(),independent_test=False,
        data_manifest=json.loads((OUT/'manifest.json').read_text()))
    (results_dir/'run_manifest.json').write_text(json.dumps(metadata,indent=2))
    for metric,filename in [('Hit Rate@10','validation_hit_rate'),('NDCG@10','validation_ndcg'),('Coverage','catalog_coverage')]:
        fig,ax=plt.subplots(figsize=(9,6));ax.barh(selected_results.model[::-1],selected_results[metric][::-1],color='#287b8e')
        ax.set(xlabel=metric+' (fraction)',title='V2 development validation — '+metric)
        fig.text(.02,.01,'Same users and training catalog; family winners selected on this validation sample.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(figures/f'{filename}.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for r in selected_results.itertuples(index=False):
        row=selected_results[selected_results.model==r.model].iloc[0]
        ax.scatter(row.Coverage,row['NDCG@10']);ax.annotate(row.model,(row.Coverage,row['NDCG@10']),fontsize=8)
    ax.set(xlabel='Catalog coverage (fraction)',ylabel='NDCG@10',title='V2 development: relevance versus coverage')
    fig.tight_layout();fig.savefig(figures/'accuracy_coverage.png',dpi=180);plt.close(fig)
    sensitivity=all_results[all_results.model.isin([f'User_V3_{j}' for j in range(5)])]
    fig,ax=plt.subplots();ax.plot([10,20,30,50,100],sensitivity['NDCG@10'],marker='o')
    ax.set(xlabel='User neighbors (k)',ylabel='Validation NDCG@10',title='User KNN sensitivity (support ≥2, shrinkage=10)')
    fig.tight_layout();fig.savefig(figures/'user_knn_sensitivity.png',dpi=180);plt.close(fig)
    # Historical scores are deliberately isolated; no claimed improvement across protocols.
    historical=pd.read_csv(results_dir/'historical_v1_test_reference.csv')
    historical.to_csv(results_dir/'historical_v1_test_reference.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].barh(historical.Model,historical['Hit Rate@10']);axes[0].set_title('V1 historical examined test (1,000 users)')
    new=selected_results[selected_results.family.isin(['Content_V2','User_V3','Item_V2','MF','Hybrid'])]
    axes[1].barh(new.model,new['Hit Rate@10']);axes[1].set_title('V2 development validation (different protocol)')
    for ax in axes:ax.set_xlabel('Hit Rate@10 (fraction)')
    fig.suptitle('Separate benchmarks — differences are not estimated improvements')
    fig.tight_layout();fig.savefig(figures/'v1_v2_protocol_context.png',dpi=180);plt.close(fig)
    print('Winner:',winner,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sample-size',type=int,default=1000)
    main(parser.parse_args().sample_size)
