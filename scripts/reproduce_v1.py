"""Replay historical notebook recommenders without executing notebook writes.

Only import/setup and the original function definitions are executed. Output is
written separately; original notebooks and results remain unchanged.
"""
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    processed=ROOT/'data/processed'
    train=pd.read_csv(processed/'train.csv')
    products=pd.read_json(processed/'products_preprocessed.jsonl',lines=True)
    users=train.user_id.unique();items=products.parent_asin.unique()
    um={u:i for i,u in enumerate(users)};im={p:i for i,p in enumerate(items)}
    matrix=csr_matrix((train.rating.to_numpy(),(train.user_id.map(um),train.parent_asin.map(im))),shape=(len(users),len(items)))
    user_knn=NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=31).fit(matrix)
    item_matrix=matrix.T.tocsr()
    item_knn=NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=31).fit(item_matrix)
    distances,neighbors=item_knn.kneighbors(item_matrix,n_neighbors=31)
    tfidf=TfidfVectorizer(stop_words='english',max_features=10000,min_df=2,max_df=.95)
    tfidf_matrix=tfidf.fit_transform(products.content.fillna(''))
    namespace=dict(pd=pd,np=np,train=train,products=products,user_ids=users,product_ids=items,
        user_to_index=um,product_to_index=im,user_item_matrix=matrix,user_knn=user_knn,
        item_user_matrix=item_matrix,item_knn=item_knn,item_distances=distances,item_neighbors=neighbors,
        tfidf_matrix=tfidf_matrix,cosine_similarity=cosine_similarity,
        product_index=pd.Series(products.index,index=products.parent_asin),
        user_seen=train.groupby('user_id').parent_asin.agg(set).to_dict(),
        popular_products=train.groupby('parent_asin').size().sort_values(ascending=False).index.tolist())
    for filename in ['03_Content_Based_Filtering.ipynb','04_Collaborative_Filtering.ipynb']:
        notebook=json.loads((ROOT/'notebooks'/filename).read_text())
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            source=''.join(cell['source'])
            try: tree=ast.parse(source)
            except SyntaxError:continue
            for node in tree.body:
                if isinstance(node,ast.FunctionDef):
                    exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),namespace)
    sample_path=processed/'test_users_sample.csv'
    if sample_path.is_file():
        sample=pd.read_csv(sample_path)
    else:
        # Reconstruct the original shared sample before notebooks 03/04 need it.
        # Existing historical data are never overwritten.
        test=pd.read_csv(processed/'test.csv')
        positive_users=set(train.loc[train.rating>=4,'user_id'])
        eligible=test[(test.rating>=4)&test.user_id.isin(positive_users)]
        sample=eligible.sample(n=min(1000,len(eligible)),random_state=123)[['user_id','parent_asin','rating']]
        sample.to_csv(sample_path,index=False)
    models={'Content-Based TF-IDF':'recommend_for_user','Popularity Baseline':'recommend_popular','User-Based KNN V2':'recommend_user_based_v2','Item-Based KNN':'recommend_item_based_fast'}
    records=[]
    for name,function in models.items():
        hits=[];union=set()
        for row in sample.itertuples():
            result=namespace[function](row.user_id,top_n=10)
            ids=result.parent_asin.tolist() if isinstance(result,pd.DataFrame) else result
            hits.append(row.parent_asin in ids);union.update(ids)
        records.append({'Model':name,'Precision@10':np.mean(hits)/10,'Hit Rate@10':np.mean(hits),'Coverage':len(union)/len(items)})
        print(records[-1],flush=True)
    output=pd.DataFrame(records)
    historical=pd.read_csv(processed/'final_model_results.csv')
    comparison=output.merge(historical,on='Model',suffixes=('_replay','_historical'))
    (ROOT/'reports/results').mkdir(parents=True,exist_ok=True)
    comparison.to_csv(ROOT/'reports/results/v1_reproduction.csv',index=False)

if __name__=='__main__':main()
