"""Macro user-level binary relevance metrics, fixed requested-length precision."""
import time
import numpy as np
import pandas as pd


def ranking_metrics(recommendations,relevant):
    if len(recommendations)!=len(set(recommendations)):
        raise ValueError('Duplicate recommendations')
    relevant=set(relevant)
    if not relevant:
        raise ValueError('Ranking evaluation requires at least one relevant item')
    out={}
    for k in [5,10]:
        hits=sum(x in relevant for x in recommendations[:k])
        out[f'Precision@{k}']=hits/k
        out[f'Recall@{k}']=hits/len(relevant)
        out[f'Hit Rate@{k}']=float(hits>0)
    rel=np.array([x in relevant for x in recommendations[:10]],dtype=float)
    out['NDCG@10']=float((rel/np.log2(np.arange(len(rel))+2)).sum()/sum(1/np.log2(np.arange(min(10,len(relevant)))+2)))
    ranks=np.flatnonzero(rel)
    out['MRR@10']=float(1/(ranks[0]+1)) if len(ranks) else 0.
    out['List length']=len(recommendations[:10])
    return out


def bootstrap(values,seed=42,iterations=1000):
    values=np.asarray(values)
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(iterations)])
    return np.quantile(means,[.025,.975]).tolist()


def evaluate(model,truth,content_matrix=None):
    records=[];union=set();start=time.perf_counter()
    for user,relevant in truth.items():
        recommendations=model.recommend(user,10)
        ids=[x for x,_ in recommendations]
        u=model.data.user_map.get(user)
        seen={model.data.items[i] for i in model.data.R[u].indices} if u is not None else set()
        if set(ids) & seen:
            raise AssertionError('Seen item recommendation')
        if not set(ids)<=set(model.data.items):
            raise AssertionError('Out-of-catalog recommendation')
        values=ranking_metrics(ids,relevant)
        if content_matrix is not None and len(ids)>1:
            ix=[model.data.item_map[x] for x in ids]
            sim=(content_matrix[ix] @ content_matrix[ix].T).toarray()
            values['Diversity']=float(1-sim[np.triu_indices(len(ids),1)].mean())
        else:
            values['Diversity']=np.nan
        values['user_id']=user
        records.append(values);union.update(ids)
    elapsed=time.perf_counter()-start
    per_user=pd.DataFrame(records)
    summary=per_user.drop(columns='user_id').mean().to_dict()
    summary['Coverage']=len(union)/len(model.data.items)
    summary['inference_seconds']=elapsed
    summary['users']=len(truth)
    summary['HR10_ci_low'],summary['HR10_ci_high']=bootstrap(per_user['Hit Rate@10'])
    summary['NDCG10_ci_low'],summary['NDCG10_ci_high']=bootstrap(per_user['NDCG@10'])
    return summary,per_user
