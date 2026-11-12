"""Sparse neighborhood models and observed-rating factorization.

All scores are ranking utilities unless explicitly returned by predict_rating.
Unobserved entries never enter the explicit-rating loss.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def top_indices(scores, k):
    """Stable descending finite-score ranking; item index breaks ties."""
    valid = np.flatnonzero(np.isfinite(scores))
    return valid[np.argsort(-scores[valid], kind='stable')[:k]]


@dataclass
class Dataset:
    train: pd.DataFrame
    products: pd.DataFrame

    def __post_init__(self):
        if self.train.duplicated(['user_id','parent_asin']).any():
            raise ValueError('Duplicate training pairs')
        if self.products.parent_asin.duplicated().any():
            raise ValueError('Duplicate metadata identifiers')
        self.users = sorted(self.train.user_id.unique())
        self.items = sorted(self.train.parent_asin.unique())
        self.user_map = {x:i for i,x in enumerate(self.users)}
        self.item_map = {x:i for i,x in enumerate(self.items)}
        self.products = self.products.set_index('parent_asin').reindex(self.items).fillna('').reset_index()
        self.u = self.train.user_id.map(self.user_map).to_numpy()
        self.i = self.train.parent_asin.map(self.item_map).to_numpy()
        self.y = self.train.rating.to_numpy(dtype=float)
        self.R = sparse.csr_matrix((self.y,(self.u,self.i)),shape=(len(self.users),len(self.items)))
        self.mask = self.R.copy(); self.mask.data[:] = 1
        self.positive = self.R.copy(); self.positive.data = np.maximum(self.positive.data-3,0); self.positive.eliminate_zeros()
        self.popularity = np.asarray(self.mask.sum(axis=0)).ravel()


class Model:
    def __init__(self, data):
        self.data = data

    def scores(self, user):
        u = self.data.user_map.get(user)
        values = self.data.popularity.astype(float).copy() if u is None else np.asarray(self.raw_scores(u), dtype=float).ravel().copy()
        if u is not None:
            values[self.data.R[u].indices] = -np.inf
        return values

    def recommend(self, user, k=10):
        values = self.scores(user)
        ids = top_indices(values,k)
        return [(self.data.items[i],float(values[i])) for i in ids]


class Popularity(Model):
    def raw_scores(self,u):
        return self.data.popularity


class Random(Model):
    def raw_scores(self,u):
        seed = int(hashlib.sha256(self.data.users[u].encode()).hexdigest()[:8],16)
        return np.random.default_rng(seed+42).random(len(self.data.items))


def text(value):
    if isinstance(value, dict):
        return ' '.join(f'{k} {v}' for k,v in value.items())
    if isinstance(value,list):
        return ' '.join(map(str,value))
    return str(value or '')


class Content(Model):
    def __init__(self,data, weighted=False, ngrams=1, negative=False, rerank=False):
        super().__init__(data)
        self.rerank = rerank
        p = data.products
        self.vectorizers=[]
        blocks=[]
        fields = [('title',2),('categories',1.5),('features',1),('description',0.5),('details',0.5)] if weighted else [('content',1)]
        for field, weight in fields:
            docs = (p.apply(lambda row:' '.join(text(row.get(k,'')) for k in ['title','main_category','categories','features','description']),axis=1)
                    if field=='content' else p[field].map(text) if field in p else pd.Series(['']*len(p)))
            vec = TfidfVectorizer(stop_words='english',min_df=2,max_df=.95,max_features=10000,ngram_range=(1,ngrams),sublinear_tf=weighted)
            try:
                block = vec.fit_transform(docs)*weight
            except ValueError:
                continue
            self.vectorizers.append(vec); blocks.append(block)
        if not blocks:
            self.X = sparse.eye(len(p),format='csr')
        else:
            self.X = normalize(sparse.hstack(blocks,format='csr'))
        preferences = data.R.copy()
        preferences.data = (np.where(preferences.data>=4,preferences.data-3,np.where(preferences.data<=2,-.5,0))
                            if negative else np.maximum(preferences.data-3,0) if weighted else (preferences.data>=4).astype(float))
        self.profiles = normalize(preferences @ self.X)

    def raw_scores(self,u):
        if self.profiles[u].nnz == 0:
            return self.data.popularity
        return (self.profiles[u] @ self.X.T).toarray().ravel()

    def scores(self,user):
        values = super().scores(user)
        if self.rerank and user in self.data.user_map:
            # Greedy maximal marginal relevance over the best 100 candidates.
            pool = list(top_indices(values,100)); chosen=[]; output=np.full_like(values,-np.inf)
            while pool:
                sims = (self.X[pool] @ self.X[chosen].T).toarray().max(axis=1) if chosen else np.zeros(len(pool))
                j = int(np.argmax(.8*values[pool]-.2*sims)); item=pool.pop(j)
                output[item] = 1/(1+len(chosen)); chosen.append(item)
            return output
        return values


class KNN(Model):
    def __init__(self,data,axis='user',k=30,min_common=1,shrink=0,support=1,reg=0,center=False,normalize_scores=True):
        super().__init__(data)
        self.axis,self.k,self.min_common,self.shrink = axis,k,min_common,shrink
        self.support,self.reg,self.center,self.normalize_scores = support,reg,center,normalize_scores
        self.means = np.asarray(data.R.sum(axis=1)).ravel()/np.diff(data.R.indptr)
        self.values=data.R.copy()
        if center:
            self.values.data -= np.repeat(self.means,np.diff(data.R.indptr))
        basis = self.values if axis=='user' else data.R.T.tocsr()
        mask = data.mask if axis=='user' else data.mask.T.tocsr()
        self.basis=normalize(basis); self.evidence=mask
        if axis=='item':
            rows=[]
            for i in range(len(data.items)):
                rows.append(self.neighbors(i))
            self.graph=sparse.vstack(rows,format='csr')

    def neighbors(self,index):
        sim = (self.basis[index] @ self.basis.T).toarray().ravel()
        shared = (self.evidence[index] @ self.evidence.T).toarray().ravel()
        sim *= np.divide(shared,shared+self.shrink,out=np.zeros_like(shared),where=(shared+self.shrink)>0)
        sim[(shared<self.min_common)|(sim<=0)] = -np.inf
        sim[index]=-np.inf
        ids=top_indices(sim,self.k)
        return sparse.csr_matrix((sim[ids],(np.zeros(len(ids),dtype=int),ids)),shape=(1,len(sim)))

    def raw_scores(self,u):
        if self.axis=='user':
            weights=self.neighbors(u)
            num=(weights @ self.values).toarray().ravel()
            den=(weights @ self.data.mask).toarray().ravel()
            active=weights.copy();active.data[:]=1
            evidence=(active @ self.data.mask).toarray().ravel()
        else:
            pref=self.data.positive[u]
            num=(pref @ self.graph).toarray().ravel()
            active=pref.copy();active.data[:]=1
            den=(active @ self.graph).toarray().ravel()
            binary=self.graph.copy();binary.data[:]=1
            evidence=(active @ binary).toarray().ravel()
        score = num/(den+self.reg+1e-12) if self.normalize_scores else num
        if self.center and self.axis=='user':
            score+=self.means[u]
        score[(den<=0)|(evidence<self.support)] = -np.inf
        return score


class MatrixFactorization(Model):
    """Biased MF, minibatch SGD on observed ratings, validation RMSE early stopping."""
    def __init__(self,data,validation,factors=16,lr=.02,reg=.05,epochs=40,seed=42):
        super().__init__(data)
        rng=np.random.default_rng(seed)
        self.P=rng.normal(0,.1,(len(data.users),factors)); self.Q=rng.normal(0,.1,(len(data.items),factors))
        self.bu=np.zeros(len(data.users));self.bi=np.zeros(len(data.items));self.mean=data.y.mean()
        v=validation[validation.user_id.isin(data.user_map)&validation.parent_asin.isin(data.item_map)]
        vu=v.user_id.map(data.user_map).to_numpy();vi=v.parent_asin.map(data.item_map).to_numpy();vy=v.rating.to_numpy()
        best=np.inf; patience=0; self.history=[]; best_state=None
        for epoch in range(epochs):
            order=rng.permutation(len(data.y))
            for start in range(0,len(order),256):
                ix=order[start:start+256];u=data.u[ix];i=data.i[ix]
                p=self.P[u].copy();q=self.Q[i].copy()
                err=data.y[ix]-(self.mean+self.bu[u]+self.bi[i]+np.sum(p*q,axis=1))
                np.add.at(self.bu,u,lr*(err-reg*self.bu[u]))
                np.add.at(self.bi,i,lr*(err-reg*self.bi[i]))
                np.add.at(self.P,u,lr*(err[:,None]*q-reg*p))
                np.add.at(self.Q,i,lr*(err[:,None]*p-reg*q))
            prediction=np.clip(self.mean+self.bu[vu]+self.bi[vi]+np.sum(self.P[vu]*self.Q[vi],axis=1),1,5)
            rmse=float(np.sqrt(np.mean((prediction-vy)**2)))
            self.history.append({'epoch':epoch+1,'validation_rmse':rmse})
            if rmse < best-1e-5:
                best=rmse;patience=0;best_state=[a.copy() for a in [self.P,self.Q,self.bu,self.bi]];self.best_epoch=epoch+1
            else:
                patience+=1
            if patience>=5:
                break
        if best_state is not None:
            self.P,self.Q,self.bu,self.bi=best_state

    def raw_scores(self,u):
        return self.mean+self.bu[u]+self.bi+self.Q @ self.P[u]

    def predict_rating(self,user,item):
        u=self.data.user_map.get(user);i=self.data.item_map.get(item)
        pred=self.mean+(self.bu[u] if u is not None else 0)+(self.bi[i] if i is not None else 0)
        if u is not None and i is not None:
            pred+=self.P[u] @ self.Q[i]
        return float(np.clip(pred,1,5))


class Hybrid(Model):
    """Weighted reciprocal rank fusion; absent/nonfinite scores contribute zero."""
    def __init__(self,data,components,weights,depth=100,constant=60):
        super().__init__(data)
        self.components,self.weights,self.depth,self.constant=components,weights,depth,constant

    def raw_scores(self,u):
        scores=np.zeros(len(self.data.items));user=self.data.users[u]
        for model,weight in zip(self.components,self.weights):
            ids=top_indices(model.scores(user),self.depth)
            scores[ids]+=weight/(self.constant+np.arange(1,len(ids)+1))
        # Explicit popularity fallback only where no component contributes.
        absent=scores==0
        scores[absent]=1e-12*self.data.popularity[absent]/max(1,self.data.popularity.max())
        return scores
