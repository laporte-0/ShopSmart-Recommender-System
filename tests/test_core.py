import unittest
import numpy as np
import pandas as pd
from src.models import Dataset,Popularity,Random,Content,KNN,MatrixFactorization,Hybrid,top_indices
from src.evaluation import ranking_metrics,bootstrap


class MetricsTests(unittest.TestCase):
    def test_known_multiple_relevant(self):
        m=ranking_metrics(['a','x','b'],{'a','b'})
        self.assertEqual(m['Precision@5'],.4)
        self.assertEqual(m['Recall@5'],1)
        self.assertEqual(m['Hit Rate@10'],1)
        self.assertAlmostEqual(m['NDCG@10'],1.5/(1+1/np.log2(3)))
        self.assertEqual(m['MRR@10'],1)

    def test_singleton_and_short_list(self):
        m=ranking_metrics(['x','a'],{'a'})
        self.assertEqual(m['Precision@10'],.1)
        self.assertEqual(m['Recall@10'],m['Hit Rate@10'])
        self.assertAlmostEqual(m['NDCG@10'],1/np.log2(3))
        self.assertEqual(m['MRR@10'],.5)
        self.assertEqual(ranking_metrics([],{'a'})['Hit Rate@10'],0)

    def test_rank_cutoff_and_invalid(self):
        m=ranking_metrics([str(x) for x in range(11)],{'10'})
        self.assertEqual(m['NDCG@10'],0)
        with self.assertRaises(ValueError):ranking_metrics(['a','a'],{'a'})
        with self.assertRaises(ValueError):ranking_metrics([],set())

    def test_bootstrap_paired_identity(self):
        self.assertEqual(bootstrap(np.zeros(10)),[0,0])
        self.assertEqual(bootstrap([0,1,0]),bootstrap([0,1,0]))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame([('u1','a',5),('u1','b',1),('u2','a',4),('u2','c',5),('u3','b',4),('u3','d',5)],columns=['user_id','parent_asin','rating'])
        self.products=pd.DataFrame({'parent_asin':list('abcd'),'title':['phone device','phone cable','phone case','cable case'],'categories':[['tech']]*4})
        self.data=Dataset(self.frame,self.products)

    def test_exclusion_unknown_and_mapping(self):
        mf=MatrixFactorization(self.data,self.frame,factors=2,epochs=3)
        models=[Popularity(self.data),Random(self.data),Content(self.data),KNN(self.data),KNN(self.data,axis='item'),mf]
        models.append(Hybrid(self.data,models[:2],[1,1]))
        for model in models:
            for user in self.data.users:
                ids=[x for x,_ in model.recommend(user)]
                seen=set(self.frame.loc[self.frame.user_id==user,'parent_asin'])
                self.assertFalse(set(ids)&seen)
                self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(model.recommend('unknown'),Popularity(self.data).recommend('unknown'))
        self.assertEqual(self.data.R.nnz,len(self.frame))
        self.assertTrue(1<=mf.predict_rating('unknown','a')<=5)

    def test_factorization_learns_observed_ratings_reproducibly(self):
        # Synthetic training-as-validation fixture checks optimizer behavior only.
        initial=MatrixFactorization(self.data,self.frame,factors=3,lr=.01,reg=.01,epochs=1)
        trained=MatrixFactorization(self.data,self.frame,factors=3,lr=.01,reg=.01,epochs=30)
        repeated=MatrixFactorization(self.data,self.frame,factors=3,lr=.01,reg=.01,epochs=30)
        def loss(model):
            return np.mean([(model.predict_rating(r.user_id,r.parent_asin)-r.rating)**2 for r in self.frame.itertuples()])
        self.assertLess(loss(trained),loss(initial))
        np.testing.assert_array_equal(trained.P,repeated.P)
        self.assertEqual(trained.recommend('u1'),repeated.recommend('u1'))

    def test_missing_values_not_zero_observations(self):
        self.assertEqual(self.data.R[0,2],0)
        self.assertEqual(self.data.mask[0,2],0)
        centered=KNN(self.data,center=True)
        # Mean uses two observed ratings (5+1)/2, not four entries.
        self.assertEqual(centered.means[0],3)
        self.assertEqual(centered.values[0,0],2)

    def test_support_and_shrinkage(self):
        plain=KNN(self.data,k=3)
        shrunk=KNN(self.data,k=3,shrink=10)
        self.assertTrue(np.all(shrunk.neighbors(0).toarray()<=plain.neighbors(0).toarray()))
        none=KNN(self.data,min_common=3)
        self.assertEqual(none.neighbors(0).nnz,0)
        self.assertEqual(none.recommend('u1'),[])

    def test_duplicate_rejection_and_stable_ties(self):
        with self.assertRaises(ValueError):Dataset(pd.concat([self.frame,self.frame.iloc[:1]]),self.products)
        self.assertEqual(top_indices(np.array([2.,2.,-np.inf,np.nan]),10).tolist(),[0,1])

    def test_hybrid_missing_component_scores(self):
        empty=KNN(self.data,min_common=100)
        fusion=Hybrid(self.data,[empty,Popularity(self.data)],[1,1])
        self.assertEqual([x for x,_ in fusion.recommend('u1')],[x for x,_ in Popularity(self.data).recommend('u1')])


class PreparedDataTests(unittest.TestCase):
    def test_temporal_disjointness_and_mappings(self):
        from src.data.prepare import OUT,CUTOFF,END
        if not (OUT/'train.csv').exists():self.skipTest('Prepare raw benchmark first')
        train=pd.read_csv(OUT/'train.csv');v=pd.read_csv(OUT/'validation.csv')
        self.assertTrue((train.timestamp<CUTOFF).all())
        self.assertTrue(((v.timestamp>=CUTOFF)&(v.timestamp<END)).all())
        a=set(zip(train.user_id,train.parent_asin));b=set(zip(v.user_id,v.parent_asin))
        self.assertFalse(a&b)
        self.assertFalse(train.duplicated(['user_id','parent_asin']).any())
        self.assertFalse(v.duplicated(['user_id','parent_asin']).any())
        self.assertTrue(set(v.user_id)<=set(train.user_id))
        self.assertTrue(set(v.parent_asin)<=set(train.parent_asin))
        products=pd.read_json(OUT/'products.jsonl',lines=True)
        self.assertEqual(set(products.parent_asin),set(train.parent_asin))

if __name__=='__main__':unittest.main()
