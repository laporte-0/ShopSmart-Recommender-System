"""Check generated artifacts, dataset fingerprints and saved recommender behavior."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded-only',action='store_true',help='Check published evidence without reconstructed data/model files')
    args=parser.parse_args()
    out=ROOT/'data/processed/v2'
    if not args.recorded_only:
        manifest=json.loads((out/'manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
        bundle=joblib.load(ROOT/'artifacts/recommenders.joblib')
        data=bundle['data']
        for user in data.users[:5]+['unknown-user']:
            seen=set(data.train.loc[data.train.user_id==user,'parent_asin'])
            for name,model in bundle['models'].items():
                recs=model.recommend(user)
                assert not seen & {x for x,_ in recs},(name,user)
                assert len(recs)<=10
                assert all(np.isfinite(score) and asin in data.item_map for asin,score in recs)
                assert recs==model.recommend(user),'Non-deterministic recommendations'
    required=['validation_all_runs.csv','validation_selected.csv','evaluation_sample.csv','per_user_metrics.csv','paired_comparisons.csv','run_manifest.json','v1_reproduction.csv']
    for name in required:assert (ROOT/'reports/results'/name).is_file(),name
    for name in ['validation_hit_rate','validation_ndcg','catalog_coverage','accuracy_coverage','user_knn_sensitivity','v1_v2_protocol_context']:
        assert (ROOT/'reports/figures'/f'{name}.png').is_file(),name
    historical=json.loads((ROOT/'reports/results/v1_preservation_manifest.json').read_text())
    omitted=[]
    for name,digest in historical.items():
        path=ROOT/name
        if not path.is_file() and name.startswith('data/processed/'):
            omitted.append(name)
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    run=json.loads((ROOT/'reports/results/run_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'reports/results/evaluation_sample.csv').read_bytes()).hexdigest()==run['sample_sha256']
    print('PASS: historical fingerprints, six figures, result tables and evaluation-sample hash.')
    if args.recorded_only:
        print('Recorded-evidence mode: reconstructed data/model behavior not checked.')
    else:
        print(f'PASS: {len(bundle["models"])} models × 6 users, stable unseen recommendations and V2 data hashes.')
    if omitted:
        print(f'{len(omitted)} local V1 data files absent; historical data hashes not checked for those files.')

if __name__=='__main__':main()
