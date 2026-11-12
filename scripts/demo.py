"""Compare saved recommenders from the repository root."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user',help='Known user ID; unknown IDs use training popularity')
    parser.add_argument('--models',nargs='+',help='Saved model names; default winner and popularity')
    parser.add_argument('--list-users',action='store_true')
    args=parser.parse_args()
    bundle=joblib.load(ROOT/'artifacts/recommenders.joblib')
    data=bundle['data'];models=bundle['models']
    if args.list_users:
        print('\n'.join(data.users[:20]));return
    user=args.user or data.users[0]
    print('User:',user,'| Protocol:',bundle['protocol'])
    history=data.train[data.train.user_id==user].merge(data.products,on='parent_asin')
    if history.empty:
        print('Unknown user: training-popularity fallback for all models.')
    else:
        print('\nPreviously rated products (sample):')
        print(history[['parent_asin','title','rating']].head(8).to_string(index=False))
    names=args.models or list(dict.fromkeys([bundle['winner'],'Popularity']))
    for name in names:
        if name not in models:
            parser.error(f'Unknown model {name}. Available: {list(models)}')
        recs=models[name].recommend(user)
        assert not set(x for x,_ in recs)&set(history.parent_asin)
        frame=pd.DataFrame(recs,columns=['parent_asin','ranking_score']).merge(data.products,on='parent_asin')
        print('\n'+name+' — ranking scores, not calibrated ratings:')
        print(frame[['parent_asin','title','categories','ranking_score']].to_string(index=False))

if __name__=='__main__':
    main()
