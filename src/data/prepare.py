"""Bounded-memory raw-source temporal benchmark construction."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/processed/v2'
CUTOFF = 1577836800000  # 2020-01-01 UTC
END = 1640995200000  # 2022-01-01 UTC


def clean(frame):
    frame = frame.dropna(subset=['user_id', 'parent_asin', 'rating', 'timestamp'])
    return frame[frame.rating.between(1, 5)].copy()


def prepare(catalog_size=1000, min_history=8, max_users=5000):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = ROOT / 'data/raw/Electronics.csv.gz'
    def chunks():
        return pd.read_csv(raw, chunksize=250000)
    counts = Counter()
    total = 0
    for frame in chunks():
        total += len(frame)
        frame = clean(frame)
        counts.update(frame.loc[frame.timestamp < CUTOFF, 'parent_asin'])
    catalog = {x for x, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:catalog_size]}
    print('Pass 1:', total, 'raw rows;', len(catalog), 'training-era products', flush=True)
    histories = []
    for frame in chunks():
        frame = clean(frame)
        histories.append(frame[(frame.timestamp < CUTOFF) & frame.parent_asin.isin(catalog)])
    pre = pd.concat(histories).sort_values(['timestamp','parent_asin']).drop_duplicates(['user_id','parent_asin'], keep='first')
    activity = pre.groupby('user_id').size()
    eligible = sorted(activity[activity >= min_history].index,
                      key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:max_users]
    users = set(eligible)
    train = pre[pre.user_id.isin(users)].copy()
    del histories, pre
    held = []
    for frame in chunks():
        frame = clean(frame)
        held.append(frame[(frame.timestamp >= CUTOFF) & (frame.timestamp < END) & frame.user_id.isin(users)])
    validation = pd.concat(held).sort_values(['timestamp','parent_asin']).drop_duplicates(['user_id','parent_asin'], keep='first')
    seen = pd.MultiIndex.from_frame(train[['user_id','parent_asin']])
    validation = validation[~pd.MultiIndex.from_frame(validation[['user_id','parent_asin']]).isin(seen)]
    # Candidate universe is restricted to items observed in the selected users' training.
    catalog = set(train.parent_asin)
    for name, frame in [('train',train), ('validation_all',validation), ('validation', validation[validation.parent_asin.isin(catalog)])]:
        frame.sort_values(['user_id','timestamp','parent_asin']).to_csv(OUT / f'{name}.csv', index=False)
    products = {}
    with gzip.open(ROOT / 'data/raw/meta_Electronics.jsonl.gz','rt') as handle:
        for line in handle:
            p = json.loads(line)
            asin = p.get('parent_asin')
            if asin in catalog and asin not in products:
                products[asin] = {k:p.get(k) for k in ['parent_asin','title','main_category','categories','features','description','details','price']}
            if len(products) == len(catalog):
                break
    with (OUT / 'products.jsonl').open('w') as handle:
        for asin in sorted(catalog):
            handle.write(json.dumps(products.get(asin, {'parent_asin':asin,'title':'Metadata unavailable'}))+'\n')
    manifest = dict(protocol='raw_global_time_v2_development', raw_rows=total, cutoff_utc='2020-01-01', validation_end_utc='2022-01-01',
        catalog_requested=catalog_size, catalog_size=len(catalog), min_history=min_history, max_users=max_users,
        users=len(users), train_rows=len(train), validation_rows=len(validation[validation.parent_asin.isin(catalog)]),
        validation_all_rows=len(validation), outside_catalog_rows=int((~validation.parent_asin.isin(catalog)).sum()),
        metadata_missing=len(catalog)-len(products), independent_test=False,
        selection='Pre-cutoff event-count top catalog, unique-pair user threshold, SHA256 user cap; earliest observed rating per pair')
    manifest['sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if p.suffix in ['.csv','.jsonl']}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--catalog-size',type=int,default=1000)
    parser.add_argument('--min-history',type=int,default=8)
    parser.add_argument('--max-users',type=int,default=5000)
    args=parser.parse_args()
    prepare(args.catalog_size,args.min_history,args.max_users)
