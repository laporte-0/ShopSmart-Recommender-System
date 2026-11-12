"""Deterministic TF-IDF product retrieval, independent of personalized ML models."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import html
import json
from pathlib import Path
import pickle
import re
import tempfile

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
CACHE_VERSION = 4


def _text(value):
    """Flatten catalog text without interpreting it as instructions or code."""
    if isinstance(value, list):
        return ' '.join(filter(None, (_text(item) for item in value)))
    if isinstance(value, dict):
        return ' '.join(f'{_text(k)} {_text(v)}' for k, v in value.items())
    if not isinstance(value, str):
        return ''
    value = html.unescape(re.sub(r'<[^>]*>', ' ', value))
    return ' '.join(value.split())


@dataclass(frozen=True)
class SearchHit:
    product_id: str
    title: str
    categories: str
    features: str
    description: str
    retrieval_score: float | None = None

    @property
    def document(self):
        return (f'Title: {self.title}\nCategories: {self.categories}\n'
                f'Features: {self.features}\nDescription: {self.description}')

    def context(self):
        # Bounded evidence budget; quotes are validated against this exact text.
        document = self.document[:1600]
        evidence_options = []
        for line in document.splitlines():
            if ': ' not in line or not line.split(': ', 1)[1].strip():
                continue
            value = line.split(': ', 1)[1]
            evidence_options.append(value[:300].rsplit(' ', 1)[0] if len(value) > 300 else value)
        return {
            'product_id': self.product_id,
            'title': self.title,
            'catalog_document': document,
            'evidence_options': evidence_options,
        }


def load_catalog(path=None):
    """Load existing JSONL metadata; never download data implicitly."""
    candidates = [Path(path)] if path is not None else [
        ROOT / 'data/processed/v2/products.jsonl',
        ROOT / 'data/processed/products_preprocessed.jsonl',
        ROOT / 'data/processed/products_shopsmart.jsonl',
    ]
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding='utf-8') as handle:
                return [json.loads(line) for line in handle if line.strip()]
    raise FileNotFoundError(
        'Product catalog missing. Run python -m src.data.prepare or pass '
        '--catalog PATH to an existing Amazon product JSONL file.'
    )


class ProductIndex:
    """L2-normalized TF-IDF search, with stable ID ties and content-hashed caching.

    Missing/invalid IDs or titles are excluded. For duplicate IDs the first valid
    record wins, preserving an explicit deterministic mapping. Prices and review
    aggregate statistics are neither indexed nor passed to generation.
    """

    def __init__(self, records, cache_path=None):
        products = {}
        self.excluded_records = 0
        for record in records:
            if not isinstance(record, dict):
                self.excluded_records += 1
                continue
            product_id = record.get('parent_asin')
            title = _text(record.get('title'))
            if (not isinstance(product_id, str)
                    or not re.fullmatch(r'[A-Z0-9]{10}', product_id)
                    or not title or title.lower() in {'nan', 'none', 'metadata unavailable'}
                    or product_id in products):
                self.excluded_records += 1
                continue
            products[product_id] = SearchHit(
                product_id=product_id, title=title,
                categories=_text(record.get('categories')) or _text(record.get('main_category')),
                features=_text(record.get('features')),
                description=_text(record.get('description')),
            )
        if not products:
            raise ValueError('No valid catalog records with product IDs and titles')
        self.products = [products[key] for key in sorted(products)]
        self.by_id = {product.product_id: product for product in self.products}
        documents = [product.document for product in self.products]
        fingerprint_input = dict(
            version=CACHE_VERSION, sklearn=sklearn.__version__,
            products=[asdict(product) for product in self.products],
        )
        self.fingerprint = sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
        self.cache_hit = False
        cache_path = Path(cache_path) if cache_path is not None else None
        if cache_path is not None and cache_path.is_file():
            try:
                cached = joblib.load(cache_path)
                if (cached['fingerprint'] == self.fingerprint
                        and cached['matrix'].shape[0] == len(self.products)):
                    self.vectorizer, self.matrix = cached['vectorizer'], cached['matrix']
                    self.cache_hit = True
                    return
            except (OSError, ValueError, EOFError, KeyError, TypeError,
                    ImportError, AttributeError, IndexError, pickle.UnpicklingError):
                pass  # Rebuild a damaged or incompatible local cache.
        self.vectorizer = TfidfVectorizer(
            stop_words='english', sublinear_tf=True, ngram_range=(1, 2),
            max_features=30000, norm='l2',
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
            try:
                joblib.dump(dict(fingerprint=self.fingerprint, vectorizer=self.vectorizer,
                                 matrix=self.matrix), temporary, compress=3)
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)

    def search(self, question, top_k=5):
        if top_k < 1 or top_k > 10:
            raise ValueError('top_k must be between 1 and 10')
        if not isinstance(question, str) or not question.strip():
            return []
        vector = self.vectorizer.transform([question])
        if vector.nnz == 0:
            return []
        scores = (self.matrix @ vector.T).toarray().ravel()
        valid = np.flatnonzero(scores > 0)
        order = valid[np.argsort(-scores[valid], kind='stable')[:top_k]]
        return [SearchHit(**{**asdict(self.products[i]), 'retrieval_score': float(scores[i])})
                for i in order]

    def lookup(self, product_ids):
        """Metadata for ML-selected IDs in original recommendation order."""
        return [self.by_id[product_id] for product_id in dict.fromkeys(product_ids)
                if product_id in self.by_id]
