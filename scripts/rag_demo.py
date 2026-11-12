"""Natural-language product discovery and explanations for ML recommendations."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import ProductIndex, OllamaClient, answer_question, load_catalog
from src.rag.service import OllamaUnavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', help='Product request; prompts interactively when omitted')
    parser.add_argument('--catalog', type=Path, help='Existing product metadata JSONL')
    parser.add_argument('--user', help='Explain existing ML recommendations for this user')
    parser.add_argument('--recommender', help='Saved model name; default: validation NDCG winner')
    parser.add_argument('--top-k', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--model', help='Downloaded Ollama model; alternatively OLLAMA_MODEL')
    parser.add_argument('--host', help='Ollama server URL; alternatively OLLAMA_HOST')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retrieval-only', action='store_true', help='Skip the Ollama call')
    parser.add_argument('--check-ollama', action='store_true', help='Check server/model availability and exit')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--json', action='store_true', help='Print structured results instead of prose')
    args = parser.parse_args()
    try:
        client = OllamaClient(args.host, args.model, args.timeout)
        if args.check_ollama:
            try:
                print(json.dumps({'status': 'available', 'models': client.available_models()}, indent=2))
            except OllamaUnavailable as exc:
                print(json.dumps({'status': 'unavailable', 'reason': str(exc)}))
                return 1
            return 0
        query = args.query
        if not query:
            if args.user:
                query = 'Explain these recommendations using their catalog descriptions and state missing information.'
            else:
                try:
                    query = input('Describe the electronics you are looking for: ').strip()
                except EOFError:
                    parser.error('Provide --query TEXT or type a request interactively')
        index = ProductIndex(load_catalog(args.catalog),
                             cache_path=None if args.no_cache else ROOT / '.cache/rag/index.joblib')
        recommendation_info = None
        product_ids = None
        if args.user:
            import joblib
            bundle_path = ROOT / 'artifacts/recommenders.joblib'
            if not bundle_path.is_file():
                parser.error('Saved ML bundle missing; run scripts/run_experiments.py first')
            bundle = joblib.load(bundle_path)
            name = args.recommender or bundle['winner']
            if name not in bundle['models']:
                parser.error(f'Unknown recommender {name}; choose from {list(bundle["models"])}')
            recommendations = bundle['models'][name].recommend(args.user, args.top_k)
            product_ids = [product_id for product_id, _ in recommendations]
            recommendation_info = {
                'model': name, 'unknown_user_popularity_fallback': args.user not in bundle['data'].user_map,
                'items': [{'product_id': product_id, 'ml_ranking_score': score}
                          for product_id, score in recommendations],
                'missing_metadata_ids': [product_id for product_id in product_ids if product_id not in index.by_id],
            }
        result = answer_question(index, query, top_k=args.top_k, product_ids=product_ids,
                                 client=client, retrieval_only=args.retrieval_only)
        if args.json:
            print(json.dumps({**asdict(result), 'question': query, 'recommendations': recommendation_info,
                              'index_cache_hit': index.cache_hit}, indent=2, ensure_ascii=False))
            return 0
        print(f'\nMode: {result.mode} | Status: {result.status}')
        if recommendation_info:
            print('ML recommender:', recommendation_info['model'])
            if recommendation_info['unknown_user_popularity_fallback']:
                print('Unknown user: ML recommendations use training-popularity fallback.')
            for item in recommendation_info['items']:
                print(f'  [{item["product_id"]}] ML ranking score: {item["ml_ranking_score"]:.6f}')
            if recommendation_info['missing_metadata_ids']:
                print('Metadata unavailable for:', ', '.join(recommendation_info['missing_metadata_ids']))
        for hit in result.hits:
            print(f'\n[{hit.product_id}] {hit.title}')
            print('Categories:', hit.categories or 'Not provided')
            if hit.retrieval_score is not None:
                print(f'TF-IDF cosine similarity: {hit.retrieval_score:.4f} (not a rating)')
            print('Catalog features:', hit.features[:500] or 'Not provided')
            print('Catalog description:', hit.description[:350] or 'Not provided')
        if result.answer:
            print(f'\nLLM-generated explanation ({result.model}):\n{result.answer}')
        else:
            print('\n' + (result.reason or 'No generated explanation.'))
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
