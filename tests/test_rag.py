"""RAG tests use a synthetic catalog and mocked HTTP; no live model required."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from src.rag import ProductIndex, OllamaClient, answer_question, build_prompt
from src.rag.service import OllamaUnavailable


CATALOG = [
    {'parent_asin': 'B000000001', 'title': 'Office monitor', 'categories': ['Electronics', 'Monitors'],
     'features': ['Adjustable office monitor stand'], 'description': ['Monitor for desktop work']},
    {'parent_asin': 'B000000002', 'title': 'Wireless headphones', 'categories': ['Audio'],
     'features': ['Wireless audio connection'], 'description': ['Headphones for listening']},
    {'parent_asin': 'B000000003', 'title': 'USB cable', 'features': ['USB charging cable']},
]


def response(value):
    handle = MagicMock()
    handle.__enter__.return_value = io.BytesIO(json.dumps(value).encode())
    return handle


def generated(product_id='B000000001', quote='Office monitor'):
    return json.dumps({'products': [{
        'product_id': product_id, 'evidence_quote': quote,
        'inferred_suitability': 'The office description suggests a possible fit for desk work.',
        'missing_information': 'Refresh rate is not provided.',
    }]})


class RetrievalTests(unittest.TestCase):
    def test_ordering_scores_and_mapping(self):
        index = ProductIndex(CATALOG)
        hits = index.search('office monitor', 3)
        self.assertEqual(hits[0].product_id, 'B000000001')
        self.assertTrue(all(a.retrieval_score >= b.retrieval_score for a, b in zip(hits, hits[1:])))
        self.assertTrue(all(0 < hit.retrieval_score <= 1.000001 for hit in hits))
        self.assertEqual(index.search('wireless headphones')[0].title, 'Wireless headphones')
        self.assertEqual(index.lookup(['missing', 'B000000003', 'B000000001', 'B000000003']),
                         [index.by_id['B000000003'], index.by_id['B000000001']])

    def test_invalid_and_duplicate_records(self):
        invalid = [None, {}, {'parent_asin': None, 'title': 'Bad'},
                   {'parent_asin': 'not an asin', 'title': 'Bad'},
                   {'parent_asin': 'B000000004', 'title': None},
                   {'parent_asin': 'B000000005', 'title': 'Metadata unavailable'}, CATALOG[0]]
        index = ProductIndex(CATALOG + invalid)
        self.assertEqual(len(index.products), 3)
        self.assertEqual(index.excluded_records, len(invalid))
        with self.assertRaises(ValueError):
            ProductIndex([{}])

    def test_no_evidence_and_invalid_k(self):
        index = ProductIndex(CATALOG)
        for query in ['', 'the and or', 'zzzznonexistent']:
            self.assertEqual(index.search(query), [])
        with self.assertRaises(ValueError):
            index.search('monitor', 0)

    def test_cache_reuse_invalidation_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'index.joblib'
            first = ProductIndex(CATALOG, cache)
            cached = ProductIndex(CATALOG, cache)
            self.assertTrue(cached.cache_hit)
            self.assertEqual(first.search('monitor'), cached.search('monitor'))
            changed = [*CATALOG, {'parent_asin': 'B000000004', 'title': 'Gaming monitor'}]
            modified = ProductIndex(changed, cache)
            self.assertFalse(modified.cache_hit)
            self.assertNotEqual(first.fingerprint, modified.fingerprint)
            cache.write_bytes(b'corrupt local cache')
            recovered = ProductIndex(CATALOG, cache)
            self.assertFalse(recovered.cache_hit)
            self.assertEqual(first.search('monitor'), recovered.search('monitor'))

    def test_ties_sorted_by_id(self):
        records = [dict(parent_asin=asin, title='Identical monitor')
                   for asin in ['B000000002', 'B000000001']]
        self.assertEqual([h.product_id for h in ProductIndex(records).search('monitor')],
                         ['B000000001', 'B000000002'])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.index = ProductIndex(CATALOG)

    def test_prompt_contains_linked_evidence_without_scores(self):
        hits = self.index.search('monitor')
        prompt = json.loads(build_prompt('Office work?', hits))
        self.assertEqual(prompt['question'], 'Office work?')
        self.assertEqual(prompt['products'][0]['product_id'], hits[0].product_id)
        self.assertIn('Office monitor', prompt['products'][0]['catalog_document'])
        self.assertNotIn('retrieval_score', prompt['products'][0])
        from src.rag.service import SYSTEM_PROMPT
        self.assertIn('untrusted', SYSTEM_PROMPT)
        self.assertIn('Do not add unsupported specifications', SYSTEM_PROMPT)
        with self.assertRaises(ValueError):
            build_prompt('', hits)

    @patch('urllib.request.urlopen')
    def test_supported_api_and_successful_generation(self, urlopen):
        urlopen.side_effect = [
            response({'models': [{'name': 'tiny:latest', 'size': 100}]}),
            response({'done': True, 'response': generated()}),
        ]
        result = answer_question(self.index, 'office monitor', client=OllamaClient(model='tiny:latest'))
        self.assertEqual(result.status, 'generated')
        self.assertIn('[B000000001] Office monitor', result.answer)
        self.assertIn('Inferred suitability:', result.answer)
        request = urlopen.call_args_list[1].args[0]
        self.assertTrue(request.full_url.endswith('/api/generate'))
        payload = json.loads(request.data)
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['model'], 'tiny:latest')
        self.assertIn('format', payload)

    @patch('urllib.request.urlopen', side_effect=urllib.error.URLError('server down'))
    def test_missing_server_fallback(self, urlopen):
        result = answer_question(self.index, 'monitor', client=OllamaClient())
        self.assertEqual(result.status, 'retrieval_only')
        self.assertIsNone(result.answer)
        self.assertEqual(result.hits[0].product_id, 'B000000001')
        self.assertIn('Ollama request failed', result.reason)

    @patch('urllib.request.urlopen', side_effect=TimeoutError('timeout'))
    def test_timeout_fallback(self, urlopen):
        self.assertEqual(answer_question(self.index, 'monitor').status, 'retrieval_only')

    @patch('urllib.request.urlopen')
    def test_missing_model_never_generates_or_downloads(self, urlopen):
        urlopen.return_value = response({'models': []})
        result = answer_question(self.index, 'monitor', client=OllamaClient(model='absent'))
        self.assertEqual(result.status, 'retrieval_only')
        self.assertEqual(urlopen.call_count, 1)
        self.assertTrue(urlopen.call_args.args[0].full_url.endswith('/api/tags'))

    def test_no_match_and_explicit_retrieval_skip_generation(self):
        client = MagicMock()
        result = answer_question(self.index, 'zzzznonexistent', client=client)
        self.assertEqual(result.status, 'no_results')
        result = answer_question(self.index, 'monitor', client=client, retrieval_only=True)
        self.assertEqual(result.status, 'retrieval_only')
        client.generate.assert_not_called()

    def test_invalid_evidence_or_unknown_ids_are_not_generated_answers(self):
        for raw in [generated(quote='Invented 240Hz specification'), generated(product_id='B999999999'),
                    'not json', '[]', json.dumps({'products': []})]:
            client = MagicMock()
            client.generate.return_value = ('tiny:latest', raw)
            result = answer_question(self.index, 'monitor', client=client)
            self.assertEqual(result.status, 'retrieval_only')
            self.assertIsNone(result.answer)

    def test_explanation_keeps_ml_selected_ids_and_order(self):
        result = answer_question(self.index, 'Why these?',
                                 product_ids=['B000000003', 'B000000001'], retrieval_only=True)
        self.assertEqual(result.mode, 'explanation')
        self.assertEqual([h.product_id for h in result.hits], ['B000000003', 'B000000001'])
        self.assertTrue(all(h.retrieval_score is None for h in result.hits))


if __name__ == '__main__':
    unittest.main()
