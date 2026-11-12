"""Optional catalog retrieval and grounded local-generation extension."""

from .retrieval import ProductIndex, SearchHit, load_catalog
from .service import OllamaClient, RAGResult, answer_question, build_prompt

__all__ = [
    'ProductIndex', 'SearchHit', 'load_catalog', 'OllamaClient',
    'RAGResult', 'answer_question', 'build_prompt',
]
