# Local catalog RAG

## Installation and configuration

Python dependencies are already in `requirements.txt`; HTTP uses the standard
library. RAG product discovery needs a local product JSONL catalog, not the full
rating history or trained recommendation bundle. Run `python -m src.data.prepare`
with the original source files, or provide `--catalog PATH` to an existing Amazon
Electronics metadata file. The default is `data/processed/v2/products.jsonl`,
falling back to the original prepared metadata files.

Install Ollama using its [official Linux instructions](https://docs.ollama.com/linux)
or the Windows application when serving from Windows/WSL. Start/check it:

```bash
ollama serve                 # Separate terminal; skip if already running
ollama list
.venv/bin/python scripts/rag_demo.py --check-ollama
```

If no model is installed, choose a small local model from the
[Ollama library](https://ollama.com/library), inspect its memory/download size,
and download it yourself with `ollama pull MODEL_NAME`. The application never
installs Ollama or downloads weights. Keep model downloads outside the repository.
The supplied environment already has `mistral:latest` (7.2B, approximately 4.37 GB
of weights); it was reused rather than downloading another model.

| Setting | CLI | Environment / default |
| --- | --- | --- |
| Model | `--model NAME` | `OLLAMA_MODEL`; otherwise smallest installed local model |
| Server | `--host URL` | `OLLAMA_HOST`; otherwise `http://localhost:11434` |
| Request timeout | `--timeout SECONDS` | 60 seconds |
| Products | `--top-k N` | 1–5; default 5 |
| Skip generation | `--retrieval-only` | Always works without Ollama |
| Catalog | `--catalog PATH` | Existing prepared catalog, as above |
| Structured output | `--json` | Includes explicit generation/fallback status |

Only downloaded local model names returned by `GET /api/tags` are accepted.
The adapter sends `POST /api/generate` with `stream=false`, temperature 0, seed
42, a 4,096-token context setting and a 600-token output cap. A fixed seed is
best-effort generation control, not a guarantee across model/server versions.
See [generation API](https://docs.ollama.com/api/generate) and
[model listing](https://docs.ollama.com/api/tags).

For WSL with Ollama on Windows, try localhost first. If Windows networking does
not expose it there, configure an explicit reachable local server address with
`OLLAMA_HOST` / `--host` and consult the
[Ollama FAQ](https://docs.ollama.com/faq). Do not expose an unauthenticated local
server to the public internet. A sandbox may block even loopback sockets; in
that case the CLI honestly returns retrieval-only results. The release's live
check was run with authorized loopback access.

## How retrieval and grounding work

- Validate parent-ASINs (10 uppercase alphanumeric characters) and nonempty
  titles. Exclude absent/invalid records; first valid duplicate ID wins.
- Flatten titles/categories/features/descriptions into linked documents. Prices
  and aggregate rating counts are not indexed or sent to generation.
- Fit TF-IDF with word unigrams/bigrams and rank by cosine similarity,
  with stable product-ID tie ordering. This is lexical retrieval, not a learned
  semantic model.
- Cache the vectorizer and sparse document matrix under `.cache/rag/`.
  The fingerprint includes product content, cache version and scikit-learn
  version. Changed content or damaged/incompatible caches are rebuilt. Cache and
  model files use joblib: load only trusted locally generated files.
- Pass bounded documents, IDs and exact evidence excerpt choices to Ollama.
  The system prompt treats questions/catalog text as untrusted data and requires
  catalog-only claims, no invented prices/discounts/specifications/scores,
  explicit suitability inferences and missing information.
- Constrain the generated JSON to known IDs and excerpt options. Validate each
  product ID and its corresponding exact quote before rendering the explanation.
  Product titles come from the catalog, not the generated text. These mechanical
  checks do not verify the semantics of every suitability inference.

Status values are `generated`, `retrieval_only` and `no_results`. An unavailable
server, absent model, timeout, malformed JSON, unknown ID or unsupported quote
cannot produce a `generated` result. Zero-overlap queries never invoke the LLM.
A returned excerpt may contain a manufacturer's claim; quoting it does not
independently verify its accuracy.

## Discovery and recommendation explanation

```bash
.venv/bin/python scripts/rag_demo.py --query "Recommend a monitor for office work." --top-k 3
.venv/bin/python scripts/rag_demo.py --query "I need wireless headphones." --retrieval-only
.venv/bin/python scripts/rag_demo.py --user USER_ID --recommender Item_V1 --top-k 3
```

For `--user`, the existing serialized ML model selects products first; metadata
are looked up in that exact order. ML ranking scores are shown separately and
are never passed to the LLM. Unknown users use the existing popularity fallback.
Missing metadata are reported. The LLM explains possible suitability from
metadata, not the causal reason for a KNN score or undisclosed user preferences.

## Verification and limits

`tests/test_rag.py` covers retrieval ordering, mappings, invalid/duplicate
metadata, cache reuse/invalidation/recovery, evidence-linked prompts, successful
mocked HTTP, missing servers/models, timeouts, unsupported evidence and explicit
fallback. Tests require no running server or installed model.

The release retains a real discovery call in
`reports/results/rag_live_demo.json`; its `status` is the authoritative live
outcome. Notebook 09 displays that recorded answer separately from its default
retrieval-only examples. Enabling `RUN_LIVE` there makes a new server call.
The first development call paraphrased an evidence quote and was rejected;
constraining generation to exact excerpt options resolved that issue in the
verified final example. Recommendation explanation also passed a live call, recorded in
`reports/results/rag_explanation_live.json`. Both final calls returned
`generated` with Ollama 0.24.0 and `mistral:latest`.

No answer-quality benchmark, human relevance assessment, discount calculation,
current stock lookup, conversation memory, agents or model finetuning is added.
Lexical retrieval may miss synonyms or retrieve adjacent accessories. Metadata
may be outdated; inferred suitability and model-generated missing-information
statements require review. This extension leaves all core ML measurements intact.
