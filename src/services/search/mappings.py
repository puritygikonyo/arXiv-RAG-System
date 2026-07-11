"""
OpenSearch index mappings for arXiv papers.

A mapping tells OpenSearch:
  - What fields exist in your documents
  - What type each field is (text, keyword, date, integer...)
  - How each field should be indexed and searched

Think of it like a database schema, but for a search engine.

FIELD TYPES USED HERE:
  text     → full-text searchable, gets tokenised into words
  keyword  → exact match only, not tokenised (good for IDs, categories)
  date     → date/time values, enables range queries
"""

PAPERS_INDEX_MAPPING: dict = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,    # no replicas on single node — avoids yellow status
        "index.knn": True,          # REQUIRED to use knn_vector fields
        "analysis": {
            "analyzer": {
                "paper_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "stop",
                        "asciifolding",
                    ],
                }
            }
        },
    },
    "mappings": {
        "properties": {

            # ------------------------------------------------------------------
            # Identity fields — exact match only, never tokenised
            # ------------------------------------------------------------------
            "arxiv_id": {
                "type": "keyword",
            },
            "pdf_url": {
                "type": "keyword",
                "index": False,
            },

            # ------------------------------------------------------------------
            # Text fields — full-text searchable with BM25
            # ------------------------------------------------------------------
            "title": {
                "type": "text",
                "analyzer": "paper_analyzer",
                "boost": 3,
                "fields": {
                    "keyword": {"type": "keyword"}
                },
            },
            "abstract": {
                "type": "text",
                "analyzer": "paper_analyzer",
                "boost": 2,
            },
            "authors": {
                "type": "text",
                "analyzer": "paper_analyzer",
                "boost": 1,
                "fields": {
                    "keyword": {"type": "keyword"}
                },
            },
            "full_text": {
                "type": "text",
                "analyzer": "paper_analyzer",
            },

            # ------------------------------------------------------------------
            # Category fields — exact match for filtering
            # ------------------------------------------------------------------
            "primary_category": {
                "type": "keyword",
            },
            "categories": {
                "type": "keyword",
            },

            # ------------------------------------------------------------------
            # Date fields
            # ------------------------------------------------------------------
            "published_at": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
            "ingested_at": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },

            # ------------------------------------------------------------------
            # Phase 6 — vector embedding field
            # Declared now so we don't have to reindex in Phase 6.
            # index.knn: True (above in settings) is required for this field.
            # ------------------------------------------------------------------
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,   # matches JINA_EMBEDDING_DIMENSIONS in config
            },
        }
    },
}