class DocumentMetadata:
    """Document class metadata keys"""

    __slots__ = ()

    DOCUMENT_ID = "document_id"
    SOURCE = "source"
    PAGE = "page"
    PAGE_NAME = "page_name"
    PAGE_NUMBER = "page_number"
    COLUMN_NAME = "column_name"
    CELL_INDEX = "cell_index"
    NTH_CHUNK = "nth_chunk"
    EMBEDDING_MODEL = "embedding_model"
    URLS = "urls"


class MetricMetadata:
    """Metric configurations"""

    __slots__ = ()

    class RagasMetrics:
        """RAGAS metrics"""

        __slots__ = ()

        CONTEXT_RECALL = "ContextRecall"
        CONTEXT_PRECISION = "ContextPrecision"
        FAITHFULNESS = "Faithfulness"
        ANSWER_RELEVANCY = "AnswerRelevancy"

    class RagasMetadata:
        """HuggingFace Dataset formatting for RAGAS."""

        __slots__ = ()

        QUESTION = "question"
        GROUND_TRUTH_ANSWER = "ground_truths"
        GENERATED_ANSWER = "answer"
        CONTEXTS = "contexts"
        GROUND_TRUTH_CONTEXT = "ground_truth_context"

    class HitRateMetadata:
        """HitRate metrics metadata."""

        __slots__ = ()

        HIT_RATE = "hitrate"
        RETRIEVED = "retrieved"
        EXPECTED = "expected"

    # Allowed evaluations for synthetic and ground truth metrics
    ALLOWED_EVALS_SYNTHETHIC = [
        RagasMetrics.CONTEXT_PRECISION,
        RagasMetrics.FAITHFULNESS,
        RagasMetrics.ANSWER_RELEVANCY,
        HitRateMetadata.HIT_RATE,
    ]
    ALLOWED_EVALS_GROUND_TRUTH = [
        RagasMetrics.CONTEXT_RECALL,
        RagasMetrics.CONTEXT_PRECISION,
        RagasMetrics.FAITHFULNESS,
        RagasMetrics.ANSWER_RELEVANCY,
    ]
