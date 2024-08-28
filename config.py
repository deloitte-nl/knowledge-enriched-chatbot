from pathlib import Path
import logging

from parameters import *

LOGGING_LEVEL = logging.INFO
PATH_TO_OPENAI_CONFIG = Path(__file__).parent.joinpath("config.json")

# Input parameters
load_args = ExcelLoaderArguments(
    document_path = Path(__file__).parent.joinpath(
    "data", "documents", "UrbanGreeningInitiative.xlsx"
    ),
    sheet_name="Sheet1",
    document_type="customexcel",
    column_names=["Subcategory", "Explanation"],
    load_from_existing_vector_db=False,
)

vector_args = VectorDBArguments(
    chunking_strategy="TokenizedSentenceSplitting",
    chunk_overlap=128,
    chunk_size=256,
    db_path=Path(__file__).parent.joinpath("data", "chromaDB"), 
    embedding_source="HuggingFace",
)

retriever_args = QARetrievalArguments(
    top_k_candidate_documents=8,
    citation_metadata=dm.CELL_INDEX,
    similarity_search=True,
    similarity_score_threshold=0.2,
    combine_document_strategy = "stuff_with_citation",
    prompt_path=Path(__file__).parent.joinpath(
    "src", "RAG", "prompts", "prompt_messages_citation.json"),
    retrieval_LLM=None,
)

kg_args = KGIntegratedQARetrievalArguments(
    prompt_path=Path(__file__).parent.joinpath(
    "src", "KG", "prompts", "kg_qa_prompt.txt"),
    integrated_prompt_path=Path(__file__).parent.joinpath(
    "src", "KG", "prompts", "integrated_qa_prompt.txt"),
    examples_path=Path(__file__).parent.joinpath(
    "data", "KG_example_queries", "KG_Cypher_Examples.xlsx"),
    num_examples=4,
    verbose=False,
    cypher_LLM=None,
    integration_LLM=None,
)

# Settings for the evaluation
eval_args = EvaluationArguments(
    retriever_to_evaluate="vectordb",
    dataset_size=1,
    generative_LLM=None,
    evaluative_LLM=None,
    output_eval_dataset=True,
    evaluate_synthetic_metrics=["hitrate", "ContextPrecision", "Faithfulness"],
    evaluate_ground_truth_metrics=["Faithfulness", "ContextRecall", "ContextPrecision"],
    ground_truth_filepath=Path(__file__).parent.joinpath("data", "documents", "ground_truth_evaldataset.csv"),
)


