import json
from pathlib import Path

import config as pm
from parameters import *
from src.eval.evaluation_pipeline import EvaluationPipeline
from src.KG.cypher_qa_retriever import CypherQARetriever
from src.KG.integrated_qa_retriever import IntegratedQARetriever
from src.RAG.qa_retriever import CustomQARetriever
from src.RAG.vector_store import ChromaVectorStore
from src.utils import initialize_logger, set_openai_env_vars

logger = initialize_logger(__name__)


logger.info("Load and set OpenAI key configuration")
config_path = Path(__file__).parent.joinpath("config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

logger.info("Set OpenAI environment variables")
set_openai_env_vars(pm.PATH_TO_OPENAI_CONFIG)

logger.info("Read existing ChromaVectorStore")
chroma_db = ChromaVectorStore(
    chunked_documents=[],
    db_path=pm.vector_args.db_path,
    embedding_source=pm.vector_args.embedding_source,
)

logger.info("Create the retriever and the QA pipeline")
if pm.retriever_args.similarity_search:
    vector_store_retriever = chroma_db.chromadb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": pm.retriever_args.similarity_score_threshold,
            "k": pm.retriever_args.top_k_candidate_documents,
        },
    )
else:
    vector_store_retriever = chroma_db.chromadb.as_retriever(
        search_kwargs={"k": pm.retriever_args.top_k_candidate_documents}
    )


logger.info("Create an instance of CustomQARetriever")
qa_retriever = CustomQARetriever(
    retriever=vector_store_retriever,
    prompt_path=pm.retriever_args.prompt_path,
    citation_metadata=pm.retriever_args.citation_metadata,
    llm=pm.retriever_args.retrieval_LLM,
)

if pm.eval_args.retriever_to_evaluate != "vectordb":
    logger.info("Create an instance of the Cypher QA retriever")
    cypher_qa_retriever = CypherQARetriever(
        num_examples=pm.kg_args.num_examples,
        examples_path=pm.kg_args.examples_path,
        prompt_path=pm.kg_args.prompt_path,
        llm=pm.kg_args.cypher_LLM,
    )

    logger.info("Create an instance of IntegratedQARetriever")
    qa_retriever = IntegratedQARetriever(
        vector_qa_retriever=qa_retriever,
        cypher_qa_retriever=cypher_qa_retriever,
        integrated_prompt_path=pm.kg_args.integrated_prompt_path,
        llm=pm.kg_args.integration_LLM,
    )

logger.info("Create an instance of EvaluationPipeline")
evaluator = EvaluationPipeline(vector_store=chroma_db, qa_retriever=qa_retriever)

logger.info("Evaluate based on synthetic data")
synthetic_results = evaluator.evaluate_synthetic(
    dataset_size=pm.eval_args.dataset_size,
    metrics=pm.eval_args.evaluate_synthetic_metrics,
    output_eval_dataset=pm.eval_args.output_eval_dataset,
    evaluative_llm=pm.eval_args.generative_LLM,
    generative_llm=pm.eval_args.evaluative_LLM,
)

logger.info("Evaluate based on the ground truth dataset")
results = evaluator.evaluate_ground_truth(
    metrics=pm.eval_args.evaluate_ground_truth_metrics,
    output_eval_dataset=pm.eval_args.output_eval_dataset,
    ground_truth_filepath=pm.eval_args.ground_truth_filepath,
    evaluative_llm=pm.eval_args.evaluative_LLM,
)
