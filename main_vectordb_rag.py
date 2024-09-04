from time import perf_counter

import config as pm
from src.constants import DocumentMetadata
from parameters import *
from src.RAG.document_chunker import DocumentChunker
from src.RAG.document_loader import DocumentLoader
from src.RAG.qa_retriever import CustomQARetriever
from src.utils import initialize_logger, set_openai_env_vars
from src.RAG.vector_store import ChromaVectorStore

dm = DocumentMetadata()
logger = initialize_logger(__name__)

logger.info("Set OpenAI environment variables")
set_openai_env_vars(pm.PATH_TO_OPENAI_CONFIG)


def main():
    """Main execution flow for the QA system."""

    logger.info("Read or create vectorDB with embedded and chunked documents")
    if pm.load_args.load_from_existing_vector_db:
        docs_to_load = []
    else:
        document_loader = DocumentLoader(
            document_path=pm.load_args.document_path,
            document_type=pm.load_args.document_type,
            kwargs={
                "sheet_name": pm.load_args.sheet_name,
                "column_names": pm.load_args.column_names,
            },
        )
        t1_start = perf_counter()
        documents = document_loader.load()
        t1_stop = perf_counter()
        logger.info(
            f"Elapsed time loading the documents: {int(t1_stop - t1_start)} seconds"
        )

        document_chunker = DocumentChunker(
            chunking_strategy=pm.vector_args.chunking_strategy,
            kwargs={
                "chunk_overlap": pm.vector_args.chunk_overlap,
                "chunk_size": pm.vector_args.chunk_size,
            },
        )

        t2_start = perf_counter()
        docs_to_load = document_chunker.split_documents(documents=documents)
        t2_stop = perf_counter()
        logger.info(
            f"Elapsed time chunking the documents: {int(t2_stop - t2_start)} seconds"
        )

    t1_start = perf_counter()
    chroma_db = ChromaVectorStore(
        chunked_documents=docs_to_load,
        db_path=pm.vector_args.db_path,
        embedding_source=pm.vector_args.embedding_source,
    )
    t1_stop = perf_counter()
    logger.info(
        f"Elapsed time storing and embedding the documents in ChromaDB vector store: {int(t1_stop - t1_start)} seconds"
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

    qa_retriever = CustomQARetriever(
        retriever=vector_store_retriever,
        prompt_path=pm.retriever_args.prompt_path,
        citation_metadata=pm.retriever_args.citation_metadata,
        llm=pm.retriever_args.retrieval_LLM,
    )

    logger.info("Asks questions")
    Q = "What community programs have been developed to engage citizens in sustainability efforts within the Urban Greening Initiative?"
    result = qa_retriever.qa(Q)
    referenced_chunks = [
        doc.page_content.replace("\n", " ") for doc in result["source_documents"]
    ]

    logger.info(
        f"""
    Q: 
    {Q}

    A: 
    {result['result']}

    Source documents:
    {referenced_chunks}
    """
    )


if __name__ == "__main__":
    main()
