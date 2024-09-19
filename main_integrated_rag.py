# Knowledge-enriched chatbot
# Copyright (C) 2024 Deloitte Risk Advisory B.V. 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You have received a copy of the GNU Affero General Public License
# and our supplemental terms in LICENSE.MD in the root folder.  

import config as pm
from src.KG.cypher_qa_retriever import CypherQARetriever
from src.KG.integrated_qa_retriever import IntegratedQARetriever
from src.RAG.document_chunker import DocumentChunker
from src.RAG.document_loader import DocumentLoader
from src.RAG.qa_retriever import CustomQARetriever
from src.RAG.vector_store import ChromaVectorStore
from src.utils import initialize_logger, set_openai_env_vars

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
        documents = document_loader.load()

        document_chunker = DocumentChunker(
            chunking_strategy=pm.vector_args.chunking_strategy,
            kwargs={
                "chunk_overlap": pm.vector_args.chunk_overlap,
                "chunk_size": pm.vector_args.chunk_size,
            },
        )

        docs_to_load = document_chunker.split_documents(documents=documents)

    chroma_db = ChromaVectorStore(
        chunked_documents=docs_to_load,
        db_path=pm.vector_args.db_path,
        embedding_source=pm.vector_args.embedding_source,
    )

    logger.info("Create the retriever for the QA pipeline")
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

    logger.info("Create an instance of the vectorDB QA retriever")
    qa_retriever = CustomQARetriever(
        retriever=vector_store_retriever,
        prompt_path=pm.retriever_args.prompt_path,
        citation_metadata=pm.retriever_args.citation_metadata,
        llm=pm.retriever_args.retrieval_LLM,
    )

    logger.info("Create an instance of the Cypher QA retriever")
    cypher_qa_retriever = CypherQARetriever(
        num_examples=pm.kg_args.num_examples,
        examples_path=pm.kg_args.examples_path,
        prompt_path=pm.kg_args.prompt_path,
        llm=pm.kg_args.cypher_LLM,
    )

    logger.info("Create an instance of IntegratedQARetriever")
    integrated_qa_retriever = IntegratedQARetriever(
        vector_qa_retriever=qa_retriever,
        cypher_qa_retriever=cypher_qa_retriever,
        integrated_prompt_path=pm.kg_args.integrated_prompt_path,
        llm=pm.kg_args.integration_LLM,
    )

    try:
        Q = "What community programs have been developed to engage citizens in sustainability efforts within the Urban Greening Initiative?"
        result = integrated_qa_retriever.qa(Q, verbose=pm.kg_args.verbose)

        logger.info(
            f"""
        Q: 
        {Q}

        A: 
        {result}

        """
        )
    finally:
        logger.info("Closing the Cypher QA retriever")
        integrated_qa_retriever.cypher_qa_retriever.close()


if __name__ == "__main__":
    main()
