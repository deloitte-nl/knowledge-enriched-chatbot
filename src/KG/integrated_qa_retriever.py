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


from pathlib import Path

from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

from src.KG.cypher_qa_retriever import CypherQARetriever
from src.RAG.qa_retriever import CustomQARetriever
from src.utils import create_Azure_LLM, initialize_logger

logger = initialize_logger(__name__)

class IntegratedQARetriever:
    """
    A module that combines the context from a QA LLM based on a graph database and a QA LLM
    based on a vector store of documents into a new LLM which aims to answer a user question.

    Args:
        vector_qa_retriever: An instance of CustomQARetriever for retrieving QA information from a vector store of documents.
        cypher_qa_retriever: An instance of CypherQARetriever for retrieving QA information from a graph database.
        llm: The Azure LLM to be used. 
        integrated_prompt_path: Path to the prompt with instructions aimed at the LLM that integrates the KG and vector DB output.
    """

    def __init__(
        self,
        vector_qa_retriever: CustomQARetriever,
        cypher_qa_retriever: CypherQARetriever,
        llm: AzureChatOpenAI = None,
        integrated_prompt_path: Path = None,
    ):
        self.vector_qa_retriever = vector_qa_retriever
        self.cypher_qa_retriever = cypher_qa_retriever

        if llm is None:
            self.llm = create_Azure_LLM(temperature=0)
        else:
            self.llm = llm
        integrated_qa_prompt = self._load_prompt(path_to_prompt=integrated_prompt_path)
        self.integrated_qa_chain = LLMChain(prompt=integrated_qa_prompt, llm=self.llm)

    def _load_prompt(self, path_to_prompt=None):
        """Load a prompt template for the integrated retriever"""
        with open(path_to_prompt, "rt", encoding="utf-8") as f:
            system_message_template = f.read()

        return PromptTemplate(
            input_variables=["schema", "question"], template=system_message_template
        )

    def qa(self, Q, verbose=False, return_context=False):
        """
        Run a question answering LLM which uses context from documents and from
        a graph database.

        Args:
            Q: The user question to be answered.
            verbose: Whether the context and individual answers of KG and document Q&A should be printed.
            return_context: Whether the function should return the KG and document context along with the answer in a dictionary.
        """

        logger.info("Get the context from the Cypher KG QA retriever")
        try:
            KG_result = self.cypher_qa_retriever.run_KG_qa(Q, verbose=False)
            KG_answer = KG_result["result"]
            KG_context = KG_result["intermediate_steps"][1]["context"]

            if verbose:
                logger.info(f"KG context: {KG_context}")
                logger.info(f"KG answer: {KG_answer}")
                logger.info("\n-----------------------------\n")

        except Exception as e:
            logger.info("An incorrect cypher query was produced")
            KG_context = "[]"

        logger.info("Get the context from the vector DB retriever")
        LLM_result = self.vector_qa_retriever.qa(Q)
        LLM_context = [
            doc.page_content.replace("\n", " ")
            for doc in LLM_result["source_documents"]
        ]

        if verbose:
            logger.info(f"Document context: {LLM_context}")
            logger.info(f"Document answer: {LLM_result}")
            logger.info("\n-----------------------------\n")

        logger.info("Run the integrated QA chain with the given question")
        result = self.integrated_qa_chain.run(
            question=Q, documents=LLM_context, graph_context=KG_context
        )

        if return_context:
            return {
                "result": result,
                "source_documents": LLM_result["source_documents"],
                "graph_context": KG_context,
            }
        else:
            return result
