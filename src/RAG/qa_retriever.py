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


import json
from pathlib import Path
from typing import Any, Optional

from langchain.callbacks.manager import Callbacks
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.vectorstores.base import VectorStoreRetriever

from src.constants import DocumentMetadata
from src.utils import initialize_logger, create_Azure_LLM

logger = initialize_logger(__name__)
dm = DocumentMetadata()


class CustomQARetriever:
    """
    A module that retrieves candidate chunked documents from the database given a question
    Then prompts an LLM with the question and the candidate chunked documents to answer the question.
    Optionally reads in custom JSON QA prompt template from path.

    Args:
        retriever: Retriever initialized from ChromaDB vectorstore.
        prompt_path: Path to the JSON file containing the prompt messages.
        return_source_documents: Whether to include the source documents retrieved from the vector store in the LLM result.
        combine_documents_strategy: The strategy used to combine the candidate chunked documents that are given to the LLM.
        llm: The Azure LLM to be used for answering questions.
        citation_metadata: The metadata the LLM should receive and use for the source citation in the answer.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt_path: Path = None,
        return_source_documents: bool = True,
        combine_documents_strategy: str = "stuff_with_citation",
        llm: AzureChatOpenAI = None,
        citation_metadata: str = dm.CELL_INDEX,
    ):
        if llm is None:
            self.llm = create_Azure_LLM(temperature=0.1)
        else:
            self.llm = llm

        self.retriever = retriever
        self.default_prompt_path = Path(__file__).parent.joinpath(
            "prompts", "prompt_messages_citation.json"
        )
        self.prompt = self._load_prompt_if_not_none(prompt_path)
        self.return_source_documents = return_source_documents
        self.combine_documents_strategy = combine_documents_strategy
        self.citation_metadata = citation_metadata
        self.qa = self._set_retrieval_qa()

    def _load_prompt_if_not_none(self, path_to_prompt_messages: Path) -> PromptTemplate:
        """
        Load a list of messages from a JSON path that contains messages for the chat prompt template.

        Args:
            path_to_prompt_messages: Path to the JSON file containing the prompt messages.

        Returns:
            The loaded chat prompt template.
        """
        if not path_to_prompt_messages:
            logger.info("No prompt path provided, using default prompt.")
            path_to_prompt_messages = self.default_prompt_path

        with open(path_to_prompt_messages, "rt", encoding="utf-8") as f:
            prompt_messages = json.load(f)

        prompt_messages = [(actor, message) for actor, message in prompt_messages]

        return ChatPromptTemplate.from_messages(prompt_messages)

    def _set_retrieval_qa(self) -> RetrievalQA:
        if self.combine_documents_strategy == "stuff":
            logger.info(
                "Setting up the QA retriever with 'stuff' as document combining strategy"
            )
            return RetrievalQA.from_llm(
                llm=self.llm,
                prompt=self.prompt,
                retriever=self.retriever,
                return_source_documents=self.return_source_documents,
            )
        elif self.combine_documents_strategy == "stuff_with_citation":
            logger.info(
                "Setting up the QA retriever with 'stuff_with_citation' as document combining strategy. Metadata will be included for stuffed documents to use for citation."
            )
            return RetrievalQAMetadata.from_llm(
                llm=self.llm,
                citation_metadata=self.citation_metadata,
                prompt=self.prompt,
                retriever=self.retriever,
                return_source_documents=self.return_source_documents,
            )
        else:
            raise NotImplementedError(
                f"Strategy '{self.combine_documents_strategy}' not available for combining documents, currently only 'stuff' and 'stuff_with_citation' is implemented."
            )


class RetrievalQAMetadata(RetrievalQA):
    """
    A Langchain RetrievalQA class child that includes metadata
    when stuffing the retrieved documents into the LLM QA prompt.
    """

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        citation_metadata: str,
        prompt: Optional[PromptTemplate] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt, callbacks=callbacks)
        template = f"Context:\nDocument[{{{citation_metadata}}}]: {{page_content}}"
        document_prompt = PromptTemplate(
            input_variables=["page_content", citation_metadata], template=template
        )

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )
