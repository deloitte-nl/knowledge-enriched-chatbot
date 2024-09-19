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


import copy
import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import SpacyTextSplitter

from src.constants import DocumentMetadata
from src.utils import initialize_logger, num_tokens_from_string

dm = DocumentMetadata()
logger = initialize_logger(__name__)


class DocumentChunker:
    """
    Document chunking class that:
        1. Chunks the documents into smaller pieces, so that they're ready for embedding.
        2. Adds metadata on chunk level: nth number of chunk in document, chunk ID.
        3. Returns chunked documents with metadata.

    Args:
        documents: List of loaded documents.
        chunking_strategy: Chunking strategy, e.g., tokenized sentence chunking.
        kwargs: Input parameters that are conditional to the chunking strategy.
    """

    output: List[Document]

    def __init__(self, chunking_strategy: str, **kwargs: dict):

        self.chunking_strategy = chunking_strategy
        self.__dict__.update(kwargs)
        self.chunker = self._set_chunker()

    def _set_chunker(self):
        """
        Instantiates a document chunker based on the document strategy.
        Custom chunking strategies can be added by creating one that inherits from the Langchain class
        and adding them here.
        """
        if self.chunking_strategy.lower() == "sentencesplitting":
            """Splitting text using Spacy package."""
            logger.info("Using Langchain SpacyTextSplitter for chunking the documents")
            return SpacyTextSplitter(
                chunk_size=self.kwargs["chunk_size"],
                chunk_overlap=self.kwargs["chunk_overlap"],
                separator=self.kwargs.get("separator", "\n\n"),
                pipeline=self.kwargs.get("pipeline", "en_core_web_sm"),
            )
        elif self.chunking_strategy.lower() == "tokenizedsentencesplitting":
            """Splitting text using Spacy package."""
            logger.info("Using Langchain SpacyTextSplitter for chunking the documents")
            return SpacyTextSplitter.from_tiktoken_encoder(
                chunk_size=self.kwargs["chunk_size"],
                chunk_overlap=self.kwargs["chunk_overlap"],
                separator=self.kwargs.get("separator", "\n\n"),
                pipeline=self.kwargs.get("pipeline", "en_core_web_sm"),
                encoding_name=self.kwargs.get("encoding_name", "cl100k_base"),
            )
        else:
            raise NotImplementedError(
                f"Chunking strategy {self.chunking_strategy} not available. Only 'SentenceSplitting' is currently implemented."
            )

    def _get_urls_from_page_content(self, document: Document) -> str:
        """
        Filters out URLs in page_content, adds URLs to document metadata if exists.
        Returns string because metadata from Langchain Document does not accept list format.

        Args:
            document: Document object.

        Returns:
            Extracted URLs as a string.
        """
        # regular expresson to match URLs
        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # merge all urls into a string, stored as "urls" in metadata
        urls = [
            url.split("\\", 1)[0] for url in url_pattern.findall(document.page_content)
        ]
        return "\n ".join(urls) if urls else ""

    def _add_chunk_metadata(self, chunked_documents: List[Document]) -> List[Document]:
        """
        Adds information to metadata about the ordering index of the chunk within a document ('nth chunk'). Logs number of tokens.

        Args:
            chunked_documents: List of chunked Document objects.

        Returns:
            List of Document objects with added metadata.
        """
        chunked_docs_with_metadata: List[Document] = []
        document_id = None
        for chunked_doc in chunked_documents:
            # if the previous chunk was in the same document (based on the document id of the chunk), this is the n+1th chunk in the document
            if document_id == chunked_doc.metadata[dm.DOCUMENT_ID]:
                nth_chunk += 1
            # if the previous chunk was in a different document, this is the first chunk of the document
            else:
                nth_chunk = 0

            # creating a deep copy to prevent the original dict from being updated
            new_metadata = copy.deepcopy(chunked_doc.metadata)
            new_metadata[dm.NTH_CHUNK] = nth_chunk
            new_metadata[dm.URLS] = self._get_urls_from_page_content(chunked_doc)
            doc_with_new_metadata = Document(
                page_content=chunked_doc.page_content, metadata=new_metadata
            )
            chunked_docs_with_metadata.append(doc_with_new_metadata)
            document_id = chunked_doc.metadata[dm.DOCUMENT_ID]

            chunk_length_in_tokens = num_tokens_from_string(
                string=chunked_doc.page_content,
                encoding_name=self.kwargs.get("encoding_name", "cl100k_base"),
            )
            chunk_length_in_chars = len(chunked_doc.page_content)
            logger.info(
                f"The length of chunk {document_id}.{nth_chunk} is {chunk_length_in_tokens} tokens and {chunk_length_in_chars} characters."
            )

        return chunked_docs_with_metadata

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits the documents into chunks according to the chunking strategy. Adds metadata.

        Args:
            documents: List of Document objects.

        Returns:
            List of chunked Document objects with added metadata.
        """
        chunked_documents = self.chunker.split_documents(documents=documents)
        return self._add_chunk_metadata(chunked_documents)
