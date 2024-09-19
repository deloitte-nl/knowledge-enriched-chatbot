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

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
from collections import OrderedDict
from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document

from src.constants import DocumentMetadata
from src.RAG.custom_excel_loader import customExcelLoader
from src.utils import initialize_logger, uuid_hash

logger = initialize_logger(__name__)
dm = DocumentMetadata()


class DocumentLoader:
    """
    Document loading class that:
        1. Loads documents from path
        2. Uses the hash of the page content to create a unique documentID
        3. Returns list of Langchain Document

    Args:
        document_path: Location of the raw document(s).
        document_type: Data type of document(s), e.g., customexcel.
        kwargs: Input parameters that are conditional to the document type.

    Returns:
        List[Document]: Each Document has two attributes:
            page_content: Content of documents (For PDF, each page is stored in one page_content; for Excel, each cell is stored in one page_content).
            metadata: Information of each page_content.
    """

    output: List[Document]

    def __init__(self, document_path: Path, document_type: str, **kwargs: dict):

        self.document_path = str(document_path)
        self.document_type = document_type
        self.__dict__.update(kwargs)
        self.loader = self._set_loader()

    def _set_loader(self):
        """
        Instantiates a document loader based on the document type. PDF and Excel are allowed to be loaded.
        PDF document is loaded from the BaseLoader langchain class.
        Excel document is loaded from a custom loader inherited from the langchain class.
        """
        if self.document_type.lower() == "pdfdirectory":
            logger.info(
                "Using Langchain PyPDFDirectoryLoader for loading the documents."
            )
            return PyPDFDirectoryLoader(path=self.document_path)
        elif self.document_type.lower() == "customexcel":
            logger.info("Using customExcelLoader for loading the documents.")
            return customExcelLoader(
                file_path=self.document_path,
                # Get sheet_name from kwargs
                excel_sheet=self.kwargs["sheet_name"],
                # Get column_name from kwargs, if not specified, is set as an empty List [] as default
                excel_columns=self.kwargs.get("column_names", []),
            )
        else:
            raise NotImplementedError(
                f"Loader of document type {self.document_type} not available. Only loading of 'PDFDirectory' and 'CustomExcelLoader' is currently implemented."
            )

    def _check_emptiness(self, page_content: str) -> bool:
        """
        Check if document content is empty of text, by evaluating whether
        there are at least 3 consecutive alphabet characters in there.
        """
        min_consecutive_chars = 3
        for letter in page_content:
            if letter.isalpha():
                min_consecutive_chars -= 1
            else:
                min_consecutive_chars = 3
            if min_consecutive_chars == 0:
                return False
        return True

    def _validate_loading(self, documents: List[Document]):
        """
        Validates if the text from documents is loaded
        Logs all documents from which no text has been loaded
        Raises info if one or more documents is not loaded
        Return a new list without documents which didn't pass emptiness check
        """
        sources_not_loaded = []
        valid_documents = []
        for doc in documents:
            if self._check_emptiness(doc.page_content):
                logger.warning(
                    f"Was not able to extract text from document {doc.metadata}; removed from the outputs."
                )
                sources_not_loaded.append(doc.metadata[dm.SOURCE])
            else:
                valid_documents.append(doc)

        sources_not_loaded = list(OrderedDict.fromkeys(sources_not_loaded))
        if sources_not_loaded:
            logger.info(
                f"Empty documents rendered from the following sources: {sources_not_loaded}"
            )

        return valid_documents

    def _add_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Adds an unique document ID to the document metadata
        Filters out duplicate documents
        """
        seen_hashes = []
        docs_with_metadata = []

        for doc in documents:
            new_metadata = copy.deepcopy(doc.metadata)
            page_content_hash = uuid_hash(content=doc.page_content)

            # Ensure that only docs with an unique documentID, which is the hash of the content, are kept
            if page_content_hash not in seen_hashes:
                new_metadata[dm.DOCUMENT_ID] = page_content_hash
                doc_with_new_metadata = Document(
                    page_content=doc.page_content, metadata=new_metadata
                )
                docs_with_metadata.append(doc_with_new_metadata)

            seen_hashes.append(page_content_hash)

        logger.info(
            f"Filtered out {len(seen_hashes) - len(docs_with_metadata)} incoming documents that were duplicate"
        )

        return docs_with_metadata

    def load(self) -> List[Document]:
        """
        Loads the documents, validates whether they are properly loaded,
        adds metadata, and removes duplicate documents.
        """
        documents = self.loader.load()
        valid_documents = self._validate_loading(documents)
        valid_documents = self._add_metadata(valid_documents)
        logger.info(f"Loaded {len(valid_documents)} documents")
        return valid_documents


if __name__ == "__main__":
    path = Path(__file__).parent.parent.joinpath(
        "test", "test_data", "excel_files", "test_document_loader_excel.xlsx"
    )

    document_type = "customexcel"
    excel_sheet = "Book"
    excel_columns = []
    kwargs = {"sheet_name": excel_sheet, "column_names": excel_columns}

    document_loader = DocumentLoader(
        document_path=path, document_type=document_type, kwargs=kwargs
    )
    documents = document_loader.load()
    print(documents)
    print(len(documents))
