import logging
from pathlib import Path

import pytest
from langchain.schema import Document

from src.constants import DocumentMetadata
from src.RAG.document_loader import DocumentLoader

dm = DocumentMetadata()


@pytest.fixture
def document_loader():
    """
    Instantiate the documentLoader
    """
    document_path = Path(__file__).parent.joinpath("test_data")
    document_type = "PDFDirectory"
    document_loader = DocumentLoader(
        document_path=document_path, document_type=document_type
    )
    return document_loader


@pytest.fixture
def loaded_data(document_loader):
    """
    Loads the test data using the documentLoader
    """
    return document_loader.load()


@pytest.fixture
def documents():
    """
    Returns the output as we would expect it from a correctly working documentLoader given the test data in 'test_data'
    """
    return [
        Document(
            page_content="""This step involves defining the key performance indicators (KPIs) for your project. These should be clear, measurable outcomes that align with your business objectives. KPIs will provide a quantifiable """,
            metadata={
                dm.DOCUMENT_ID: 0,
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
            },
        ),
        Document(
            page_content="""What is the baseline performance? Define a baseline performance level, based on the current system or a considered good baseline. A baseline provides a point of comparison to gauge the AI system’s performance and improvements.""",
            metadata={
                dm.DOCUMENT_ID: 1,
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 1,
            },
        ),
        Document(
            page_content="""A complete solution architecture that details how the AI system fits into and enhances the current workflow. The solution architecture will serve as a blueprint for system implementation and integration.""",
            metadata={
                dm.DOCUMENT_ID: 2,
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 2,
            },
        ),
    ]


def test_n_loaded_docs(loaded_data):
    """
    Assert that all three test documents have been loaded
    """
    assert len(loaded_data) == 3


def test_metadata_document_id(loaded_data, documents):
    """
    Assert that the metadata document ID is unique
    """
    loaded_metadata = [doc.metadata[dm.DOCUMENT_ID] for doc in loaded_data]

    assert len(set(loaded_metadata)) == 3


def test_metadata_source(loaded_data, documents):
    """
    Assert that the metadata source is correctly construed
    """
    loaded_metadata = [Path(doc.metadata[dm.SOURCE]).stem for doc in loaded_data]
    metadata = [Path(doc.metadata[dm.SOURCE]).stem for doc in documents]

    assert loaded_metadata == metadata


def test_metadata_page(loaded_data, documents):
    """
    Assert that the metadata page is correctly construed
    """
    loaded_metadata = [doc.metadata[dm.PAGE] for doc in loaded_data]
    metadata = [doc.metadata[dm.PAGE] for doc in documents]

    assert loaded_metadata == metadata


def test_page_content(loaded_data, documents):
    """
    Assert that the complete page content is loaded
    """
    loaded_page_content = [
        doc.page_content.replace("\n", "").replace("\\n", "").replace(" ", "")
        for doc in loaded_data
    ]
    page_content = [doc.page_content.replace(" ", "") for doc in documents]

    assert loaded_page_content == page_content


def test_unloaded_sources(document_loader, caplog):
    """
    Assert correct error message is raised when loading an empy document
    """
    empty_docs = [
        Document(page_content="", metadata={dm.SOURCE: "empty_doc", dm.PAGE: 0}),
        Document(
            page_content="""
                 What is the baseline performance? Define a baseline performance level, based on the current system or a considered good baseline. 
                 A baseline provides a point of comparison to gauge the AI system’s performance and improvements.
                 """,
            metadata={dm.SOURCE: "test_data\\test_document_loader_pdf.pdf", dm.PAGE: 1},
        ),
        Document(
            page_content="    f    2    gh/n dw023ri  ><6",
            metadata={dm.SOURCE: "empty_doc_2", dm.PAGE: 2},
        ),
    ]

    caplog.set_level(logging.INFO)

    document_loader._validate_loading(empty_docs)

    assert (
        "Was not able to extract text from document {'source': 'empty_doc', 'page': 0}; removed from the outputs."
        in caplog.text
    )
    assert (
        "Was not able to extract text from document {'source': 'empty_doc_2', 'page': 2}; removed from the outputs."
        in caplog.text
    )
    assert (
        "Empty documents rendered from the following sources: ['empty_doc', 'empty_doc_2']"
        in caplog.text
    )
