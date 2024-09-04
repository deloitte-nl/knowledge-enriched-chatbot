import pytest
from langchain.schema import Document

from src.constants import DocumentMetadata
from src.RAG.document_chunker import DocumentChunker

dm = DocumentMetadata()


@pytest.fixture
def documents():
    """
    Returns testing input for the document chunker
    """
    return [
        Document(
            page_content="""This step involves defining the key performance indicators (KPIs) for your project. 
                 These should be clear, measurable outcomes that align with your business objectives. 
                 KPIs will provide a quantifiable outcome. """,
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
                dm.DOCUMENT_ID: 0,
            },
        ),
        Document(
            page_content="""
                 What is the baseline performance? 
                 Define a baseline performance level, based on the current system or a considered good baseline. 
                 A baseline provides a point of comparison to gauge the AI system’s performance and improvements.
                 """,
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 1,
                dm.DOCUMENT_ID: 1,
            },
        ),
        Document(
            page_content="""A complete solution architecture that details how the AI system fits into and enhances the current workflow. 
                 The solution architecture will serve as a blueprint for system implementation and integration.""",
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 2,
                dm.DOCUMENT_ID: 2,
            },
        ),
    ]


@pytest.fixture
def chunked_data(documents):
    """
    Returns the expected output of the document chunker based on testing input
    """
    document_chunker = DocumentChunker(
        chunking_strategy="sentenceSplitting",
        kwargs={"chunk_overlap": 5, "chunk_size": 25},
    )
    return document_chunker.split_documents(documents=documents)


def test_n_loaded_docs(chunked_data):
    """
    Assert that after chunking, the amount of documents has increased
    """
    assert len(chunked_data) > 3


def test_nth_chunk(chunked_data):
    """
    Assert that the nth chunk metadata is correctly annotated
    """
    chunks_per_page = {0: [], 1: [], 2: []}
    for chunk in chunked_data:
        chunks_per_page[chunk.metadata[dm.PAGE]].append(chunk.metadata[dm.NTH_CHUNK])
    tot_chunks = (
        max(chunks_per_page[0])
        + 1
        + max(chunks_per_page[1])
        + 1
        + max(chunks_per_page[2])
        + 1
    )
    assert len(chunked_data) == tot_chunks


def test_content_chunk(chunked_data):
    """
    Assert that the content of the chunked data is correct
    """
    assert (
        chunked_data[0].page_content
        == "This step involves defining the key performance indicators (KPIs) for your project."
    )
