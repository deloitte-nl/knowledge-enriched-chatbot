import shutil
from test.constants import TEST_DB_PATH_QA, TEST_PROMPT_PATH
from typing import List

import pytest
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

import config as pm
from src.constants import DocumentMetadata
from src.RAG.qa_retriever import CustomQARetriever
from src.RAG.vector_store import ChromaVectorStore
from src.utils import set_openai_env_vars

shutil.rmtree(TEST_DB_PATH_QA, ignore_errors=True)

set_openai_env_vars(pm.PATH_TO_OPENAI_CONFIG)


dm = DocumentMetadata()


@pytest.fixture
def chunked_documents() -> List[Document]:
    """
    Returns input documents for the chromaDB"
    """
    return [
        Document(
            page_content="""This step involves defining the key performance indicators (KPIs) for your project.""",
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
                dm.DOCUMENT_ID: 0,
                dm.NTH_CHUNK: 0,
                dm.CELL_INDEX: "B3",
            },
        ),
        Document(
            page_content="""These should be clear, measurable outcomes that align with your business objectives.""",
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
                dm.DOCUMENT_ID: 0,
                dm.NTH_CHUNK: 1,
                dm.CELL_INDEX: "B2",
            },
        ),
        Document(
            page_content="""KPIs will provide a quantifiable outcome.""",
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
                dm.DOCUMENT_ID: 1,
                dm.NTH_CHUNK: 0,
                dm.CELL_INDEX: "A1",
            },
        ),
    ]


@pytest.fixture
def chroma_db_retriever(chunked_documents: List[Document]) -> VectorStoreRetriever:
    """
    Returns chroma DB
    """
    vector_store = ChromaVectorStore(
        chunked_documents=chunked_documents,
        db_path=TEST_DB_PATH_QA,
        embedding_source="HuggingFace",
    )

    return vector_store.chromadb.as_retriever()


@pytest.fixture
def qa_retriever(chroma_db_retriever) -> CustomQARetriever:
    """
    Returns the custom QA retriever
    """

    return CustomQARetriever(retriever=chroma_db_retriever)


def test_custom_prompt(chroma_db_retriever):
    """
    Assert that if a prompt path is given, the prompt template used for QA is read from path
    """
    qa_retriever = CustomQARetriever(
        retriever=chroma_db_retriever,
        prompt_path=TEST_PROMPT_PATH,
    )
    template_0 = qa_retriever.qa.combine_documents_chain.llm_chain.prompt.messages[
        0
    ].prompt.template
    template_1 = qa_retriever.qa.combine_documents_chain.llm_chain.prompt.messages[
        1
    ].prompt.template

    assert [template_0, template_1] == [
        "You are a chatbot that gives an answer to a question. Use the pieces of the following context to answer the question of the human. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\n",
        "Question: {question}\n",
    ]


def test_referenced_sources(qa_retriever, chunked_documents):
    """
    Assert that the source documents provided by the QA retriever can be traced back to the input documents
    """
    res = qa_retriever.qa("Will KPIs provide a quantifiable outcome?")
    source_contents = [[source.page_content] for source in res["source_documents"]]
    input_documents = [[doc.page_content] for doc in chunked_documents]
    not_referenced = [
        content for content in input_documents if content not in source_contents
    ]

    exp_n_not_referenced = len(input_documents) - len(source_contents)

    assert exp_n_not_referenced == len(not_referenced)
