from typing import List
import pytest
from langchain.schema import Document
from chromadb import Collection
from src.constants import DocumentMetadata
from src.RAG.vector_store import ChromaVectorStore
from test.constants import TEST_DB_PATH_VS
import shutil

shutil.rmtree(TEST_DB_PATH_VS, ignore_errors=True)

dm = DocumentMetadata()

def collection_to_documents(chroma_collection: Collection) -> List[Document]:
    """
    Convert the content of a chromaDB to a list of Document
    """
    collected_docs = chroma_collection.get()
    page_contents = collected_docs["documents"]
    metadatas = collected_docs["metadatas"]

    documents = []
    for page_content, metadata in list(zip(page_contents, metadatas)):
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


@pytest.fixture
def chunked_documents():
    """
    Returns testing input for the document chunker"
    """
    return [
        Document(
            page_content="""This step involves defining the key performance indicators (KPIs) for your project.""",
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
                dm.DOCUMENT_ID: 0,
                dm.NTH_CHUNK: 0,
            },
        ),
        Document(
            page_content="""These should be clear, measurable outcomes that align with your business objectives.""",
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
                dm.DOCUMENT_ID: 0,
                dm.NTH_CHUNK: 1,
            },
        ),
    ]


@pytest.fixture
def new_documents():
    """
    Returns testing input for the document chunker"
    """
    return [
        Document(
            page_content="""KPIs will provide a quantifiable outcome.""",
            metadata={
                dm.SOURCE: "test_data\\test_document_loader_pdf.pdf",
                dm.PAGE: 0,
                dm.DOCUMENT_ID: 1,
                dm.NTH_CHUNK: 0,
            },
        )
    ]


@pytest.fixture
def chroma_db(chunked_documents):
    """
    Returns chroma DB
    """
    return ChromaVectorStore(
        chunked_documents=chunked_documents,
        db_path=TEST_DB_PATH_VS,
        embedding_source="HuggingFace",
    )


def test_page_contents_in_vector_store(chroma_db, chunked_documents):
    """
    Assert that all the input documents are correctly loaded in the ChromaDB
    """
    page_contents_chromadb = chroma_db.chromadb._collection.get()["documents"]

    assert page_contents_chromadb == [doc.page_content for doc in chunked_documents]


def test_metadatas_in_vector_store(chroma_db, chunked_documents):
    """
    Assert that all the metadatas are correctly loaded in the ChromaDB
    """

    # get the metadatas after loading them in the ChromaDB
    metadatas_chromadb = [
        sorted(metadata.items())
        for metadata in chroma_db.chromadb._collection.get()["metadatas"]
    ]

    # get the expected metadatas, which are the metadatas before loading into the chromaDB plus the embedding model used
    metadatas = [
        sorted(
            list(doc.metadata.items())
            + [("embedding_model", "sentence-transformers/all-mpnet-base-v2")]
        )
        for doc in chunked_documents
    ]

    assert metadatas_chromadb == metadatas


def test_adding_existing_docs(chroma_db, chunked_documents):
    """
    Assert that adding the same documents will not lead to duplicate documents being loaded
    """
    collection = chroma_db.chromadb._collection
    chroma_db.add_documents(chunked_documents)

    assert collection.count() == 2


def test_adding_additional_docs_content(chroma_db, chunked_documents, new_documents):
    """
    Assert that adding adding additional documents will lead to a ChromaDB with existing and new documents
    """
    chroma_db.add_documents(new_documents)
    page_contents_chromadb = chroma_db.chromadb._collection.get()["documents"]

    assert page_contents_chromadb == [
        doc.page_content for doc in chunked_documents + new_documents
    ]


def test_embeddings_in_vector_store(chroma_db):
    """
    Assert that embeddings are created in the ChromaDB
    """
    embeddings = chroma_db.chromadb._collection.get(include=["embeddings"])[
        "embeddings"
    ]

    assert len(embeddings[-1]) > 384


def test_raising_error_wrong_embedder(new_documents):
    """
    Assert correct error message is raised when adding new documents
    using a different embedding model than the original saved vectorstore has used
    """

    with pytest.raises(ValueError):
        ChromaVectorStore(
            chunked_documents=new_documents,
            db_path=TEST_DB_PATH_VS,
            embedding_source="OpenAI",
        )
