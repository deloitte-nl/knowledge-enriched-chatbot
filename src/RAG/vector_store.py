import copy
from pathlib import Path
from typing import List

from chromadb import PersistentClient
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

from src.constants import DocumentMetadata
from src.utils import initialize_logger, uuid_hash

logger = initialize_logger(__name__)
dm = DocumentMetadata()

DEFAULT_OPENAI_MODEL = "text-embedding-ada-002"
OPENAI_API_CHUNKSIZE = 1
DEFAULT_HUGGING_FACE_MODEL = "sentence-transformers/all-mpnet-base-v2"


class ChromaVectorStore:
    """
    Chroma storage for chunked documents and embeddings:
        1. Embeds chunked documents.
        2. Persists to or loads ChromaDB from path.
        3. Optionally embeds and adds new chunked documents to existing DB.

    Args:
        chunked_documents: List of chunked Langchain Documents with metadata.
        db_path: Path to store the ChromaDB.
        embedding_source: Name of the source of the embedding model.
        embedding_model: Name of the embedding model (ada for MVP). Defaults to None.
        collection_name: Name of the ChromaDB collection. Defaults to "chunked_document_embeddings".

    Returns:
        List[Document]: List of chunked Langchain Documents with metadata.
    """

    output: List[Document]

    def __init__(
        self,
        chunked_documents: List[Document],
        db_path: Path,
        embedding_source: str,
        embedding_model: str = None,
        collection_name: str = "chunked_document_embeddings",
    ):

        self.collection_name = collection_name
        self.embedding_source = embedding_source
        self.embedding_model = embedding_model
        self.embedder = self._set_embedder()
        self.client = PersistentClient(path=str(db_path))
        self.chromadb = self._set_chroma_db()
        self.add_documents(chunked_documents=chunked_documents)

    def _set_embedder(self):
        """
        Sets the document embedder based on the input embedding model or embedding source.
        If no embedding model is given, a default model is used.
        """
        if self.embedding_source.lower() == "openai":
            logger.info("Using OpenAI embedding model to embed the documents")
            if self.embedding_model is None:
                self.embedding_model = DEFAULT_OPENAI_MODEL
            return AzureOpenAIEmbeddings(
                deployment=self.embedding_model,
                model=self.embedding_model,
                chunk_size=OPENAI_API_CHUNKSIZE,
            )

        elif self.embedding_source.lower() == "huggingface":
            logger.info("Using Hugging Face embedding model to embed the documents")
            if self.embedding_model is None:
                self.embedding_model = DEFAULT_HUGGING_FACE_MODEL
            return HuggingFaceEmbeddings(model_name=self.embedding_model)

        else:
            raise NotImplementedError(
                f"Embedding source {self.embedding_source} not available. Only embedding models from 'HuggingFace' or 'OpenAI' are currently available."
            )

    def _filter_processed_documents(
        self, processed_docs: List[Document], incoming_docs: List[Document]
    ) -> List[Document]:
        """
        Filter already processed documents from the new incoming chunked documents.

        Args:
            processed_docs: List of processed documents already in the DB.
            incoming_docs: List of incoming chunked documents.

        Returns:
            List of unprocessed documents.
        """
        logger.info("Check if documents already exist in DB")

        # Create a hash of each of the incoming chunked documents
        hashes_incoming_docs = [
            (doc, uuid_hash(doc.page_content)) for doc in incoming_docs
        ]
        # Create a hash of each of the processed documents that are already in the DB
        hashes_processed_docs = [uuid_hash(doc) for doc in processed_docs["documents"]]
        # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        unprocessed_docs = [
            doc
            for doc, hash in hashes_incoming_docs
            if hash not in hashes_processed_docs
        ]

        logger.info(
            f"Filtered out {len(incoming_docs) - len(unprocessed_docs)} incoming documents that were already processed"
        )
        return unprocessed_docs

    def _validate_embedding_model(self, metadatas: List[dict]):
        """
        Validate whether the input embedding model is the same model used to embed the documents already in the vector store.

        Args:
            metadatas: List of metadata of the processed documents.

        Raises:
            ValueError: If the input embedding model does not match the stored embedding model.
        """
        model_embedded_documents = metadatas[0][dm.EMBEDDING_MODEL]
        if self.embedding_model != model_embedded_documents:
            raise ValueError(
                f"""Model {model_embedded_documents} is used to embed the documents in the existing vector store.
                    This does not match model {self.embedding_model} given as input parameter in the current run."""
            )

    def _add_embedding_model_to_metadata(self, metadatas: dict) -> List[dict]:
        """
        Add the embedding model used to embed the documents to the metadata.

        Args:
            metadatas: Metadata of the documents.

        Returns:
            List of metadata with the embedding model added.
        """
        new_metadatas = []
        for metadata in metadatas:
            new_metadata = copy.deepcopy(metadata)
            new_metadata[dm.EMBEDDING_MODEL] = self.embedding_model
            new_metadatas.append(new_metadata)
        return new_metadatas

    def _set_chroma_db(self) -> Chroma:
        """Sets the ChromaDB."""
        logger.info("Initializing ChromaDB")

        _ = self.client.get_or_create_collection(self.collection_name)

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            client=self.client,
        )

    def add_documents(self, chunked_documents: List[Document]):
        """
        Add new incoming chunked documents to the ChromaDB.
        Check if they are not already processed and stored in the ChromaDB.
        Validate if the input embedding model is the same as used for the stored embeddings (if present).
        Add metadata regarding the embedding model.

        Args:
            chunked_documents: List of incoming chunked documents.
        """

        collection = self.client.get_or_create_collection(self.collection_name)

        n_docs_in_collection = collection.count()
        processed_docs = collection.get()

        if n_docs_in_collection > 0:
            logger.info(f"ChromaDB already contains {n_docs_in_collection} documents.")
            self._validate_embedding_model(processed_docs["metadatas"])
            chunked_documents = self._filter_processed_documents(
                processed_docs=processed_docs, incoming_docs=chunked_documents
            )

        if len(chunked_documents) > 0:
            logger.info(
                f"Adding {len(chunked_documents)} new incoming chunked documents"
            )

            documents = []
            metadatas = []
            ids = []

            for doc in chunked_documents:
                documents.append(doc.page_content)
                metadatas.append(doc.metadata)
                ids.append(
                    f"{doc.metadata[dm.DOCUMENT_ID]}.{doc.metadata[dm.NTH_CHUNK]}"
                )

            metadatas = self._add_embedding_model_to_metadata(metadatas)

            self.chromadb.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    def get_documents(self):
        """Get documents from the ChromaDB."""
        return self.chromadb.get(include=["documents", "metadatas"])
