from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain.chat_models import AzureChatOpenAI

from src.constants import DocumentMetadata

dm = DocumentMetadata()


@dataclass
class DocumentLoaderArguments:
    """Arguments pertaining to how the documents should be loaded."""

    document_path: str = field(metadata={"help": "location of the raw document(s)"})
    document_type: Optional[str] = field(
        default=None,
        metadata={"help": "type of document(s) to load, f.e. PDFDirectory"},
    )
    load_from_existing_vector_db: Optional[str] = field(
        default=False,
        metadata={
            "help": "Whether to load, chunk, embed documents and create a vectorDB or to read an existing vectorDB from path"
        },
    )


@dataclass
class ExcelLoaderArguments(DocumentLoaderArguments):
    """Arguments pertaining to how an Excel document should be loaded."""

    sheet_name: Optional[str] = field(
        default="Sheet1", metadata={"help": "The name of the Excel sheet to load"}
    )
    column_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Names of the Excel columns to load"},
    )


@dataclass
class VectorDBArguments:
    """Arguments for chunking and storing documents in a vector DB"""

    chunk_size: int = field(
        metadata={
            "help": "Size of the chunks to attempt to create, actual sizes might be larger or smaller due to keeping sentences intact"
        }
    )

    chunk_overlap: int = field(
        metadata={"help": "Overlap between the chunks (stride in sliding window)"}
    )

    db_path: Path = field(
        metadata={"help": "path to store or to load the Chroma database from"}
    )
    embedding_source: str = field(
        metadata={"help": "Source for the model used for embedding the documents"}
    )

    chunking_strategy: Optional[str] = field(
        default="tokenizedsentencesplitting",
        metadata={"help": "The strategy to use for chunking the documents"},
    )

    separator: Optional[str] = field(
        default=" ",
        metadata={"help": "The separator to insert between sentences within a chunk"},
    )
    pipeline: Optional[str] = field(
        default="en_core_web_sm",
        metadata={
            "help": "The model to use for splitting the documents into sentences"
        },
    )
    pipeline: Optional[str] = field(
        default="cl100k_base",
        metadata={
            "help": "The encoding for tokenized chunking. 'cl100k_base' is appropriate for gpt3.5"
        },
    )
    embedding_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the embedding model, if None it reverts to a default embedding model depending on the source"
        },
    )
    collection_name: Optional[str] = field(
        default="chunked_document_embeddings",
        metadata={
            "help": "Name of the ChromaDB collection used to store the documents and embeddings"
        },
    )


@dataclass
class QARetrievalArguments:
    """Arguments pertaining to the vector database QA retriever."""

    prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the json file containing the prompt messages"},
    )
    return_source_documents: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to include the source documents retrieved from the vector store in the LLM result"
        },
    )
    combine_document_strategy: Optional[str] = field(
        default="stuff_with_citation",
        metadata={
            "help": "The strategy used to combine the candidate chunked documents that are given to the LLM. Stuff with citation makes sure that the LLM receives citation metadata to cite in the answer"
        },
    )
    temperature: Optional[float] = field(
        default=0.1,
        metadata={"help": "The sampling temperature"},
    )

    citation_metadata: Optional[str] = field(
        default=dm.SOURCE,
        metadata={
            "help": "The metadata the LLM should receive and use for the source citation in the answer"
        },
    )

    similarity_search: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to filter the retrieved documents from the vectorDB by a similarity score threshold"
        },
    )

    similarity_score_threshold: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "The similarity score threshold used for filtering the retrieved documents"
        },
    )

    top_k_candidate_documents: Optional[int] = field(
        default=3,
        metadata={
            "help": "Maximum amount of candidate documents to retrieve for a specific question"
        },
    )

    retrieval_LLM: Optional[AzureChatOpenAI] = field(
        default=None,
        metadata={
            "help": "The Azure LLM model used for answering the question based on the retrieved context from the vector database"
        },
    )


@dataclass
class KGIntegratedQARetrievalArguments:
    """Arguments pertaining to the vector database and Knowledge Graph integrated QA retriever."""

    prompt_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a text file containing the prompt messages for the KG QA retriever"
        },
    )

    integrated_prompt_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a text file containing the prompt message for the integrated vector DB and KG retriever"
        },
    )

    examples_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to an Excel with columns 'Question' and 'Answer', containing example questions and corresponding correctly formulated cypher queries"
        },
    )

    num_examples: Optional[int] = field(
        default=4,
        metadata={
            "help": "The number of cypher examples to use in the cypherQA prompt"
        },
    )

    verbose: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Verboseness of the questions answering integrated QA retriever"
        },
    )

    cypher_LLM: Optional[AzureChatOpenAI] = field(
        default=None,
        metadata={
            "help": "The LLM model used for translating a natural language query to a Cypher query"
        },
    )

    integration_LLM: Optional[AzureChatOpenAI] = field(
        default=None,
        metadata={
            "help": "The LLM model used for integrating vector DB and KG information"
        },
    )


@dataclass
class EvaluationArguments:
    """Arguments for the RAG (Retrieval-Augmented Generation) Evaluation pipeline."""

    evaluate_synthetic_metrics: List = field(
        metadata={
            "help": "The metrics to use for synthetic evaluation. Options as listed in src.constants.py."
        },
    )
    evaluate_ground_truth_metrics: List = field(
        metadata={
            "help": "The metrics to use for ground truth evaluation. Options as listed in src.constants.py."
        },
    )

    dataset_size: Optional[int] = field(
        metadata={"help": "Number of context, question pairs to be generated."},
        default=10,
    )
    output_eval_dataset: Optional[bool] = field(
        metadata={
            "help": "If True, a dataset is returned and saved, if False a score dictionary is returned and saved."
        },
        default=True,
    )
    ground_truth_filepath: Optional[Path] = field(
        metadata={
            "help": "The file path to the ground_truth json file with question, answer pairs."
        },
        default=Path(__file__).parent.joinpath(
            "src", "eval", "ground_truth_template", "ground_truth_eval_template.csv"
        ),
    )
    retriever_to_evaluate: Optional[str] = field(
        default="vectordb",
        metadata={
            "help": "Whether the 'vectordb' retriever should be evaluated or the 'integrated' retriever."
        },
    )
    generative_LLM: Optional[AzureChatOpenAI] = field(
        default=None,
        metadata={
            "help": "The Azure LLM model used for generating the synthetic dataset questions."
        },
    )

    evaluative_LLM: Optional[AzureChatOpenAI] = field(
        default=None,
        metadata={
            "help": "The Azure LLM model used for evaluating the answers of the QA pipeline."
        },
    )