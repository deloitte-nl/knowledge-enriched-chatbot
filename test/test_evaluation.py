import pytest
from langchain.schema import Document

import config as pm
from src.constants import DocumentMetadata
from src.eval.synthetic_data_generator import ContextQuestionDatasetGenerator
from src.utils import initialize_logger, set_openai_env_vars

logger = initialize_logger(__name__)

dm = DocumentMetadata()

set_openai_env_vars(pm.PATH_TO_OPENAI_CONFIG)

@pytest.fixture
def data_generator():
    return ContextQuestionDatasetGenerator()


@pytest.fixture
def documents():
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

def test_generate_dataset(data_generator, documents):
    dataset = data_generator.generate(chunked_documents=documents, dataset_size=2)
    assert isinstance(dataset, dict)
    assert len(dataset) == 2
