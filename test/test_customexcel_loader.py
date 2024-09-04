import logging
from pathlib import Path

import pytest
from langchain.schema import Document

from src.constants import DocumentMetadata
from src.RAG.document_loader import DocumentLoader

dm = DocumentMetadata()


@pytest.fixture
def document_loader(request):
    """
    Instantiate the documentLoader
    """
    document_path = Path(__file__).parent.joinpath(
        "test_data", "excel_files", "test_document_loader_excel.xlsx"
    )
    document_type = "customexcel"
    if hasattr(request, "param"):
        excel_sheet = request.param.get("excel_sheet")
        excel_columns = request.param.get("excel_columns")
    else:
        excel_sheet = "Toy"
        excel_columns = ["Colour"]
    kwargs = {"sheet_name": excel_sheet, "column_names": excel_columns}
    document_loader = DocumentLoader(
        document_path=document_path, document_type=document_type, kwargs=kwargs
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
            page_content="""Green""",
            metadata={
                dm.SOURCE: "test_data/excel_files/test_document_loader_excel.xlsx",
                dm.PAGE_NUMBER: 1,
                dm.PAGE_NAME: "Toy",
                dm.COLUMN_NAME: "Colour",
                dm.CELL_INDEX: "B2",
                dm.DOCUMENT_ID: 0,
            },
        ),
        Document(
            page_content="""Yellow""",
            metadata={
                dm.SOURCE: "test_data/excel_files/test_document_loader_excel.xlsx",
                dm.PAGE_NUMBER: 1,
                dm.PAGE_NAME: "Toy",
                dm.COLUMN_NAME: "Colour",
                dm.CELL_INDEX: "B3",
                dm.DOCUMENT_ID: 0,
            },
        ),
        Document(
            page_content="""Blue""",
            metadata={
                dm.SOURCE: "test_data/excel_files/test_document_loader_excel.xlsx",
                dm.PAGE_NUMBER: 1,
                dm.PAGE_NAME: "Toy",
                dm.COLUMN_NAME: "Colour",
                dm.CELL_INDEX: "B5",
                dm.DOCUMENT_ID: 0,
            },
        ),
        Document(
            page_content="""White""",
            metadata={
                dm.SOURCE: "test_data/excel_files/test_document_loader_excel.xlsx",
                dm.PAGE_NUMBER: 1,
                dm.PAGE_NAME: "Toy",
                dm.COLUMN_NAME: "Colour",
                dm.CELL_INDEX: "B7",
                dm.DOCUMENT_ID: 0,
            },
        ),
    ]


def test_n_loaded_docs(loaded_data):
    """
    Assert that all three test documents have been loaded
    """
    assert len(loaded_data) == 4


def test_metadata_document_id(loaded_data):
    """
    Assert that the metadata document ID is unique
    """
    loaded_metadata = [doc.metadata[dm.DOCUMENT_ID] for doc in loaded_data]

    assert len(set(loaded_metadata)) == 4


def test_metadata_source(loaded_data, documents):
    """
    Assert that the metadata source is correctly construed
    """
    loaded_metadata = [Path(doc.metadata[dm.SOURCE]).stem for doc in loaded_data]
    metadata = [Path(doc.metadata[dm.SOURCE]).stem for doc in documents]

    assert loaded_metadata == metadata


def test_metadata_page_name(loaded_data, documents):
    """
    Assert that the metadata page is correctly construed
    """
    loaded_metadata = [doc.metadata[dm.PAGE_NAME] for doc in loaded_data]
    metadata = [doc.metadata[dm.PAGE_NAME] for doc in documents]

    assert loaded_metadata == metadata


def test_metadata_page_number(loaded_data, documents):
    """
    Assert that the metadata page is correctly construed
    """
    loaded_metadata = [doc.metadata[dm.PAGE_NUMBER] for doc in loaded_data]
    metadata = [doc.metadata[dm.PAGE_NUMBER] for doc in documents]

    assert loaded_metadata == metadata


def test_metadata_column_name(loaded_data, documents):
    """
    Assert that the metadata page is correctly construed
    """
    loaded_metadata = [doc.metadata[dm.COLUMN_NAME] for doc in loaded_data]
    metadata = [doc.metadata[dm.COLUMN_NAME] for doc in documents]

    assert loaded_metadata == metadata


def test_metadata_cell_index(loaded_data, documents):
    """
    Assert that the metadata page is correctly construed
    """
    loaded_metadata = [doc.metadata[dm.CELL_INDEX] for doc in loaded_data]
    metadata = [doc.metadata[dm.CELL_INDEX] for doc in documents]

    assert loaded_metadata == metadata


def test_page_content(loaded_data, documents):
    """
    Assert that the complete page content is loaded
    """
    # why are the spaces replaced by "", but keeping the replace will not influence the test on excel
    # one small problem: need to check how the outputs of contents with enters look like in cells
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


@pytest.mark.parametrize(
    "document_loader, excel_sheet, excel_columns",
    [({"excel_sheet": "Toys"}, "Toys", ["Colour"])],
    indirect=["document_loader"],
)
def test_incorrect_sheet_name(document_loader, excel_sheet, excel_columns):
    """
    Assert correct error message is raised when sheet name is incorrect
    """
    with pytest.raises(
        ValueError, match=f"Sheet {excel_sheet} is not found in the documents."
    ):
        document_loader.load()


@pytest.mark.parametrize(
    "document_loader, excel_sheet, excel_columns",
    [
        (
            {"excel_sheet": "Toy", "excel_columns": ["Colou", "Amount"]},
            "Toy",
            ["Colou", "Amount"],
        )
    ],
    indirect=["document_loader"],
)
def test_incorrect_column_name(document_loader, excel_sheet, excel_columns):
    """
    Assert correct error message is raised when column name is incorrect
    """
    cols = ", ".join(excel_columns)

    with pytest.raises(
        ValueError,
        match=f"The following column\(s\) do not exist: {cols}",
    ):
        document_loader.load()
