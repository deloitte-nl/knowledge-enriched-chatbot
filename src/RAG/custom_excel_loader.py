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

from typing import List

from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.schema import Document

from src.utils import initialize_logger

logger = initialize_logger(__name__)


class customExcelLoader:
    """
    Custom Excel loading class for handling Excel documents:
        1. Loads documents from a specified path.
        2. Uses UnstructuredExcelLoader from Langchain to load Excel documents in HTML format.
        3. Customizes UnstructuredExcelLoader to enable loading from a specific sheet and column.
        4. Generates metadata: sheet_name, column_name, cell_index, etc.

    Args:
        file_path: Location of the document to be loaded.
        excel_sheet: Only one sheet is allowed to load at one time. If not specified or does not exist, an error will be raised.
        excel_columns: The names of the Excel columns specified are stored in the list. If not specified, all columns will be loaded.
                                   If the specified columns do not exist, an error will be raised.

    Returns:
        List[Document]: Each Document has two attributes:
            page_content: Content of documents. Each cell is one page_content in Excel.
            metadata: Contains information about each page_content.
    """

    output: List[Document]

    def __init__(self, file_path: str, excel_sheet: str, excel_columns: List[str] = []):

        self.file_path = file_path
        self.excel_sheet = excel_sheet
        self.excel_columns = excel_columns
        self.unstructured_excel_loader = UnstructuredExcelLoader(
            self.file_path, mode="elements", include_header=True
        )

    def _col_index_to_excel(self, col: int) -> str:
        """
        Convert a zero-based column index into the corresponding Excel-style column letter.

        Args:
            col: The column index.

        Returns:
            The Excel-style column letter.
        """
        excel_col = ""
        while col >= 0:
            remainder = col % 26
            excel_col = chr(65 + remainder) + excel_col
            col = col // 26 - 1
        return excel_col

    def _generate_metadata(
        self, doc: Document, column_name: str, col_idx: int, row_idx: int
    ) -> dict:
        """
        Generate metadata for cell contents, modifying the metadata created by Langchain.

        Args:
            doc: Document object containing metadata.
            column_name: Name of the column.
            col_idx: Column index.
            row_idx: Row index.

        Returns:
            Metadata for cell contents.
        """
        cell_address = self._col_index_to_excel(col_idx) + str(row_idx + 1)

        new_metadata = {
            "source": doc.metadata.get("source", None),
            "page_number": doc.metadata.get("page_number", None),
            "page_name": doc.metadata.get("page_name", None),
            "column_name": column_name,
            "cell_index": cell_address,
        }

        return new_metadata

    def _validate_column_names(self, document: Document):
        """
        Check if the specified columns exist in the loaded document.

        Args:
            document: Document object to validate.
        """
        non_existing_columns = []

        html_doc = self._get_html_from_document(document)
        headers = self._get_headers(html_doc)

        for column_name in self.excel_columns:
            if column_name not in headers:
                non_existing_columns.append(column_name)

        if non_existing_columns:
            raise ValueError(
                f"The following column(s) do not exist: {', '.join(non_existing_columns)}"
            )

    def _extract_cells(
        self,
        docs: List[Document],
    ) -> List[Document]:
        """
        Extract and process cell data from the document.
        If columns are not specified, all columns will be loaded.
        Empty cells will be skipped.

        Args:
            docs: List of Document objects.

        Returns:
            List of processed Document objects.
        """
        processed_cells = []

        document_soups = [self._get_html_from_document(doc) for doc in docs]
        headers_in_docs = [self._get_headers(soup) for soup in document_soups]

        if not self.excel_columns:
            logger.info(
                f"No excel columns specified, loading all columns: {headers_in_doc}"
            )
            self.excel_columns = headers_in_doc

        row_idx = -1
        for document_soup, headers_in_doc, doc in zip(
            document_soups, headers_in_docs, docs
        ):
            # Finds all table rows ("<tr>") in the parsed HTML ('soup')
            for _, row in enumerate(document_soup.find_all("tr")):

                # Finds all table data ("<td>") elements (cells) within the current row ("row")
                row_idx += 1
                for col_idx, cell in enumerate(row.find_all("td")):
                    # If the col_idx matches the index of the column that should be loaded
                    column_name = headers_in_doc[col_idx]
                    if column_name in self.excel_columns:
                        # If the cell contains text
                        if cell.get_text():
                            # Modify the metadata of the cell through _generate_metadata.
                            metadata = self._generate_metadata(
                                doc, column_name, col_idx, row_idx
                            )

                            # Create the elements of cells containing two attributes: page_content and metadata.
                            cell_doc = Document(
                                page_content=cell.get_text(), metadata=metadata
                            )
                            processed_cells.append(cell_doc)

        return processed_cells

    def _get_headers(self, document_soup: BeautifulSoup) -> List[str]:
        """Get the headers of the excel sheet from the Beautiful Soup read HTML text"""
        return [
            th.get_text(strip=True) for th in document_soup.find("thead").find_all("th")
        ]

    def _get_html_from_document(self, document: Document) -> BeautifulSoup:
        """Get the HTML text from a Langchain Document and read with BeautifulSoup"""
        return BeautifulSoup(document.metadata["text_as_html"], "html.parser")

    def _select_sheet(self, documents: List[Document]) -> List[Document]:
        """Select the loaded sheet. Raise an error if the sheet is not specified or the specified sheet does not exist"""
        sheets = [
            doc
            for doc in documents
            if doc.metadata.get("page_name") == self.excel_sheet
        ]
        if not sheets:
            raise ValueError(f"Sheet {self.excel_sheet} is not found in the documents.")
        return sheets

    def load(self) -> List[Document]:
        """
        Loads the documents using the unstructured excel loader
        Processes the HTML output and converts it to cell based content
        """
        loaded_sheets = self.unstructured_excel_loader.load()
        selected_sheet = self._select_sheet(loaded_sheets)
        for part_sheet in selected_sheet:
            self._validate_column_names(part_sheet)
        return self._extract_cells(selected_sheet)
