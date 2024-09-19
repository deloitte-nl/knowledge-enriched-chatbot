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

# You have received a copy of the GNU Affero General Public License
# and our supplemental terms in LICENSE.MD in the root folder.  


from pathlib import Path

import pandas as pd

from src.utils import initialize_logger

logger = initialize_logger(__name__)


class ExcelPreprocessor:
    """
    Excel preprocessing class that:
        1. Reads an Excel file.
        2. Runs preprocessing functionalities.
        3. Saves the preprocessed Excel file.

    Args:
        excel_file_path: Location of the Excel file to be preprocessed.
        output_file_path: Location of the preprocessed Excel file.
    """

    def __init__(self, excel_file_path: Path, output_file_path: Path):

        self.excel_file_path = excel_file_path
        self.output_file_path = output_file_path
        self.excel_file = pd.read_excel(self.excel_file_path, sheet_name=None)

    def merge_columns(
        self, column_1: str, column_2: str, sheet_name: str, merged_col_name: str = None
    ):
        """
        Merge columns in an Excel sheet.

        Args:
            column_1: The first column to merge.
            column_2: The second column to merge.
            sheet_name: The sheet name where the columns are located.
            merged_col_name: The name for the merged column. Defaults to None.
        """
        # create new column name
        if not merged_col_name:
            merged_col_name = " ".join([column_1, column_2])

        # get sheet
        df_sheet = self.excel_file[sheet_name]

        # remove empty records
        df_sheet = df_sheet.dropna(subset=[column_1, column_2])

        # merge columns
        df_sheet[merged_col_name] = (
            df_sheet[column_1].astype(str) + " " + df_sheet[column_2].astype(str)
        )
        logger.info(f"Merged columns '{column_1}' and '{column_2}'")

        self.excel_file[sheet_name] = df_sheet

    def save(self):
        """Save the preprocessed Excel file."""
        with pd.ExcelWriter(self.output_file_path, engine="openpyxl") as writer:
            for sheet, df in self.excel_file.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
