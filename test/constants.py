from pathlib import Path

TEST_DB_PATH_VS = Path(__file__).parent.joinpath("chromaDB_VS")
TEST_DB_PATH_QA = Path(__file__).parent.joinpath("chromaDB_QA")
TEST_PROMPT_PATH = Path(__file__).parent.joinpath("test_data", "prompt_messages.json")