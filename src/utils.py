import json
import logging
import os
import uuid

import tiktoken
from langchain.chat_models import AzureChatOpenAI

from config import LOGGING_LEVEL


def uuid_hash(content: str) -> str:
    """Create a unique hash from the page content of a document."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))


def initialize_logger(name):
    """Initialize a logger with a specific name and logging format."""
    logging.basicConfig(
        level=LOGGING_LEVEL,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(name)
    return logger


def set_openai_env_vars(path_to_openai_config: str):
    """Set environment variables for the OpenAI API based on a config JSON file."""
    with open(path_to_openai_config, "rt", encoding="utf-8") as f:
        openai_config = json.load(f)

    for k, v in openai_config.items():
        os.environ[k] = v


def create_Azure_LLM(
    temperature: float,
    model_name: str = "gpt-35-turbo",
    deployment_name: str = None,
    api_version: str = "2023-05-15",
) -> AzureChatOpenAI:
    """Set the LLM to generate the synthetic dataset."""
    if deployment_name is None:
        deployment_name = model_name
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        model_name=model_name,
        openai_api_version=api_version,
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    return llm


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
