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

import re
import warnings
import os
import pandas as pd
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from pathlib import Path

from src.utils import create_Azure_LLM, initialize_logger

logger = initialize_logger(__name__)
# TODO: add logging statements


class CypherQARetriever:
    """
    A module that makes a connection to a graph database, then prompts an LLM to write a Cypher query to
    answer a question using this graph database. The output of the Cypher query is returned and the LLM also
    provides an answer based on this graph context.


    Args:
        llm: The Azure LLM to be used.
        temperature: The temperature setting for the LLM, affecting randomness in the output.
        num_examples: The number of user questions and corresponding Cypher examples to add to the prompt for few-shot learning.
        examples_path: Path to the Excel file with columns 'Question' and 'Answer', containing example questions and corresponding correctly formulated Cypher queries.
        prompt_path: The prompt with instructions aimed at the LLM that translates a query to Cypher.
    """

    def __init__(
        self,
        llm: AzureChatOpenAI = None,
        temperature: int = 0,
        num_examples: int = 4,
        examples_path: Path = None,
        prompt_path: Path = None,
    ):
        if llm is None:
            self.llm = create_Azure_LLM(temperature=temperature)
        else:
            self.llm = llm

        logger.info("Initialize the Neo4jGraph instance")
        self.graph = Neo4jGraph(
            url=os.environ.get("GRAPH_ENDPOINT"),
            username=os.environ.get("GRAPH_USERNAME"),
            password=os.environ.get("GRAPH_PASSWORD"),
        )

        self.num_examples = num_examples
        self.examples_path = examples_path
        self.prompt_path = prompt_path

    def _clean_cypher_examples(self, input_string):
        """
        Clean the output from the SemanticSimilarityExampleSelector,
        so that only the user questions and cypher queries are returned.

        Args:
            input_string: The example selector output string which should be cleaned.

        Returns:
            Cleaned string with user questions and cypher queries.
        """

        # Remove the introductory sentence ending in a question mark
        cleaned_string = input_string.replace(
            "Give the cypher code based on the user question\n\n", ""
        )
        pattern = r"\n\n\s*([^\s].*)?\?\s*$"

        # Remove the last sentence ending in a question mark
        cleaned_string = re.sub(pattern, "", cleaned_string)

        return cleaned_string

    def _get_cypher_examples(self, user_question) -> str:
        """
        Select cypher examples using the langchain SemanticSimilarityExampleSelector
        from a list of user questions and corresponding cypher examples.
        These could be used to enter into an LLM prompt to improve the quality of cypher generation.

        Args:
            user_question: The user question for which we want to select the top k most
            similar examples from our list of example questions.

        Returns:
            A string with cypher generation examples that can be inserted into the prompt.
        """

        examples_df = pd.read_excel(self.examples_path)
        examples_list = examples_df.to_dict(orient="records")

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
        )

        example_prompt = PromptTemplate(
            input_variables=["Question", "Answer"],
            template="{Question}\n{Answer}",
        )

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # The list of examples available to select from.
            examples_list,
            # The embedding class used to produce embeddings which are used to measure semantic similarity.
            embeddings,
            # The VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # The number of examples to produce.
            k=self.num_examples,
        )

        similar_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Give the cypher code based on the user question",
            suffix="{question}\n",
            input_variables=["question"],
        )

        examples = similar_prompt.format(question=user_question)
        return self._clean_cypher_examples(examples)

    def _get_cypher_prompt(self, examples) -> PromptTemplate:
        """
        Create a prompt for cypher generation using the langchain
        PromptTemplate class and potentially add in examples.

        We have to insert the examples directly into the text before
        creating the prompt template, because adding other options to
        the prompt is currently not supported by GraphCypherQAChain.

        Args:
            examples: A string with cypher generation examples that should be inserted into the prompt.

        Returns:
            A PromptTemplate object with the cypher generation examples.
        """
        with open(self.prompt_path, "rt", encoding="utf-8") as f:
            system_message_template = f.read()

        CYPHER_TEMPLATE = system_message_template.format(examples=examples)

        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], template=CYPHER_TEMPLATE
        )

        return CYPHER_GENERATION_PROMPT

    def run_KG_qa(self, Q, verbose=False) -> dict:
        """
        Run a question answering LLM.

        Args:
            Q: The user question to be answered.
            verbose: Whether the cypher output and intermediate results should be printed.

        Returns:
            The result containing the answer and intermediate steps.
        """
        if self.num_examples > 0 and self.examples_path is not None:
            cypher_examples = self._get_cypher_examples(user_question=Q)
        else:
            cypher_examples = ""

        CYPHER_GENERATION_PROMPT = self._get_cypher_prompt(cypher_examples)

        chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            validate_cypher=True,
            verbose=verbose,
            return_intermediate_steps=True,
        )

        result = chain(Q)
        return result

    def close(self):
        """Close the Neo4j connection."""
        self.graph._driver.close()
