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


import random
from typing import List

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from src.utils import initialize_logger, create_Azure_LLM, uuid_hash

logger = initialize_logger(__name__)


class ContextQuestionDatasetGenerator:
    """
    Context Question Generator class that:
        1. Sets up a connection to an AzureOpenAI LLM.
        2. Uses LLM and prompts to create {dataset_size} number of context, question pairs.
        3. Saves context, question pairs as JSON.

    Args:
        llm: The language model to use for generating a synthetic dataset. Defaults to None, in which case a new Azure LLM with a temperature of 0.8 will be created.
    """

    def __init__(
        self,
        llm: AzureChatOpenAI = None,
    ):
        if llm is None:
            self.llm = create_Azure_LLM(temperature=0.8)
        else:
            self.llm = llm

        self.question_generation_prompt = PromptTemplate.from_template(
            """
            Context information is below.

            ---------------------
            {context}
            ---------------------

            Assume the role of an Exam Creator. 
            Based only on the information provided, formulate one question that can be answered using just this context. 
            Ensure the question is specific and directly related to the details given.
            """
        )

    def generate(self, chunked_documents: List[Document], dataset_size: int) -> dict:
        """
        Generate the questions based on the provided context.

        Args:
            chunked_documents: The documents on which the synthetic dataset is generated.
            dataset_size: The number of context, question pairs to generate.

        Returns:
            The generated questions and their context.
        """

        logger.info(f"Starting dataset generation with {dataset_size} questions")

        dataset = {}
        for _ in range(dataset_size):
            document = random.choice(chunked_documents)
            context_id = document.metadata["document_id"]
            context = document.page_content
            prompt = self.question_generation_prompt.format(context=context)
            question = self.llm.predict(prompt)
            question_id = uuid_hash(question)
            dataset[question_id] = {
                "question": question,
                "context_id": context_id,
                "context": context,
            }

        logger.info("Dataset generation completed")
        return dataset
