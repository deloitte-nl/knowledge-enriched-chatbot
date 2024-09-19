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

from pathlib import Path
from typing import List

import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import Document

from src.constants import MetricMetadata
from src.eval.rag_evaluation_metrics import RagEvaluationMetrics
from src.eval.synthetic_data_generator import ContextQuestionDatasetGenerator
from src.RAG.qa_retriever import CustomQARetriever
from src.RAG.vector_store import ChromaVectorStore
from src.utils import initialize_logger, uuid_hash

mm, rm = (MetricMetadata(), MetricMetadata.RagasMetadata())
logger = initialize_logger(__name__)


class EvaluationPipeline:
    """
    Evaluates a RAG question answering pipeline using the Ragas framework and package. There are two approaches:
        1. Ground truth evaluation: Evaluate using ground truth dataset with question and answer pairs
        2. Synthetic evaluation: Evaluate by generating and evaluating on a synthetic dataset

    Args:
        qa_retriever: Initialized instance of the RAG module custom QARetriever.
        vector_store: Initialized instance of the RAG module ChromaDB Vector store. Needed for synthetic evaluation.
        output_folder: Location to save the evaluation results. Defaults to "data/evaluation_runs".

    Returns:
        pd.DataFrame: Evaluation results returned as a variable when the `evaluate` method is called.
        JSON: Evaluation results saved to the evaluation_runs folder.
    """

    def __init__(
        self,
        qa_retriever: CustomQARetriever,
        vector_store: ChromaVectorStore = None,
        output_folder: str = "data\evaluation_runs",
    ):

        self.qa_retriever = qa_retriever
        if vector_store:
            self.documents = self._load_from_vector_store(vector_store)
        else:
            self.documents = None
        self.output_folder = output_folder

    def _load_from_vector_store(
        self, vector_store: ChromaVectorStore
    ) -> List[Document]:
        """
        Loads a List of Documents from a ChromaDB vectorstore and transforms it into the Langchain Document format.

        Args:
            vector_store: The vector store to load documents from.

        Returns:
            List of Langchain Document objects.
        """
        documents = vector_store.get_documents()

        docs_langchain_format = [
            Document(page_content=document, metadata={"document_id": id})
            for id, document in zip(documents["ids"], documents["documents"])
        ]

        if not docs_langchain_format:
            raise ValueError(
                f"Vector store is empty. Please create a ChromaDB vector store to evaluate on."
            )

        return docs_langchain_format

    def _run_rag(self, questions: dict) -> dict:
        """
        Queries the RAG module with the questions from the generated synthetic dataset.

        Args:
            questions: A dictionary of questions.
                dict
                {
                    question_id: {
                        question: str
                    }, ...
                }

        Returns:
            A dictionary with answers and contexts.
                dict
                {
                    question_id: {
                        question: str,
                        answer: str,
                        contexts: List[uuid.int, str]}, ..
                }
        """
        for question_id in questions.keys():
            (
                questions[question_id][rm.GENERATED_ANSWER],
                questions[question_id][rm.CONTEXTS],
            ) = (
                {},
                {},
            )

            result = self.qa_retriever.qa(questions[question_id][rm.QUESTION])

            context = {
                f"{doc.metadata['document_id']}.{doc.metadata['nth_chunk']}": doc.page_content
                for doc in result["source_documents"]
            }

            questions[question_id][rm.GENERATED_ANSWER] = result["result"]
            questions[question_id][rm.CONTEXTS] = context

        return questions

    def _format_ground_truth_dataset(self, ground_truth_filepath: str) -> dict:
        """Transforms a CSV into JSON structure to fit the evaluation pipeline.

        Args:
            ground_truth_filepath: The file path to the ground truth CSV file.

        Returns:
            The formatted ground truth dataset.
        """
        df = pd.read_csv(ground_truth_filepath, sep=";")

        df["hash"] = df["question"].apply(uuid_hash)

        df = df.set_index("hash")

        gt_dict = df.to_dict(orient="index")

        return gt_dict

    def _validate_metrics(
        self, metrics: List[MetricMetadata.RagasMetrics], eval_type: str
    ):
        """
        Validates if the metrics are supported for the evaluation method chosen.

        Args:
            metrics: List of metrics to validate.
            eval_type: The type of evaluation ("synthetic" or "ground_truth").
        """

        if eval_type == "ground_truth":
            for metric in metrics:
                if metric not in mm.ALLOWED_EVALS_GROUND_TRUTH:
                    raise ValueError(
                        f"Metric {metric} is not a valid metric '{eval_type}', please list one of {mm.ALLOWED_EVALS_GROUND_TRUTH}."
                    )
        elif eval_type == "synthetic":
            for metric in metrics:
                if metric not in mm.ALLOWED_EVALS_SYNTHETHIC:
                    raise ValueError(
                        f"Metric {metric} is not a valid metric for evaluation method '{eval_type}', please list one of {mm.ALLOWED_EVALS_SYNTHETHIC}."
                    )
        else:
            raise ValueError(
                f"Only evaluation type 'synthetic' or 'ground_truth' are supported, not {eval_type}"
            )

    def _evaluate(
        self,
        data: dict,
        metrics: List[MetricMetadata.RagasMetrics],
        output_eval_dataset: bool,
        llm: AzureChatOpenAI,
    ) -> pd.DataFrame:
        """
        Evaluates the RAG module on the questions in the dataset.

        Args:
            data: The dataset to evaluate.
            metrics: The list of metrics to evaluate on.
            output_eval_dataset: Whether to save the evaluation results.
            llm: The Azure LLM model used for evaluation.

        Returns:
            Evaluation results as a DataFrame
        """
        retrieved_data = self._run_rag(questions=data)

        rag_evaluation_metrics = RagEvaluationMetrics(
            data=retrieved_data, output_folder=self.output_folder, llm=llm
        )

        eval_data = rag_evaluation_metrics.run(
            metrics=metrics,
            output_eval_dataset=output_eval_dataset,
        )

        return eval_data

    def evaluate_ground_truth(
        self,
        metrics: List[MetricMetadata.RagasMetrics],
        ground_truth_filepath: Path,
        output_eval_dataset: bool = True,
        evaluative_llm: AzureChatOpenAI = None,
    ) -> pd.DataFrame:
        """
        Evaluates the RAG module on a groudn truth dataset with question answer pairs.

        Args:
            metrics: The list of metrics to evaluate on.
            ground_truth_filepath: The file path to the ground_truth JSON file with question, answer pairs.
            output_eval_dataset: Whether to save the evaluation results.
            llm: The Azure LLM model used for evaluation.

        Returns:
           Evaluation results as a DataFrame
        """

        self._validate_metrics(metrics=metrics, eval_type="ground_truth")

        logger.info(f"Reading ground truth file from {ground_truth_filepath}.")

        data = self._format_ground_truth_dataset(
            ground_truth_filepath=ground_truth_filepath
        )

        eval_data = self._evaluate(data, metrics, output_eval_dataset, evaluative_llm)

        return eval_data

    def evaluate_synthetic(
        self,
        metrics: List[MetricMetadata.RagasMetrics],
        dataset_size: int,
        output_eval_dataset: bool = True,
        generative_llm: AzureChatOpenAI = None,
        evaluative_llm: AzureChatOpenAI = None,
    ) -> pd.DataFrame:
        """
        Evaluates the RAG module on a groudn truth dataset with question answer pairs.

        Args:
            metrics: The list of metrics to evaluate on.
            dataset_size: Number of synthetic context, question pairs to be generated.
            output_eval_dataset: Whether to save the evaluation results.
            generative_llm: The Azure LLM model used for generating synthetic data.
            evaluative_llm: The Azure LLM model used for evaluation.

        Returns:
            Evaluation results as a DataFrame
        """
        self._validate_metrics(metrics=metrics, eval_type="synthetic")

        generator = ContextQuestionDatasetGenerator(llm=generative_llm)

        data = generator.generate(
            chunked_documents=self.documents,
            dataset_size=dataset_size,
        )

        eval_data = self._evaluate(data, metrics, output_eval_dataset, evaluative_llm)

        return eval_data
