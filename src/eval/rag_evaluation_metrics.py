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


import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset
from langchain.chat_models import AzureChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLM
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.metrics.base import MetricWithLLM

from src.constants import MetricMetadata
from src.utils import create_Azure_LLM, initialize_logger

logger = initialize_logger(__name__)
mm, rm, hm = (
    MetricMetadata.RagasMetrics(),
    MetricMetadata.RagasMetadata(),
    MetricMetadata.HitRateMetadata(),
)


class RagEvaluationMetrics:
    """
    Evaluation class that:
        1. Evaluates a dataset consisting of ground truth context, question, RAG-generated answer, and top-k documents on:
            1. Hit rate
            2. Context precision
            3. Context recall
            4. Faithfulness
            5. Answer relevancy

    Args:
        data: Dataset in the form of a dictionary.
            dict
            {
                question_id: uuid {
                    question: str,
                    context_id: uuid.int,
                    answer: str,
                    context: str,
                    contexts: {uuid.int: str, ...}
                }, ...
            }
        llm: The language model to use for evaluation. Defaults to None, in which case a new Azure LLM with a temperature of 0.2 will be created.
        output_folder: The folder path where evaluation runs are stored. Defaults to "data\evaluation_runs".
    Returns:
        pd.DataFrame: Dataframe with evaluation scores per ground truth context, question, RAG-generated answer, and retrieved document pair on provided evaluation metrics.
        dict: Dictionary with aggregated evaluation scores on inputted evaluation metrics.
    """

    def __init__(
        self,
        data: dict,
        llm: AzureChatOpenAI = None,
        output_folder="data\evaluation_runs",
    ):
        self.data = data

        if llm is None:
            self.llm = create_Azure_LLM(temperature=0.2)
        else:
            self.llm = llm
        self.ragas_model = LangchainLLM(llm=self.llm)
        self.ragas_metric_dict = {
            mm.CONTEXT_PRECISION: self.initialize_ragas_metric(ContextPrecision),
            mm.CONTEXT_RECALL: self.initialize_ragas_metric(ContextRecall),
            mm.FAITHFULNESS: self.initialize_ragas_metric(Faithfulness),
            mm.ANSWER_RELEVANCY: self.initialize_ragas_metric(AnswerRelevancy),
        }
        self.output_folder = output_folder

    def _to_hf_dataset(self) -> Dataset:
        """
        Transform the JSON file into HuggingFace Dataset format for RAGAS evaluation.

        Returns:
            HuggingFace dataset.
        """
        # get all keys from the subdicts in the dictionary to find the provided evaluation strategy ground_truth vs no ground truth
        eval_columns = list(self.data[next(iter(self.data))].keys())

        logger.info("transforming dataset to Huggingface format")

        transformed_dict = {
            rm.QUESTION: list(subdict[rm.QUESTION] for subdict in self.data.values()),
            rm.GENERATED_ANSWER: list(
                subdict[rm.GENERATED_ANSWER] for subdict in self.data.values()
            ),
            rm.CONTEXTS: [
                list(subdict[rm.CONTEXTS].values()) for subdict in self.data.values()
            ],
        }

        if rm.GROUND_TRUTH_ANSWER in eval_columns:
            transformed_dict[rm.GROUND_TRUTH_ANSWER] = [
                [subdict[rm.GROUND_TRUTH_ANSWER]] for subdict in self.data.values()
            ]

        hf_dataset = Dataset.from_dict(transformed_dict)

        return hf_dataset

    def initialize_ragas_metric(self, metric: MetricWithLLM):
        """Initialize Ragas metric."""
        initialized_metric = metric(name=str(metric.__name__))
        initialized_metric.llm = self.ragas_model
        initialized_metric.init_model()

        return initialized_metric

    def _run_hit_rate_metric(self) -> pd.DataFrame:
        """
        Calculation of the hit rate score.

        Returns:
           DataFrame with hit rate scores.
        """
        logger.info("calculating hit rate score")

        eval_results = [
            {
                hm.HIT_RATE: (
                    1
                    if self.data[question_id]["context_id"]
                    in self.data[question_id][rm.CONTEXTS].keys()
                    else 0
                ),
                hm.RETRIEVED: list(self.data[question_id][rm.CONTEXTS].keys()),
                hm.EXPECTED: self.data[question_id]["context_id"],
                "question_id": question_id,
            }
            for question_id in self.data.keys()
        ]

        hit_rate_df = pd.DataFrame(eval_results)

        return hit_rate_df

    def _run_ragas_metrics(
        self, metrics: List[MetricWithLLM], dataset: Dataset
    ) -> pd.DataFrame:
        """
        Executes the RAGAS evaluation for the provided RAGAS metrics.

        Args:
            metrics: List of RAGAS metrics to evaluate.
            dataset: The dataset to evaluate.

        Returns:
            DataFrame with RAGAS evaluation results.
        """
        result = evaluate(dataset=dataset, metrics=metrics)
        ragas_df = result.to_pandas()
        ragas_df["question_id"] = list(self.data.keys())

        return ragas_df

    def _run_metrics(self, metrics: List) -> pd.DataFrame:
        """
        Runs the evaluation for the provided metrics.

        Args:
            metric: List of metrics to evaluate.

        Returns:
            DataFrame with evaluation results.
        """
        hf_dataset = self._to_hf_dataset()

        selected_ragas_metrics = [
            self.ragas_metric_dict[metric]
            for metric in metrics
            if metric in self.ragas_metric_dict.keys()
        ]
        selected_hitrate_metric = [metric for metric in metrics if metric == "hitrate"]

        if selected_ragas_metrics and selected_hitrate_metric:
            return self._run_ragas_metrics(
                metrics=selected_ragas_metrics,
                dataset=hf_dataset,
            ).merge(self._run_hit_rate_metric(), on="question_id", how="inner")

        if selected_hitrate_metric:
            return self._run_hit_rate_metric()

        if selected_ragas_metrics:
            return self._run_ragas_metrics(
                metrics=selected_ragas_metrics,
                dataset=hf_dataset,
            )

    def _save(self, data: dict) -> None:
        """Saves dict to JSON."""

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        output_file_name = os.path.join(
            self.output_folder, f"evaluation_data_{timestamp}.json"
        )

        data = [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in item.items()}
            for item in data
        ]

        with open(output_file_name, "w") as f:
            json.dump(data, f, indent=4)

    def run(self, metrics: List[str], output_eval_dataset: bool) -> pd.DataFrame:
        """
        Runs the evaluation and returns a dataset or evaluation scores and saves to CSV or JSON.

        Args:
            metrics: List of metrics to evaluate.
            output_eval_dataset: Whether to output the evaluation dataset.

        Returns:
            Evaluation dataset.
        """
        eval_dataset = self._run_metrics(metrics=metrics)

        average_scores = {metric: eval_dataset[metric].mean() for metric in metrics}
        logger.info(average_scores)

        if output_eval_dataset:
            eval_dict = eval_dataset.to_dict("records")
            self._save(data=eval_dict)

        return eval_dataset
